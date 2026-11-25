# Transcription benchmarks

How Drop to Text went from about 90 seconds to about 2 seconds for an 8½-minute recording, and why it now uses NVIDIA Parakeet on the Neural Engine (through [FluidAudio](https://github.com/FluidInference/FluidAudio)) instead of Whisper on Candle.

All numbers are from one **Apple M4 (10-core, 24 GB), macOS 26.7**, using release builds, measured in November 2026.

## Summary

| | Before (Whisper turbo on Candle) | After (Parakeet v3 on FluidAudio) |
|---|---:|---:|
| 8½-min m4a, warm model, end to end | **~95 s** | **2.0 s** |
| of which speech-to-text | 91.3 s | 1.3 s |
| Real-time factor (end to end) | 5.4× | ~255× |
| Model load, every launch | 4.4 s | 0.13 to 0.5 s |
| Model load, first launch ever | 1.6 GB download | 470 MB download + 14.5 s one-off Neural Engine compile |
| Peak memory (process footprint) | 6.9 GB | 0.7 GB |
| Synthetic clip WER (true reference, clean / noisy) | 1.37% / 1.37% | 1.71% / 1.71% |
| Hallucinated text on a 10-minute near-silent recording | "Thank you." loops | none |

**Target met:** under 15 s for an 8-minute recording with a warm model, and under 10 s, with comparable accuracy. Whisper is still available as an option for languages Parakeet doesn't cover.

## Test audio

- **call**: 8 min 30 s two-person call, AAC m4a (44.1 kHz stereo). Real conversation with fillers, cross-talk and long pauses. This is the recording the original "about 1½ minutes" figure came from.
- **env10**: 10 min 27 s ambient room recording that is almost entirely silence, with faint background noise and distant voices. Used to test for hallucinations.
- **env18**: 18 min 35 s far-field office recording: several speakers, a noisy room, overlapping speech. The hardest file.
- **synth**: a 95 s scripted renovation-planning dialogue (293 words) read by three macOS voices: Karen (Australian), Daniel (British) and Moira (Irish), with numbers, place names and Australian vocabulary ("arvo", "tradies", "skip bin", Parramatta, Wollongong). Tested clean and band-limited to 200–3400 Hz with pink noise. It's the only file with a true reference transcript.

The recordings are private and aren't in the repository. Accuracy is WER after OpenAI's English text normaliser (`whisper_normalizer`, `jiwer`).

## 1. The original implementation

Profiled on the call, 18 windows of 30 s, 2,030 tokens decoded, with no temperature fallbacks:

| Stage | Time |
|---|---:|
| Audio decode + resample (Symphonia, rubato) | 0.42 s |
| Mel spectrogram | 0.9 s |
| Whisper encoder (Candle, Metal) | 32.3 s: **1.8 s per 30 s window** |
| Whisper decoder (Candle, Metal) | 68.0 s: **33 ms per token** |
| Paragraph segmentation (MiniLM embeddings) | 0.9 s |

Why it was slow:

- **No self-attention KV cache.** Candle's Whisper only caches cross-attention keys and values, so every decoding step pushed the whole token prefix through the decoder again. That makes decoding quadratic in tokens per window.
- **F32 weights and activations** (`candle_transformers::models::whisper::DTYPE`). Switching to F16 needs a fork of the model code, because the positional embedding is built in F32.
- **A GPU-to-CPU sync for every token**, to read back 51,866 logits and apply the timestamp rules on the CPU.
- **Candle's Metal kernels are slow for this model.** The same Whisper-turbo encoder takes 1.2 s for all 18 windows on GGML/Metal, against 32 s on Candle.
- It *was* already a release build, used the Metal GPU, loaded the model once and kept it loaded. Debug-versus-release and model loading weren't the problem.

## 2. Alternatives evaluated

Times are speech-to-text only, on the call, with a warm model.

| Engine | Model | Time | Notes |
|---|---|---:|---|
| Candle (current) | Whisper large-v3-turbo F32 | 91.3 s | baseline |
| [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (GGML, Metal) | Whisper large-v3-turbo Q8_0 | 27.7 s | encoder 1.2 s, **decoder 26.4 s** |
| transcribe.cpp | Whisper large-v3-turbo F16 | 31.4 s | |
| [parakeet-rs](https://crates.io/crates/parakeet-rs) 0.4.4 (ONNX Runtime, CPU) | Parakeet TDT v3 FP32 | 20.7 to 24.6 s | 10.6 GB peak; files must be split at about 4 min |
| parakeet-rs (ONNX Runtime, Core ML EP) | Parakeet TDT v3 FP32 | failed | ballooned to a 60 GB footprint without finishing |
| transcribe.cpp (GGML, Metal) | Parakeet TDT v2 Q8_0 | 9.2 s | encoder 6.5 s, TDT decoder 2.7 s on CPU; **dropped a 20-word passage** |
| transcribe.cpp | Parakeet TDT v3 Q8_0 | 12.4 s | |
| FluidAudio CLI (Core ML, Neural Engine) | Parakeet TDT v2 | 1.55 s | |
| FluidAudio CLI | Parakeet TDT v3 | 1.71 s | |
| FluidAudio CLI | Parakeet Ultra (post-trained v3) | 1.73 s | |
| **Drop to Text (FluidAudio v3 via Swift bridge)** | **Parakeet TDT v3** | **1.3 to 1.5 s** | output identical to the FluidAudio CLI |

Not run: MLX implementations (Python-first, or would need a port to `mlx-rs`), a pure-Candle Parakeet port (no existing implementation, and Candle's Metal performance was the bottleneck above), and parakeet-rs's burn/Metal backend (same FP32 weights; the ONNX numbers already ruled the crate out).

**Optimised Whisper isn't enough.** Even on a fast runtime, Whisper's autoregressive decoder takes about 13 ms per token and about 2,000 tokens for this file, so its floor is around 28 s. Getting Whisper under 15 s would need *batched* decoding of all 30 s windows at once (as in insanely-fast-whisper) on a runtime with F16, a KV cache and flash attention. Candle has none of these today, and batching loses the seek-to-last-timestamp seam handling, which brings back missing-words-at-boundaries risk. Parakeet's TDT decoder skips frames instead of generating every token one at a time, so it doesn't have this floor.

### Accuracy

**Synthetic clip, against the true script (WER, 293 words):**

| | clean | noisy |
|---|---:|---:|
| Whisper turbo, Candle (old app) | 1.37% | 1.37% |
| Whisper turbo, GGML | 1.02% | 1.37% |
| Parakeet v2, FluidAudio | 3.07% | 2.05% |
| **Parakeet v3, FluidAudio (new app)** | **1.71%** | **1.71%** |
| Parakeet Ultra, FluidAudio | 1.71% | 2.39% |
| Parakeet v2, transcribe.cpp | 1.37% | 1.71% |

Every engine is within 1 to 2 words of the others on 293. The differences are number formatting and one or two place names.

**Real recordings, word disagreement with the old app.** This is not accuracy: there's no human reference, so it only shows how far each engine is from what users get today.

| | call | env18 |
|---|---:|---:|
| Whisper turbo, GGML | 5.0% | 10.3% |
| Parakeet v2, FluidAudio | 7.1% | 16.2% |
| **Parakeet v3, FluidAudio** | **7.7%** | **17.5%** |
| Parakeet Ultra, FluidAudio | 7.3% | 18.0% |
| Parakeet v2, transcribe.cpp | 7.3% | 16.6% |

Breaking down the 113 differing words between the old app and the new one on the call: 7% are different spellings of one product name, 12% are fillers and backchannels ("yeah", "yep", "um") that Whisper tends to drop, and the other ~80% are real wording differences in fast, overlapping speech. Reading those by hand, neither engine wins consistently: Parakeet v3 is more plausible in some phrases where Whisper adds or mishears words, and Whisper is better in others that v3 garbles. (Parakeet v2 also caught a sentence both Whisper and v3 missed.)

On env18 (far-field, noisy, overlapping speakers) neither engine is reliable and they disagree a lot. Whisper produces more fluent text but invents some of it; Parakeet sometimes stutters on repeated words ("the the the"). Treat transcripts of recordings like this with care whichever engine you use.

**Silence and hallucination (env10, about 10 minutes of near-silence):**

- Old app (Whisper on Candle, even with its energy gate): "Thank you." five times, "I'll see you next time", "\*clicks\*", "\*cough\*".
- Whisper on GGML: "Thank you." about twenty times.
- Parakeet v2: about 20 low-confidence words (0.16 to 0.6) of distant chatter.
- Parakeet Ultra: about 20 words of nonsense.
- **Parakeet v3: nothing**, so the app shows "(no speech detected)".

**Completeness.** I checked Parakeet v3's word timings for gaps against Whisper's output and the signal level. The only gaps over 5 s in the call are three stretches of about 31 s where the speakers were silent (−50 dBFS against about −25 dBFS for speech). Nothing was dropped. FluidAudio logs `E5RT ... zero shape error` once per run; it's a Core ML shape-inference warning that FluidAudio already works around, and it doesn't affect the output.

## 3. Decision

**Parakeet TDT 0.6B v3 on FluidAudio (Core ML, Neural Engine), through a small Swift bridge.**

- **Speed:** about 6× faster than the next-best Parakeet runtime and about 20× faster than the best Whisper runtime. Encoding runs on the Neural Engine, which leaves the GPU and CPU mostly idle.
- **Completeness:** the only Parakeet runtime tested that didn't drop speech. It handles long audio itself (overlapping windows decoded in parallel, then a token-level seam merge), and returns word timings, punctuation and capitalisation.
- **v3 rather than v2:** English accuracy here was on a par (better on the synthetic clip, slightly further from Whisper on the real ones); it was the only variant with no hallucinations on near-silence; and it auto-detects 25 European languages instead of English only. FluidAudio recommends Ultra over v3, but on this audio Ultra wasn't more accurate and it hallucinated on silence.
- **Whisper stays as an option** for languages outside Parakeet's 25 (Chinese, Vietnamese, Arabic, Hindi, Japanese, Korean and others). It costs little to keep: Candle is still needed for the paragraph embedder, and the Whisper code is unchanged. It's only downloaded and loaded the first time someone picks it.

Trade-offs accepted:

- **macOS 14 minimum** (was 13). FluidAudio requires it.
- **A Swift build step.** `build.rs` runs `swift build` on `src-tauri/swift/`, about 90 s from clean, and links the static library. Xcode or the Command Line Tools with Swift 6.1+ is needed, and the build fetches FluidAudio from GitHub at a pinned commit. The app links only the Swift runtime that ships with macOS, so nothing extra goes in the bundle. The app is 22 MB.
- **No language label** for Parakeet transcripts. FluidAudio doesn't report the language it detected.
- **Cancel** stops within about 50 ms of polling plus the window in flight. A cancelled job keeps the windows that had finished.

## 4. Final results (the app)

`cargo run --release --example cli -- <files>` prints this breakdown. These are three runs of the original m4a in one process, with the model loaded once:

| Stage | Time |
|---|---:|
| Read + decode + resample m4a | 0.39 s |
| Speech-to-text (Parakeet v3, Neural Engine) | 1.29 s |
| Paragraph segmentation | 0.33 s |
| **Total** | **2.0 s** (~255× real time) |

Other files: env10 (10.5 min) 2.5 s, env18 (18.6 min) 6.8 s, synth (1.6 min) 0.35 s.

Cold start:

| | Time |
|---|---:|
| First launch ever: download 470 MB | ~67 s here (depends on the connection) |
| First launch ever: compile for this Mac's Neural Engine | 14.5 s (once; Core ML caches it) |
| Every later launch: model load | 0.13 to 0.5 s |
| Fresh process to finished transcript of the call | ~2.2 s |

## Reproducing

```sh
cd src-tauri
cargo run --release --example cli -- recording.m4a recording.m4a   # second run = warm model
cargo run --release --example cli -- --whisper recording.m4a       # the Whisper backend
```

The comparison runs used FluidAudio's `fluidaudiocli transcribe` (commit `e2ea498`), transcribe.cpp's `transcribe-cli` (commit `cfa237c`, `handy-computer/*-gguf` models) and a small parakeet-rs 0.4.4 harness.
