# Drop to Text

A tiny Mac app: drop an audio file on it and get a transcript. Everything runs locally on your Mac. Nothing is uploaded.

- **Speech-to-text:** [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) on the Apple Neural Engine, through [FluidAudio](https://github.com/FluidInference/FluidAudio) (Core ML). It covers 25 European languages and detects the language automatically. For other languages you can switch to [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo), which runs in pure Rust on [Candle](https://github.com/huggingface/candle) with the Metal GPU. Whisper is much slower but covers about 100 languages.
- **Paragraphs:** the transcript is split where the topic changes or the speaker pauses, with a timestamp on each paragraph. Topic changes are detected with [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) sentence embeddings.
- **Audio formats:** m4a/AAC (iPhone voice memos), ALAC, mp3, wav, flac, ogg/vorbis, aiff and caf are decoded in Rust with [Symphonia](https://github.com/pdeljanov/Symphonia). Anything else, such as Opus voice notes or video files, works too if `ffmpeg` is installed (`brew install ffmpeg`).
- **App shell:** [Tauri 2](https://tauri.app) with a plain HTML/CSS/JS interface. No bundler and no framework.

On an M4 Mac, an 8½-minute recording takes about 2 seconds to transcribe with Parakeet, against about 1½ minutes with Whisper. See [BENCHMARKS.md](BENCHMARKS.md) for the measurements and how the engine was chosen.

## Usage

- Drop one or more audio files anywhere on the window, or click the drop area to pick files.
- The transcript appears as it's produced. When it's done, **Copy** copies the text as plain paragraphs (without the timestamps).
- **Cancel** stops the current transcription. **Clear** goes back to the start screen. Dropping a new file replaces whatever is showing.
- The menu next to the status picks the engine: **Parakeet** (the default) or **Whisper**. The app remembers your choice.

On first launch the app downloads the models, about 560 MB in total, to
`~/Library/Application Support/com.droptotext.app/models`, then spends about 15 seconds preparing Parakeet for your Mac's Neural Engine. Both happen only once; after that the model loads in under a second. Whisper (another 1.6 GB) is downloaded only the first time you pick it. Models already in a Hugging Face cache (`~/.cache/huggingface/hub`) or FluidAudio's cache are reused instead of downloaded again.

## Development

Requirements: macOS 14 or later on Apple Silicon, Xcode or the Command Line Tools (Swift 6.1 or later), Rust (stable) and Node.js. The first build also fetches FluidAudio from GitHub and compiles it, which takes a minute or two.

```sh
npm install        # installs the Tauri CLI
npm run dev        # run the app with live reload
npm run build      # build "Drop to Text.app" and a .dmg into src-tauri/target/release/bundle/
```

To test transcription without the GUI:

```sh
cd src-tauri
cargo run --release --example cli -- path/to/recording.m4a
# --whisper uses the Whisper engine; list a file twice to see warm-model timings
# CANCEL_AFTER=5 simulates pressing Cancel after 5 seconds
```

### Checks

These are the same checks CI runs on every push and pull request:

```sh
cd src-tauri
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cd .. && npm run check:ui   # syntax-checks the inline script in ui/index.html
```

The unit tests don't need the models, so they run quickly. (The first `cargo` command builds the Swift bridge, so it takes longer.)

## How it works

```
ui/index.html            the interface (drag and drop, engine picker, progress, results)
src-tauri/src/
  lib.rs                 Tauri commands: transcribe, cancel, model_status
  audio.rs               decode any file to 16 kHz mono (Symphonia + rubato, ffmpeg fallback)
  engine.rs              picks the backend, then splits the result into paragraphs
  parakeet.rs            Rust side of the Parakeet bridge
  whisper.rs             Whisper decoding loop
  segment.rs             sentence splitting and paragraph breaks
  download.rs            resumable model downloads with timeouts and retries
src-tauri/swift/         Swift package: FluidAudio behind a small C API (ptb_load, ptb_transcribe)
src-tauri/build.rs       builds the Swift package and links it statically
```

Parakeet runs through FluidAudio, a Swift library. `build.rs` compiles `src-tauri/swift/` into a static library with `swift build` and links it into the app. Only FluidAudio's speech-to-text part is built (no TTS or diarisation). The app links only the Swift runtime that ships with macOS, so there's nothing extra to bundle. FluidAudio splits long audio into overlapping windows, decodes several at once on the Neural Engine and merges the seams. It returns word timings, and the paragraph step uses them to find pauses.

The Whisper decoding follows OpenAI's reference implementation:

- Audio is processed in 30-second windows. Each window resumes from the last complete timestamp, so words that cross a window boundary aren't lost.
- It applies Whisper's timestamp rules, and retries a window at higher temperatures when the output looks like a repetition loop.
- Silent windows are skipped before they reach the model. Whisper tends to invent text like "Thank you." on silence, and the turbo model's own no-speech check isn't reliable.

Paragraph breaks are scored with a TextTiling-style approach. The app compares the embeddings of the few sentences before and after each sentence boundary, and adds weight where there's a pause. The strongest boundaries become breaks, with a minimum and maximum length per paragraph.

## Licences

- Parakeet TDT 0.6B v3 weights: © NVIDIA, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Converted to Core ML by FluidInference.
- FluidAudio: Apache 2.0.
- Whisper large-v3-turbo weights: © OpenAI, MIT.
- all-MiniLM-L6-v2: Apache 2.0.
