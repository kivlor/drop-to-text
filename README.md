# Drop to Text

A tiny Mac app: drop an audio file on it and get a transcript. Everything runs locally on your Mac. Nothing is uploaded.

- **Speech-to-text:** [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo), running in pure Rust on [Candle](https://github.com/huggingface/candle) with the Metal GPU
- **Paragraphs:** the transcript is split where the topic changes or the speaker pauses, with a timestamp on each paragraph. Topic changes are detected with [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) sentence embeddings.
- **Audio formats:** m4a/AAC (iPhone voice memos), ALAC, mp3, wav, flac, ogg/vorbis, aiff and caf are decoded in Rust with [Symphonia](https://github.com/pdeljanov/Symphonia). Anything else, such as Opus voice notes or video files, works too if `ffmpeg` is installed (`brew install ffmpeg`).
- **App shell:** [Tauri 2](https://tauri.app) with a plain HTML/CSS/JS interface. No bundler and no framework.

On an Apple Silicon Mac, an 8½-minute recording takes about 1½ minutes to transcribe.

## Usage

- Drop one or more audio files anywhere on the window, or click the drop area to pick files.
- The transcript appears as it's produced. When it's done, **Copy** copies the text as plain paragraphs (without the timestamps).
- **Cancel** stops the current transcription. **Clear** goes back to the start screen. Dropping a new file replaces whatever is showing.

On first launch the app downloads the models, about 1.6 GB in total, to
`~/Library/Application Support/com.droptotext.app/models`. If you already have them in a Hugging Face cache (`~/.cache/huggingface/hub`), it uses those instead.

## Development

Requirements: macOS on Apple Silicon, Rust (stable) and Node.js.

```sh
npm install        # installs the Tauri CLI
npm run dev        # run the app with live reload
npm run build      # build "Drop to Text.app" and a .dmg into src-tauri/target/release/bundle/
```

To test transcription without the GUI:

```sh
cd src-tauri
cargo run --release --example cli -- path/to/recording.m4a
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

The unit tests don't need the models, so they run quickly.

## How it works

```
ui/index.html            the interface (drag and drop, progress, results)
src-tauri/src/
  lib.rs                 Tauri commands: transcribe, cancel, model_status
  audio.rs               decode any file to 16 kHz mono (Symphonia + rubato, ffmpeg fallback)
  engine.rs              Whisper decoding loop
  segment.rs             sentence splitting and paragraph breaks
  download.rs            resumable model downloads with timeouts and retries
```

The Whisper decoding follows OpenAI's reference implementation:

- Audio is processed in 30-second windows. Each window resumes from the last complete timestamp, so words that cross a window boundary aren't lost.
- It applies Whisper's timestamp rules, and retries a window at higher temperatures when the output looks like a repetition loop.
- Silent windows are skipped before they reach the model. Whisper tends to invent text like "Thank you." on silence, and the turbo model's own no-speech check isn't reliable.

Paragraph breaks are scored with a TextTiling-style approach. The app compares the embeddings of the few sentences before and after each sentence boundary, and adds weight where there's a pause. The strongest boundaries become breaks, with a minimum and maximum length per paragraph.
