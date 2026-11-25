//! Headless test and benchmark:
//!   cargo run --release --example cli -- [--whisper] <audio file> [more files...]
//! Prints the transcript to stdout and a timing breakdown to stderr. With several files the model
//! stays loaded, so the later files show warm-model times.
use drop_to_text_lib::{
    audio,
    engine::{Backend, Engine},
};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let backend = if let Some(i) = args.iter().position(|a| a == "--whisper") {
        args.remove(i);
        Backend::Whisper
    } else {
        Backend::Parakeet
    };
    assert!(!args.is_empty(), "usage: cli [--whisper] <audio file>...");

    let t0 = Instant::now();
    let mut engine = Engine::load(&mut |m, p| eprintln!("[load] {m} {p:?}"))?;
    engine.ensure_loaded(backend, &mut |m, p| eprintln!("[load] {m} {p:?}"))?;
    eprintln!(
        "model load: {:.2}s ({})",
        t0.elapsed().as_secs_f32(),
        backend.label()
    );

    for path in &args {
        let t1 = Instant::now();
        let pcm = audio::load_audio(path.as_ref())?;
        let audio_secs = pcm.len() as f32 / audio::TARGET_RATE as f32;
        let decode = t1.elapsed().as_secs_f32();
        // CANCEL_AFTER=<secs> simulates the user pressing Cancel.
        let cancel_after: Option<f32> = std::env::var("CANCEL_AFTER")
            .ok()
            .and_then(|v| v.parse().ok());
        let t2 = Instant::now();
        let cancel = || cancel_after.is_some_and(|c| t2.elapsed().as_secs_f32() > c);
        let t = engine.transcribe(backend, &pcm, &cancel, &mut |f, _| {
            eprint!("\r  {:.0}%", f * 100.0)
        })?;
        let total = decode + t2.elapsed().as_secs_f32();
        eprintln!(
            "\r{path}\n  audio {audio_secs:.1}s | read+resample {decode:.2}s | speech-to-text {:.2}s | \
             paragraphs {:.2}s | total {total:.2}s ({:.0}x real time) | language={:?} cancelled={}",
            t.asr_time.as_secs_f32(),
            t.paragraph_time.as_secs_f32(),
            audio_secs / total,
            t.language,
            t.cancelled
        );
        for p in &t.paragraphs {
            println!("[{:>5.1}s] {}\n", p.start, p.text);
        }
    }
    Ok(())
}
