//! Headless test: cargo run --release --example cli -- <audio file>
use drop_to_text_lib::{audio, engine::Engine};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: cli <audio file>");
    let t0 = Instant::now();
    let mut engine = Engine::load(&mut |m, p| eprintln!("[load] {m} {p:?}"))?;
    eprintln!("loaded in {:.1}s on {}", t0.elapsed().as_secs_f32(), engine.device_name());
    let t1 = Instant::now();
    let pcm = audio::load_audio(path.as_ref())?;
    eprintln!("decoded {:.1}s of audio in {:.2}s", pcm.len() as f32 / 16000.0, t1.elapsed().as_secs_f32());
    let t2 = Instant::now();
    // CANCEL_AFTER=<secs> simulates the user pressing Cancel.
    let cancel_after: Option<f32> = std::env::var("CANCEL_AFTER").ok().and_then(|v| v.parse().ok());
    let cancel = || cancel_after.is_some_and(|c| t2.elapsed().as_secs_f32() > c);
    let t = engine.transcribe(&pcm, &cancel, &mut |f, _| eprintln!("  {:.0}%", f * 100.0))?;
    eprintln!("transcribed in {:.1}s, language={:?}, cancelled={}", t2.elapsed().as_secs_f32(), t.language, t.cancelled);
    for p in &t.paragraphs {
        println!("[{:>5.1}s] {}\n", p.start, p.text);
    }
    Ok(())
}
