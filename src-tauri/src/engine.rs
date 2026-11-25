//! The transcription engine: a speech-to-text backend, then paragraph segmentation.
//!
//! - **Parakeet** (default): NVIDIA Parakeet TDT 0.6B v3 on the Neural Engine via FluidAudio.
//!   About 300x real time on an M4, 25 European languages. Loaded at startup.
//! - **Whisper**: Whisper large-v3-turbo on Candle (Metal). Much slower, but covers ~100 languages.
//!   Loaded (and downloaded) only the first time someone picks it.
//!
//! See BENCHMARKS.md for how these were chosen.

use crate::download;
use crate::parakeet::Parakeet;
use crate::segment::{self, Embedder, Paragraph, Segment};
use crate::whisper::WhisperEngine;
use anyhow::{bail, Result};
use candle_core::Device;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Parakeet,
    Whisper,
}

impl Backend {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "parakeet" => Ok(Self::Parakeet),
            "whisper" => Ok(Self::Whisper),
            _ => bail!("unknown engine {s:?}"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Parakeet => "Parakeet v3 · Neural Engine",
            Self::Whisper => "Whisper large-v3-turbo · Metal GPU",
        }
    }
}

pub struct Transcript {
    pub text: String,
    pub paragraphs: Vec<Paragraph>,
    pub language: Option<String>,
    pub cancelled: bool,
    /// Speech-to-text time, excluding paragraph segmentation.
    pub asr_time: Duration,
    pub paragraph_time: Duration,
}

pub struct Engine {
    device: Device,
    parakeet: Parakeet,
    whisper: Option<WhisperEngine>,
    embedder: Option<Embedder>,
}

impl Engine {
    /// Load Parakeet and the paragraph model, downloading them on first run.
    /// `status(message, progress_fraction)`.
    pub fn load(status: download::StatusFn) -> Result<Self> {
        let device = Device::new_metal(0).unwrap_or(Device::Cpu);
        let parakeet = Parakeet::load(status)?;
        // Optional: small sentence-embedding model for paragraph segmentation (~90 MB).
        status("Loading paragraph model…".into(), None);
        let embedder = match Embedder::load(&device, status) {
            Ok(e) => Some(e),
            Err(e) => {
                eprintln!("paragraph model unavailable, using pauses only: {e:#}");
                None
            }
        };
        Ok(Self {
            device,
            parakeet,
            whisper: None,
            embedder,
        })
    }

    pub fn is_loaded(&self, backend: Backend) -> bool {
        backend == Backend::Parakeet || self.whisper.is_some()
    }

    /// Load a backend that isn't loaded yet (only Whisper is ever lazy).
    pub fn ensure_loaded(&mut self, backend: Backend, status: download::StatusFn) -> Result<()> {
        if backend == Backend::Whisper && self.whisper.is_none() {
            self.whisper = Some(WhisperEngine::load(&self.device, status)?);
        }
        Ok(())
    }

    /// Transcribe 16 kHz mono PCM. `on_update(fraction_done, text_so_far)`; `cancelled()` is polled
    /// while it runs and stops early, returning whatever was transcribed so far.
    pub fn transcribe(
        &mut self,
        backend: Backend,
        pcm: &[f32],
        cancelled: &dyn Fn() -> bool,
        on_update: &mut dyn FnMut(f32, &str),
    ) -> Result<Transcript> {
        let started = Instant::now();
        let (segments, language, was_cancelled): (Vec<Segment>, _, _) = match backend {
            Backend::Parakeet => {
                let out = self.parakeet.transcribe(pcm, cancelled, on_update)?;
                (out.segments, None, out.cancelled)
            }
            Backend::Whisper => {
                let Some(whisper) = self.whisper.as_mut() else {
                    bail!("Whisper isn't loaded");
                };
                let out = whisper.transcribe(pcm, cancelled, on_update)?;
                (out.segments, out.language, out.cancelled)
            }
        };
        let asr_time = started.elapsed();

        let started = Instant::now();
        let paragraphs = segment::paragraphs(&segments, self.embedder.as_ref());
        let paragraph_time = started.elapsed();
        let text = paragraphs
            .iter()
            .map(|p| p.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        Ok(Transcript {
            text,
            paragraphs,
            language,
            cancelled: was_cancelled,
            asr_time,
            paragraph_time,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_names_round_trip() {
        assert_eq!(Backend::parse("parakeet").unwrap(), Backend::Parakeet);
        assert_eq!(Backend::parse("whisper").unwrap(), Backend::Whisper);
        assert!(Backend::parse("vosk").is_err());
    }
}
