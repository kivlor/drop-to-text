//! Whisper speech-to-text on Candle (pure Rust, Metal GPU on Apple Silicon).
//!
//! Long audio is transcribed the same way OpenAI's reference implementation does it: decode a
//! 30 s window with timestamps, then seek to the last complete timestamp so words that straddle
//! a window boundary aren't lost. Includes timestamp rules, no-speech detection and temperature
//! fallback to avoid the usual Whisper repetition loops.

use crate::segment::{self, Embedder, Paragraph, Segment};
use crate::{audio, download};
use anyhow::{anyhow, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, model::Whisper, Config};
use std::io::Write;
use tokenizers::Tokenizer;

pub const MODEL_ID: &str = "openai/whisper-large-v3-turbo";
const MAX_SAMPLED_TOKENS: usize = 224;
const MAX_INITIAL_TIMESTAMP: u32 = 50; // 1.0 s, in 20 ms timestamp units

pub struct Engine {
    model: Whisper,
    tokenizer: Tokenizer,
    device: Device,
    config: Config,
    mel_filters: Vec<f32>,
    sot: u32,
    transcribe: u32,
    eot: u32,
    no_speech: Option<u32>,
    ts_begin: u32,
    lang_tokens: Vec<u32>,
    suppress: Vec<bool>,
    rng: u64,
    embedder: Option<Embedder>,
}

pub struct Transcript {
    pub text: String,
    pub paragraphs: Vec<Paragraph>,
    pub language: Option<String>,
    pub cancelled: bool,
}

/// Returned (as an error) from the decode loop when the user cancels.
#[derive(Debug)]
struct Cancelled;
impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled")
    }
}
impl std::error::Error for Cancelled {}

struct Decoded {
    tokens: Vec<u32>, // sampled tokens, without prompt or EOT
    avg_logprob: f64,
    no_speech_prob: f64,
    compression_ratio: f64,
    hit_limit: bool,
}

impl Engine {
    /// Load (downloading on first run) the model. `status(message, progress_fraction)`.
    pub fn load(status: &mut dyn FnMut(String, Option<f32>)) -> Result<Self> {
        let device = Device::new_metal(0).unwrap_or(Device::Cpu);

        status("Checking model files…".into(), None);
        let config_path = download::fetch(MODEL_ID, "config.json", status)?;
        let tokenizer_path = download::fetch(MODEL_ID, "tokenizer.json", status)?;
        let weights_path = download::fetch(MODEL_ID, "model.safetensors", status)?;

        status("Loading model into memory…".into(), None);
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!("{e}"))?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], m::DTYPE, &device)? };
        let model = Whisper::load(&vb, config.clone())?;

        let tok = |s: &str| {
            tokenizer.token_to_id(s).ok_or_else(|| anyhow!("token {s} missing from tokenizer"))
        };
        let sot = tok(m::SOT_TOKEN)?;
        let transcribe = tok(m::TRANSCRIBE_TOKEN)?;
        let translate = tok(m::TRANSLATE_TOKEN)?;
        let eot = tok(m::EOT_TOKEN)?;
        let no_timestamps = tok(m::NO_TIMESTAMPS_TOKEN)?;
        let no_speech = m::NO_SPEECH_TOKENS.iter().find_map(|t| tokenizer.token_to_id(t));
        let ts_begin = tokenizer.token_to_id("<|0.00|>").unwrap_or(no_timestamps + 1);
        let lang_tokens: Vec<u32> = (sot + 1..translate).collect();

        let vocab = config.vocab_size;
        let mut suppress = vec![false; vocab];
        for &t in &config.suppress_tokens {
            if (t as usize) < vocab {
                suppress[t as usize] = true;
            }
        }
        // All special tokens between EOT and the first timestamp are never sampled.
        for t in (eot + 1)..ts_begin {
            suppress[t as usize] = true;
        }

        let mel_filters = audio::mel_filters(config.num_mel_bins);

        // Optional: small sentence-embedding model for paragraph segmentation (~90 MB).
        status("Loading paragraph model…".into(), None);
        let embedder = match Embedder::load(&device, status) {
            Ok(e) => Some(e),
            Err(e) => {
                eprintln!("paragraph model unavailable, using pauses only: {e:#}");
                None
            }
        };
        status("Ready".into(), Some(1.0));
        Ok(Self {
            model,
            tokenizer,
            device,
            config,
            mel_filters,
            sot,
            transcribe,
            eot,
            no_speech,
            ts_begin,
            lang_tokens,
            suppress,
            rng: 0x2545_f491_4f6c_dd1d,
            embedder,
        })
    }

    pub fn device_name(&self) -> &'static str {
        if self.device.is_metal() { "Metal GPU" } else { "CPU" }
    }

    /// Transcribe 16 kHz mono PCM. `on_update(fraction_done, text_so_far)`; `cancelled()` is
    /// polled between tokens and stops early, returning whatever was transcribed so far.
    pub fn transcribe(
        &mut self,
        pcm: &[f32],
        cancelled: &dyn Fn() -> bool,
        on_update: &mut dyn FnMut(f32, &str),
    ) -> Result<Transcript> {
        let content_frames = pcm.len() / m::HOP_LENGTH;
        // Pad with 30 s of silence so every window can be a full 3000 mel frames.
        let mut padded = Vec::with_capacity(pcm.len() + m::N_SAMPLES);
        padded.extend_from_slice(pcm);
        padded.resize(pcm.len() + m::N_SAMPLES, 0.0);
        let mel = m::audio::pcm_to_mel(&self.config, &padded, &self.mel_filters);
        let n_mels = self.config.num_mel_bins;
        let total_frames = mel.len() / n_mels;
        let mel = Tensor::from_vec(mel, (1, n_mels, total_frames), &self.device)?;

        let mut seek = 0usize;
        let mut language: Option<u32> = None;
        let mut segments: Vec<Segment> = Vec::new();
        let mut was_cancelled = false;
        let joined = |segs: &[Segment]| segs.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ");

        while seek + 10 < content_frames {
            if cancelled() {
                was_cancelled = true;
                break;
            }
            let seg_size = m::N_FRAMES.min(content_frames - seek);
            let window = &pcm[seek * m::HOP_LENGTH..(seek + seg_size) * m::HOP_LENGTH];
            if !has_voice(window) {
                // Whisper hallucinates ("Thank you.") on silence, and turbo's no-speech
                // probability is unreliable, so gate on signal energy first.
                seek += seg_size;
                on_update((seek as f32 / content_frames as f32).min(1.0), &joined(&segments));
                continue;
            }
            let mel_seg = mel.narrow(2, seek, m::N_FRAMES)?.contiguous()?;
            let features = self.model.encoder.forward(&mel_seg, true)?;

            let lang = match language {
                Some(l) => l,
                None => *language.insert(self.detect_language(&features)?),
            };
            let d = match self.decode_with_fallback(&features, lang, cancelled) {
                Ok(d) => d,
                Err(e) if e.is::<Cancelled>() => {
                    was_cancelled = true;
                    break;
                }
                Err(e) => return Err(e),
            };

            if d.no_speech_prob > m::NO_SPEECH_THRESHOLD && d.avg_logprob < m::LOGPROB_THRESHOLD {
                seek += seg_size;
            } else {
                let (text_end, advance) = self.split_at_last_timestamp(&d.tokens, seg_size);
                let offset = seek as f32 / 100.0;
                let window_end = (seek + seg_size) as f32 / 100.0;
                segments.extend(self.timed_segments(&d.tokens[..text_end], offset, window_end)?);
                seek += advance;
            }
            on_update((seek as f32 / content_frames as f32).min(1.0), &joined(&segments));
        }

        let paragraphs = segment::paragraphs(&segments, self.embedder.as_ref());

        let language = language.and_then(|l| {
            self.tokenizer
                .id_to_token(l)
                .map(|s| s.trim_start_matches("<|").trim_end_matches("|>").to_string())
        });
        let text = paragraphs.iter().map(|p| p.text.as_str()).collect::<Vec<_>>().join("\n\n");
        Ok(Transcript { text, paragraphs, language, cancelled: was_cancelled })
    }

    /// Turn `<|t0|> text <|t1|><|t1|> text <|t2|>` into timed segments (absolute seconds).
    fn timed_segments(&self, tokens: &[u32], offset: f32, window_end: f32) -> Result<Vec<Segment>> {
        let mut out = Vec::new();
        let mut start = offset;
        let mut buf: Vec<u32> = Vec::new();
        for &t in tokens {
            if t >= self.ts_begin {
                let time = offset + (t - self.ts_begin) as f32 * 0.02;
                if buf.is_empty() {
                    start = time;
                } else {
                    let text = self.decode_text(&buf)?.trim().to_string();
                    if !text.is_empty() {
                        out.push(Segment { start, end: time, text });
                    }
                    buf.clear();
                    start = time;
                }
            } else if t < self.eot {
                buf.push(t);
            }
        }
        if !buf.is_empty() {
            let text = self.decode_text(&buf)?.trim().to_string();
            if !text.is_empty() {
                out.push(Segment { start, end: window_end.max(start), text });
            }
        }
        Ok(out)
    }

    /// OpenAI's seek logic: if the window ended mid-segment, only keep text up to the last
    /// complete timestamp pair and resume from there. Returns (text token end, frames to advance).
    fn split_at_last_timestamp(&self, s: &[u32], seg_size: usize) -> (usize, usize) {
        let is_ts = |t: u32| t >= self.ts_begin;
        let n = s.len();
        let single_ending = n >= 2 && !is_ts(s[n - 2]) && is_ts(s[n - 1]);
        let last_pair = (1..n).rev().find(|&i| is_ts(s[i]) && is_ts(s[i - 1]));
        match last_pair {
            Some(i) if !single_ending => {
                let pos = (s[i - 1] - self.ts_begin) as usize * 2; // 20 ms units -> 10 ms frames
                let advance = if pos == 0 { seg_size } else { pos.min(seg_size) };
                (i, advance)
            }
            _ => (n, seg_size),
        }
    }

    fn decode_text(&self, tokens: &[u32]) -> Result<String> {
        let text_tokens: Vec<u32> = tokens.iter().copied().filter(|&t| t < self.eot).collect();
        self.tokenizer.decode(&text_tokens, true).map_err(|e| anyhow!("{e}"))
    }

    fn detect_language(&mut self, features: &Tensor) -> Result<u32> {
        let t = Tensor::new(&[self.sot], &self.device)?.unsqueeze(0)?;
        let ys = self.model.decoder.forward(&t, features, true)?;
        let logits: Vec<f32> = self.model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?.to_vec1()?;
        self.lang_tokens
            .iter()
            .copied()
            .max_by(|&a, &b| logits[a as usize].total_cmp(&logits[b as usize]))
            .ok_or_else(|| anyhow!("no language tokens"))
    }

    fn decode_with_fallback(
        &mut self,
        features: &Tensor,
        lang: u32,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Decoded> {
        let mut first = None;
        for &temp in m::TEMPERATURES.iter() {
            let d = self.decode(features, lang, temp, cancelled)?;
            let needs_fallback = d.no_speech_prob <= m::NO_SPEECH_THRESHOLD
                && (d.hit_limit
                    || d.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                    || d.avg_logprob < m::LOGPROB_THRESHOLD);
            if !needs_fallback {
                return Ok(d);
            }
            // If every temperature trips the checks, the greedy result is the most faithful.
            first.get_or_insert(d);
        }
        Ok(first.unwrap())
    }

    fn decode(
        &mut self,
        features: &Tensor,
        lang: u32,
        temperature: f64,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Decoded> {
        let prompt = [self.sot, lang, self.transcribe];
        let mut tokens: Vec<u32> = prompt.to_vec();
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = 0f64;
        let mut hit_limit = true;

        for i in 0..MAX_SAMPLED_TOKENS {
            if cancelled() {
                return Err(Cancelled.into());
            }
            let t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let ys = self.model.decoder.forward(&t, features, i == 0)?;
            let seq_len = ys.dim(1)?;

            if i == 0 {
                if let Some(ns) = self.no_speech {
                    let l: Vec<f32> =
                        self.model.decoder.final_linear(&ys.i((..1, ..1))?)?.i(0)?.i(0)?.to_vec1()?;
                    no_speech_prob = softmax(&l)[ns as usize] as f64;
                }
            }

            let logits = self.model.decoder.final_linear(&ys.i((..1, seq_len - 1..))?)?;
            let mut logits: Vec<f32> = logits.i(0)?.i(0)?.to_vec1()?;
            self.apply_rules(&mut logits, &tokens[prompt.len()..]);

            let logprobs = log_softmax(&logits);
            let next = if temperature > 0.0 {
                self.sample(&logits, temperature)
            } else {
                argmax(&logits)
            };
            sum_logprob += logprobs[next as usize] as f64;
            if next == self.eot {
                hit_limit = false;
                break;
            }
            tokens.push(next);
        }

        let sampled = tokens[prompt.len()..].to_vec();
        let text = self.decode_text(&sampled)?;
        Ok(Decoded {
            avg_logprob: sum_logprob / (sampled.len() + 1) as f64,
            compression_ratio: compression_ratio(&text),
            tokens: sampled,
            no_speech_prob,
            hit_limit,
        })
    }

    /// Whisper's logit filters: suppressed tokens + timestamp grammar.
    fn apply_rules(&self, logits: &mut [f32], sampled: &[u32]) {
        let ninf = f32::NEG_INFINITY;
        let tb = self.ts_begin as usize;
        let eot = self.eot as usize;
        let n = logits.len();
        for (l, &s) in logits.iter_mut().zip(self.suppress.iter()) {
            if s {
                *l = ninf;
            }
        }

        if sampled.is_empty() {
            // Must start with a timestamp, and no later than 1.0 s.
            logits[..tb].fill(ninf);
            let max_init = (tb + MAX_INITIAL_TIMESTAMP as usize + 1).min(n);
            logits[max_init..].fill(ninf);
            return;
        }

        let is_ts = |t: u32| t >= self.ts_begin;
        let k = sampled.len();
        let last_was_ts = is_ts(sampled[k - 1]);
        let penultimate_was_ts = k < 2 || is_ts(sampled[k - 2]);
        if last_was_ts {
            if penultimate_was_ts {
                logits[tb..].fill(ninf); // a pair just closed: text must follow
            } else {
                logits[..eot].fill(ninf); // a segment just ended: timestamp or EOT
            }
        }
        // Timestamps never go backwards.
        if let Some(&max_ts) = sampled.iter().filter(|&&t| is_ts(t)).max() {
            let limit = if last_was_ts && !penultimate_was_ts { max_ts } else { max_ts + 1 };
            logits[tb..(limit as usize).min(n)].fill(ninf);
        }
        // If timestamps are collectively more likely than any single text token, emit one.
        let lp = log_softmax(logits);
        let ts_lse = logsumexp(&lp[tb..]);
        let max_text = lp[..tb].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if ts_lse > max_text {
            logits[..tb].fill(ninf);
        }
    }

    fn sample(&mut self, logits: &[f32], temperature: f64) -> u32 {
        let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature as f32).collect();
        let probs = softmax(&scaled);
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let r = (self.rng.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64;
        let mut acc = 0f64;
        for (i, &p) in probs.iter().enumerate() {
            acc += p as f64;
            if acc >= r {
                return i as u32;
            }
        }
        argmax(logits)
    }
}

/// True if at least ~150 ms of the window is louder than about -48 dBFS.
fn has_voice(pcm: &[f32]) -> bool {
    const FRAME: usize = 480; // 30 ms at 16 kHz
    const THRESHOLD: f32 = 0.004;
    pcm.chunks(FRAME)
        .filter(|f| (f.iter().map(|x| x * x).sum::<f32>() / f.len() as f32).sqrt() > THRESHOLD)
        .nth(4)
        .is_some()
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn logsumexp(v: &[f32]) -> f32 {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return max;
    }
    max + v.iter().map(|&x| (x - max).exp()).sum::<f32>().ln()
}

fn log_softmax(v: &[f32]) -> Vec<f32> {
    let lse = logsumexp(v);
    v.iter().map(|&x| x - lse).collect()
}

fn softmax(v: &[f32]) -> Vec<f32> {
    log_softmax(v).into_iter().map(f32::exp).collect()
}

/// gzip-style compression ratio; high values mean the model is looping on repeated text.
fn compression_ratio(text: &str) -> f64 {
    if text.is_empty() {
        return 0.0;
    }
    let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    let _ = enc.write_all(text.as_bytes());
    let compressed = enc.finish().map(|c| c.len()).unwrap_or(1).max(1);
    text.len() as f64 / compressed as f64
}
