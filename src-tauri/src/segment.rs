//! Paragraph segmentation for transcripts.
//!
//! Whisper's timed segments are split into sentences, each sentence is embedded with a small
//! sentence-transformer (all-MiniLM-L6-v2 on Candle), and paragraph breaks are placed where the
//! topic shifts (a dip in similarity between the sentences before and after a boundary,
//! TextTiling-style) or where the speaker paused. If the embedder isn't available, pauses and
//! length alone are used.

use crate::download;
use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use serde::Serialize;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

pub const EMBED_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// A stretch of transcribed text with absolute times in seconds.
#[derive(Clone, Debug)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Paragraph {
    pub start: f32,
    pub text: String,
}

pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    pub fn load(device: &Device, status: download::StatusFn) -> Result<Self> {
        let config = download::fetch(EMBED_MODEL_ID, "config.json", status)?;
        let tok = download::fetch(EMBED_MODEL_ID, "tokenizer.json", status)?;
        let weights = download::fetch(EMBED_MODEL_ID, "model.safetensors", status)?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config)?)?;
        let mut tokenizer = Tokenizer::from_file(tok).map_err(|e| anyhow!("{e}"))?;
        tokenizer.with_padding(Some(PaddingParams::default()));
        tokenizer
            .with_truncation(Some(TruncationParams { max_length: 256, ..Default::default() }))
            .map_err(|e| anyhow!("{e}"))?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DTYPE, device)? };
        let model = BertModel::load(vb, &config)?;
        Ok(Self { model, tokenizer, device: device.clone() })
    }

    /// L2-normalised mean-pooled sentence embeddings.
    pub fn embed(&self, sentences: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(sentences.len());
        for batch in sentences.chunks(32) {
            let enc = self.tokenizer.encode_batch(batch.to_vec(), true).map_err(|e| anyhow!("{e}"))?;
            let ids: Vec<Tensor> = enc
                .iter()
                .map(|e| Tensor::new(e.get_ids(), &self.device))
                .collect::<candle_core::Result<_>>()?;
            let mask: Vec<Tensor> = enc
                .iter()
                .map(|e| Tensor::new(e.get_attention_mask(), &self.device))
                .collect::<candle_core::Result<_>>()?;
            let ids = Tensor::stack(&ids, 0)?;
            let mask = Tensor::stack(&mask, 0)?;
            let types = ids.zeros_like()?;
            let hidden = self.model.forward(&ids, &types, Some(&mask))?; // (b, seq, dim)
            let m = mask.to_dtype(DTYPE)?.unsqueeze(2)?;
            let summed = hidden.broadcast_mul(&m)?.sum(1)?;
            let counts = m.sum(1)?;
            let pooled = summed.broadcast_div(&counts)?;
            let norm = pooled.sqr()?.sum_keepdim(1)?.sqrt()?;
            let pooled = pooled.broadcast_div(&norm)?;
            out.extend(pooled.to_vec2::<f32>()?);
        }
        Ok(out)
    }
}

/// Split timed segments into sentences. Segments are first joined into one character stream
/// (times interpolated per character) so sentences spanning a segment boundary stay whole.
fn sentences(segments: &[Segment]) -> Vec<Segment> {
    let mut chars: Vec<(char, f32)> = Vec::new();
    for seg in segments {
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        if !chars.is_empty() {
            chars.push((' ', seg.start));
        }
        let n = text.chars().count().max(1) as f32;
        for (i, c) in text.chars().enumerate() {
            chars.push((c, seg.start + (seg.end - seg.start) * (i as f32 / n)));
        }
    }
    let end_time = segments.last().map(|s| s.end).unwrap_or(0.0);

    let mut out = Vec::new();
    let mut start_idx = 0;
    for i in 0..chars.len() {
        let c = chars[i].0;
        let ends = matches!(c, '.' | '?' | '!' | '。' | '？' | '！')
            && (i + 1 == chars.len() || chars[i + 1].0.is_whitespace());
        if ends || i + 1 == chars.len() {
            let s: String = chars[start_idx..=i].iter().map(|(c, _)| c).collect();
            let s = s.trim().to_string();
            if !s.is_empty() {
                let end = chars.get(i + 1).map(|x| x.1).unwrap_or(end_time);
                out.push(Segment { start: chars[start_idx].1, end, text: s });
            }
            start_idx = i + 1;
        }
    }
    // Merge tiny fragments ("Yeah." "Right.") into the previous sentence so they don't anchor
    // paragraph boundaries on their own.
    let mut merged: Vec<Segment> = Vec::new();
    for s in out {
        match merged.last_mut() {
            Some(prev) if s.text.len() < 12 && s.start - prev.end < 1.0 => {
                prev.text.push(' ');
                prev.text.push_str(&s.text);
                prev.end = s.end;
            }
            _ => merged.push(s),
        }
    }
    merged
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-6)
}

fn mean(vs: &[Vec<f32>]) -> Vec<f32> {
    let mut m = vec![0f32; vs[0].len()];
    for v in vs {
        for (a, b) in m.iter_mut().zip(v) {
            *a += b;
        }
    }
    m
}

const MIN_CHARS: usize = 160;
const MAX_CHARS: usize = 900;
const WINDOW: usize = 3;

pub fn paragraphs(segments: &[Segment], embedder: Option<&Embedder>) -> Vec<Paragraph> {
    let sents = sentences(segments);
    if sents.is_empty() {
        return Vec::new();
    }
    let n = sents.len();

    // Boundary score for a break *before* sentence i (1..n). Higher = better place to break.
    let mut score = vec![0f32; n];
    let embeddings = embedder.and_then(|e| {
        let texts: Vec<String> = sents.iter().map(|s| s.text.clone()).collect();
        e.embed(&texts).ok()
    });
    if let Some(emb) = &embeddings {
        let mut sim = vec![1f32; n];
        for i in 1..n {
            let left = &emb[i.saturating_sub(WINDOW)..i];
            let right = &emb[i..(i + WINDOW).min(n)];
            sim[i] = cosine(&mean(left), &mean(right));
        }
        // Depth score: how far the similarity dips below the peaks on either side.
        for i in 1..n {
            let lpeak = (1..=i).rev().map(|j| sim[j]).fold(sim[i], f32::max);
            let rpeak = (i..n).map(|j| sim[j]).fold(sim[i], f32::max);
            score[i] = (lpeak - sim[i]) + (rpeak - sim[i]);
        }
    }
    for i in 1..n {
        let gap = (sents[i].start - sents[i - 1].end).max(0.0);
        score[i] += (gap / 2.0).min(1.0) * 0.6; // pauses of 2s+ count strongly
    }

    let scored: Vec<f32> = score[1..].to_vec();
    let mean_s = scored.iter().sum::<f32>() / scored.len().max(1) as f32;
    let std_s = (scored.iter().map(|s| (s - mean_s).powi(2)).sum::<f32>() / scored.len().max(1) as f32)
        .sqrt();
    let threshold = mean_s + 0.5 * std_s;

    // prefix[i] = characters in sentences[..i]
    let mut prefix = vec![0usize; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + sents[i].text.len() + 1;
    }
    let chars_between = |a: usize, b: usize| prefix[b] - prefix[a];

    // Take the strongest boundaries first, keeping every paragraph at least MIN_CHARS long.
    let mut breaks: std::collections::BTreeSet<usize> = Default::default();
    let fits = |breaks: &std::collections::BTreeSet<usize>, i: usize| {
        let prev = breaks.range(..i).next_back().copied().unwrap_or(0);
        let next = breaks.range(i + 1..).next().copied().unwrap_or(n);
        chars_between(prev, i) >= MIN_CHARS && chars_between(i, next) >= MIN_CHARS
    };
    let mut order: Vec<usize> = (1..n).collect();
    order.sort_by(|&a, &b| score[b].total_cmp(&score[a]));
    for &i in &order {
        if score[i] > threshold && fits(&breaks, i) {
            breaks.insert(i);
        }
    }
    // Then split anything that's still too long at its best internal boundary.
    // (A paragraph with no boundary that leaves both halves a reasonable size is left whole,
    // rather than splitting off a fragment.)
    let mut unsplittable: std::collections::HashSet<(usize, usize)> = Default::default();
    loop {
        let mut bounds: Vec<usize> = vec![0];
        bounds.extend(breaks.iter().copied());
        bounds.push(n);
        let long = bounds
            .windows(2)
            .map(|w| (w[0], w[1]))
            .find(|&(a, b)| {
                chars_between(a, b) > MAX_CHARS && b - a >= 2 && !unsplittable.contains(&(a, b))
            });
        let Some((a, b)) = long else { break };
        let best = (a + 1..b)
            .filter(|&i| chars_between(a, i) >= MIN_CHARS && chars_between(i, b) >= MIN_CHARS)
            .max_by(|&x, &y| score[x].total_cmp(&score[y]));
        match best {
            Some(i) => {
                breaks.insert(i);
            }
            None => {
                unsplittable.insert((a, b));
            }
        }
    }

    let mut out = Vec::new();
    let mut bounds = vec![0];
    bounds.extend(breaks.iter().copied());
    bounds.push(n);
    for w in bounds.windows(2) {
        if w[0] >= w[1] {
            continue;
        }
        let text = sents[w[0]..w[1]].iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ");
        out.push(Paragraph { start: sents[w[0]].start, text });
    }
    out
}
