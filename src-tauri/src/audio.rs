//! Audio loading: any file -> 16 kHz mono f32 PCM.
//!
//! Decoding is pure Rust via Symphonia (m4a/AAC, ALAC, mp3, wav, flac, ogg/vorbis, caf, ...).
//! If Symphonia can't handle the file (e.g. Opus voice notes) and ffmpeg is installed, we fall
//! back to ffmpeg so that "any kind of audio file" really does work.

use anyhow::{anyhow, bail, Context, Result};
use std::path::Path;
use std::process::Command;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub const TARGET_RATE: u32 = 16_000;

pub fn load_audio(path: &Path) -> Result<Vec<f32>> {
    match decode_symphonia(path) {
        Ok(pcm) => Ok(pcm),
        Err(sym_err) => match decode_ffmpeg(path) {
            Ok(pcm) => Ok(pcm),
            Err(ff_err) => Err(anyhow!(
                "Couldn't decode this file.\n• Built-in decoder: {sym_err:#}\n• ffmpeg: {ff_err:#}"
            )),
        },
    }
}

fn decode_symphonia(path: &Path) -> Result<Vec<f32>> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("unrecognised container")?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL && t.codec_params.sample_rate.is_some())
        .ok_or_else(|| anyhow!("no audio track"))?;
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("unsupported codec")?;

    let mut mono: Vec<f32> = Vec::new();
    let mut buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymError::DecodeError(_)) => continue, // skip corrupt packet
            Err(e) => return Err(e.into()),
        };
        let spec = *decoded.spec();
        let channels = spec.channels.count().max(1);
        let sb = match &mut buf {
            Some(b) if b.capacity() >= decoded.capacity() * channels => b,
            _ => buf.insert(SampleBuffer::new(decoded.capacity() as u64, spec)),
        };
        sb.copy_interleaved_ref(decoded);
        for frame in sb.samples().chunks_exact(channels) {
            mono.push(frame.iter().sum::<f32>() / channels as f32);
        }
    }
    if mono.is_empty() {
        bail!("file contained no audio samples");
    }
    resample(&mono, sample_rate)
}

fn find_ffmpeg() -> Option<std::path::PathBuf> {
    // GUI apps on macOS don't inherit the shell PATH, so check the usual spots explicitly.
    let candidates = [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/local/bin/ffmpeg",
    ];
    candidates
        .iter()
        .map(std::path::PathBuf::from)
        .find(|p| p.exists())
        .or_else(|| {
            std::env::var_os("PATH").and_then(|paths| {
                std::env::split_paths(&paths)
                    .map(|d| d.join("ffmpeg"))
                    .find(|p| p.exists())
            })
        })
}

fn decode_ffmpeg(path: &Path) -> Result<Vec<f32>> {
    let ffmpeg = find_ffmpeg().ok_or_else(|| anyhow!("not installed (brew install ffmpeg)"))?;
    let out = Command::new(ffmpeg)
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(path)
        .args([
            "-vn",
            "-ac",
            "1",
            "-ar",
            &TARGET_RATE.to_string(),
            "-f",
            "f32le",
            "-",
        ])
        .output()
        .context("running ffmpeg")?;
    if !out.status.success() {
        bail!("{}", String::from_utf8_lossy(&out.stderr).trim());
    }
    let pcm: Vec<f32> = out
        .stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if pcm.is_empty() {
        bail!("no audio stream");
    }
    Ok(pcm)
}

fn resample(input: &[f32], from_rate: u32) -> Result<Vec<f32>> {
    use rubato::{FftFixedIn, Resampler};
    if from_rate == TARGET_RATE {
        return Ok(input.to_vec());
    }
    let chunk = 4096;
    let mut rs = FftFixedIn::<f32>::new(from_rate as usize, TARGET_RATE as usize, chunk, 2, 1)?;
    let delay = rs.output_delay();
    let expected = (input.len() as u64 * TARGET_RATE as u64).div_ceil(from_rate as u64) as usize;
    let mut out = Vec::with_capacity(expected + delay + chunk);

    let mut pos = 0;
    while input.len() - pos >= rs.input_frames_next() {
        let n = rs.input_frames_next();
        let o = rs.process(&[&input[pos..pos + n]], None)?;
        out.extend_from_slice(&o[0]);
        pos += n;
    }
    if pos < input.len() {
        let o = rs.process_partial(Some(&[&input[pos..]]), None)?;
        out.extend_from_slice(&o[0]);
    }
    // Flush the resampler's internal delay.
    while out.len() < expected + delay {
        let o = rs.process_partial::<&[f32]>(None, None)?;
        if o[0].is_empty() {
            break;
        }
        out.extend_from_slice(&o[0]);
    }
    let end = (delay + expected).min(out.len());
    Ok(out[delay.min(end)..end].to_vec())
}

/// Slaney-style mel filterbank, identical to `librosa.filters.mel(sr=16000, n_fft=400, n_mels)`,
/// which is what Whisper was trained with. Row-major `[n_mels][n_fft/2 + 1]`.
pub fn mel_filters(n_mels: usize) -> Vec<f32> {
    const SR: f64 = 16_000.0;
    const N_FFT: usize = 400;
    let n_freqs = N_FFT / 2 + 1;

    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    let hz_to_mel = |f: f64| {
        if f >= min_log_hz {
            min_log_mel + (f / min_log_hz).ln() / logstep
        } else {
            f / f_sp
        }
    };
    let mel_to_hz = |m: f64| {
        if m >= min_log_mel {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        } else {
            f_sp * m
        }
    };

    let fftfreqs: Vec<f64> = (0..n_freqs).map(|i| i as f64 * SR / N_FFT as f64).collect();
    let (mmin, mmax) = (hz_to_mel(0.0), hz_to_mel(SR / 2.0));
    let mel_f: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mmin + (mmax - mmin) * i as f64 / (n_mels + 1) as f64))
        .collect();

    let mut w = vec![0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let enorm = 2.0 / (mel_f[m + 2] - mel_f[m]);
        for (k, &f) in fftfreqs.iter().enumerate() {
            let lower = (f - mel_f[m]) / (mel_f[m + 1] - mel_f[m]);
            let upper = (mel_f[m + 2] - f) / (mel_f[m + 2] - mel_f[m + 1]);
            w[m * n_freqs + k] = (lower.min(upper).max(0.0) * enorm) as f32;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a mono 16-bit PCM WAV containing a sine wave.
    fn write_wav(path: &Path, rate: u32, secs: f32, freq: f32) {
        let n = (rate as f32 * secs) as usize;
        let data: Vec<u8> = (0..n)
            .flat_map(|i| {
                let v = (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin();
                ((v * 0.5 * i16::MAX as f32) as i16).to_le_bytes()
            })
            .collect();
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&rate.to_le_bytes());
        wav.extend_from_slice(&(rate * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);
        std::fs::write(path, wav).unwrap();
    }

    #[test]
    fn decodes_and_resamples_wav_to_16k() {
        let dir = std::env::temp_dir().join(format!("dtt-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tone.wav");
        write_wav(&path, 44_100, 2.0, 440.0);
        let pcm = decode_symphonia(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(pcm.len(), 32_000, "2 s at 16 kHz");
        let peak = pcm.iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!(
            (0.4..0.6).contains(&peak),
            "amplitude preserved, got {peak}"
        );
    }

    #[test]
    fn resample_is_identity_at_target_rate() {
        let input: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        assert_eq!(resample(&input, TARGET_RATE).unwrap(), input);
    }

    #[test]
    fn resample_length_matches_ratio() {
        let input = vec![0.1f32; 48_000];
        assert_eq!(resample(&input, 48_000).unwrap().len(), 16_000);
    }

    #[test]
    fn rejects_non_audio() {
        let dir = std::env::temp_dir().join(format!("dtt-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.m4a");
        std::fs::write(&path, b"definitely not audio").unwrap();
        assert!(decode_symphonia(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mel_filters_shape_and_normalisation() {
        let f = mel_filters(128);
        assert_eq!(f.len(), 128 * 201);
        assert!(f.iter().all(|&x| x >= 0.0));
        // Every mel band has some weight; slaney-normalised peaks are small.
        for band in f.chunks(201) {
            let max = band.iter().cloned().fold(0f32, f32::max);
            assert!(max > 0.0 && max < 0.05);
        }
    }
}
