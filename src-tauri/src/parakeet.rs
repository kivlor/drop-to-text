//! NVIDIA Parakeet TDT 0.6B v3 speech-to-text through FluidAudio (Core ML on the Apple Neural
//! Engine). FluidAudio is Swift; swift/ wraps it in a small C ABI that build.rs compiles and
//! links statically. This module is the safe Rust side of that ABI.
//!
//! FluidAudio handles long audio itself: it splits it into overlapping windows, decodes several
//! at once and merges the seams. It returns word timings, which become one `Segment` per word.

use crate::download;
use crate::segment::Segment;
use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::NonNull;

/// FluidAudio downloads the Core ML model (about 470 MB) from the Hugging Face repo
/// FluidInference/parakeet-tdt-0.6b-v3-coreml into this folder of the app's models directory.
const MODEL_DIR: &str = "parakeet-tdt-0.6b-v3";

type Callback = unsafe extern "C" fn(*mut c_void, f64, *const c_char) -> bool;

extern "C" {
    fn ptb_load(
        version: i32,
        models_dir: *const c_char,
        callback: Option<Callback>,
        ctx: *mut c_void,
        error: *mut *mut c_char,
    ) -> *mut c_void;
    fn ptb_transcribe(
        engine: *mut c_void,
        samples: *const f32,
        count: isize,
        callback: Option<Callback>,
        ctx: *mut c_void,
        error: *mut *mut c_char,
    ) -> *mut c_char;
    fn ptb_release(engine: *mut c_void);
    fn ptb_free_string(s: *mut c_char);
}

/// The Rust closure behind a C callback: `(fraction or -1, text) -> cancel?`.
type Progress<'a> = &'a mut dyn FnMut(f64, Option<&str>) -> bool;

unsafe extern "C" fn trampoline(ctx: *mut c_void, fraction: f64, text: *const c_char) -> bool {
    let f = &mut *(ctx as *mut Progress);
    let text = (!text.is_null()).then(|| CStr::from_ptr(text).to_str().unwrap_or_default());
    // Never unwind into Swift; treat a panic as a request to stop.
    catch_unwind(AssertUnwindSafe(|| f(fraction, text))).unwrap_or(true)
}

/// Take ownership of a string returned by the bridge.
unsafe fn take_string(s: *mut c_char) -> String {
    let out = CStr::from_ptr(s).to_string_lossy().into_owned();
    ptb_free_string(s);
    out
}

pub struct Parakeet {
    handle: NonNull<c_void>,
}

// The Swift engine is an actor-backed object with no thread affinity; we only ever use it from
// one thread at a time (it lives behind the app's engine mutex).
unsafe impl Send for Parakeet {}

impl Drop for Parakeet {
    fn drop(&mut self) {
        unsafe { ptb_release(self.handle.as_ptr()) }
    }
}

pub struct ParakeetOutput {
    pub segments: Vec<Segment>,
    pub cancelled: bool,
}

#[derive(Deserialize)]
struct RawOutput {
    cancelled: bool,
    #[serde(default)]
    words: Vec<RawSpan>,
    #[serde(default)]
    windows: Vec<RawSpan>,
}

#[derive(Deserialize)]
struct RawSpan {
    #[serde(alias = "w")]
    text: String,
    s: f32,
    e: f32,
}

impl Parakeet {
    /// Load the model, downloading it on first use. The first load on a Mac also compiles the
    /// model for the Neural Engine (a one-off); later loads take well under a second.
    pub fn load(status: download::StatusFn) -> Result<Self> {
        let dir = download::models_dir()?.join(MODEL_DIR);
        let dir = CString::new(dir.to_string_lossy().as_bytes())?;
        status("Loading speech model…".into(), None);
        let mut on_progress = |fraction: f64, text: Option<&str>| {
            let fraction = (fraction >= 0.0).then_some(fraction as f32);
            match text {
                Some("download") => status("Downloading speech model…".into(), fraction),
                Some("compile") => status("Optimising speech model for this Mac…".into(), None),
                _ => {}
            }
            false
        };
        let mut progress: Progress = &mut on_progress;
        let mut err: *mut c_char = std::ptr::null_mut();
        let handle = unsafe {
            ptb_load(
                3,
                dir.as_ptr(),
                Some(trampoline),
                &mut progress as *mut Progress as *mut c_void,
                &mut err,
            )
        };
        match NonNull::new(handle) {
            Some(handle) => Ok(Self { handle }),
            None if !err.is_null() => Err(anyhow!("Couldn't load the speech model: {}", unsafe {
                take_string(err)
            })),
            None => Err(anyhow!("Couldn't load the speech model")),
        }
    }

    /// Transcribe 16 kHz mono PCM. `on_update(fraction_done, text_so_far)`; `cancelled()` is polled
    /// every 50 ms. On cancel, returns the windows that had finished.
    pub fn transcribe(
        &mut self,
        pcm: &[f32],
        cancelled: &dyn Fn() -> bool,
        on_update: &mut dyn FnMut(f32, &str),
    ) -> Result<ParakeetOutput> {
        let mut on_progress = |fraction: f64, text: Option<&str>| {
            if let Some(text) = text {
                let windows: Vec<&str> = text.split('\u{1E}').collect();
                on_update(
                    fraction.clamp(0.0, 1.0) as f32,
                    &merge_windows(&windows).join(" "),
                );
            }
            cancelled()
        };
        let mut progress: Progress = &mut on_progress;
        let mut err: *mut c_char = std::ptr::null_mut();
        let json = unsafe {
            ptb_transcribe(
                self.handle.as_ptr(),
                pcm.as_ptr(),
                pcm.len() as isize,
                Some(trampoline),
                &mut progress as *mut Progress as *mut c_void,
                &mut err,
            )
        };
        if json.is_null() {
            let msg = if err.is_null() {
                "unknown error".to_string()
            } else {
                unsafe { take_string(err) }
            };
            return Err(anyhow!("Transcription failed: {msg}"));
        }
        parse_output(&unsafe { take_string(json) })
    }
}

fn parse_output(json: &str) -> Result<ParakeetOutput> {
    let raw: RawOutput = serde_json::from_str(json)?;
    let spans = if raw.cancelled {
        let texts: Vec<&str> = raw.windows.iter().map(|w| w.text.as_str()).collect();
        let merged = merge_windows(&texts);
        raw.windows
            .iter()
            .zip(merged)
            .map(|(w, text)| RawSpan {
                text,
                s: w.s,
                e: w.e,
            })
            .collect()
    } else {
        raw.words
    };
    let segments = spans
        .into_iter()
        .filter(|s| !s.text.trim().is_empty())
        .map(|s| Segment {
            start: s.s,
            end: s.e.max(s.s),
            text: s.text.trim().to_string(),
        })
        .collect();
    Ok(ParakeetOutput {
        segments,
        cancelled: raw.cancelled,
    })
}

/// Consecutive windows overlap by a few seconds, so each window's text usually starts with words
/// the previous one ended with. Drop that repeated lead-in from every window after the first.
/// (FluidAudio does a proper token-level merge for the final result; this is for the live text
/// and for what's kept on cancel.)
fn merge_windows(windows: &[&str]) -> Vec<String> {
    let norm = |w: &str| -> String {
        w.chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect()
    };
    let mut out: Vec<String> = Vec::with_capacity(windows.len());
    let mut prev: Vec<String> = Vec::new();
    for w in windows {
        let words: Vec<&str> = w.split_whitespace().collect();
        let keys: Vec<String> = words.iter().map(|w| norm(w)).collect();
        // Longest run (at least 2 words, at most 40) that ends `prev` and starts this window.
        let max = keys.len().min(prev.len()).min(40);
        let overlap = (2..=max)
            .rev()
            .find(|&k| prev[prev.len() - k..] == keys[..k])
            .unwrap_or(0);
        out.push(words[overlap..].join(" "));
        prev = keys;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_become_segments() {
        let out = parse_output(
            r#"{"cancelled":false,"words":[{"w":"Hello,","s":0.32,"e":0.64},{"w":" ","s":0.7,"e":0.7},{"w":"world.","s":0.72,"e":1.1}]}"#,
        )
        .unwrap();
        assert!(!out.cancelled);
        let texts: Vec<_> = out.segments.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, ["Hello,", "world."]);
        assert_eq!(out.segments[1].start, 0.72);
    }

    #[test]
    fn cancelled_output_keeps_finished_windows() {
        let out = parse_output(
            r#"{"cancelled":true,"windows":[{"text":"First window.","s":0.0,"e":14.9}]}"#,
        )
        .unwrap();
        assert!(out.cancelled);
        assert_eq!(out.segments.len(), 1);
        assert_eq!(out.segments[0].text, "First window.");
    }

    #[test]
    fn window_overlap_is_removed() {
        let merged = merge_windows(&[
            "And we gotta remember, I know, I know. So it's",
            "I know, I know. So it's not absolutely not true.",
            "Something new.",
            "",
        ]);
        assert_eq!(
            merged,
            [
                "And we gotta remember, I know, I know. So it's",
                "not absolutely not true.",
                "Something new.",
                "",
            ]
        );
        // A single shared word isn't treated as overlap.
        assert_eq!(merge_windows(&["the end", "end game"])[1], "end game");
    }

    #[test]
    fn trampoline_forwards_and_survives_panics() {
        let mut seen = Vec::new();
        let mut f = |fraction: f64, text: Option<&str>| {
            seen.push((fraction, text.map(str::to_string)));
            false
        };
        let mut p: Progress = &mut f;
        let text = CString::new("hi").unwrap();
        let stop =
            unsafe { trampoline(&mut p as *mut Progress as *mut c_void, 0.5, text.as_ptr()) };
        assert!(!stop);
        assert_eq!(seen, [(0.5, Some("hi".to_string()))]);

        let mut boom = |_: f64, _: Option<&str>| -> bool { panic!("boom") };
        let mut p: Progress = &mut boom;
        let stop = unsafe {
            trampoline(
                &mut p as *mut Progress as *mut c_void,
                -1.0,
                std::ptr::null(),
            )
        };
        assert!(
            stop,
            "a panicking callback cancels instead of unwinding into Swift"
        );
    }
}
