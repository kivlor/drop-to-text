pub mod audio;
mod download;
pub mod engine;
mod parakeet;
pub mod segment;
mod whisper;

use engine::{Backend, Engine};
use serde::Serialize;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::{AppHandle, Emitter, Manager, State};

#[derive(Clone, Serialize)]
struct ModelStatus {
    /// "loading" | "ready" | "error"
    state: &'static str,
    message: String,
    progress: Option<f32>,
}

#[derive(Default)]
struct AppState {
    engine: Mutex<Option<Engine>>,
    status: Mutex<Option<ModelStatus>>,
    /// Job ids the user has cancelled; checked by the running job between tokens.
    cancelled: Mutex<HashSet<String>>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Progress {
    id: String,
    stage: &'static str,
    fraction: f32,
    text: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TranscriptOut {
    text: String,
    paragraphs: Vec<segment::Paragraph>,
    cancelled: bool,
    language: Option<String>,
    audio_secs: f32,
    elapsed_secs: f32,
}

fn set_status(app: &AppHandle, st: ModelStatus) {
    let state = app.state::<Arc<AppState>>();
    *state.status.lock().unwrap() = Some(st.clone());
    let _ = app.emit("model-status", st);
}

#[tauri::command]
fn model_status(state: State<'_, Arc<AppState>>) -> Option<ModelStatus> {
    state.status.lock().unwrap().clone()
}

#[tauri::command]
fn cancel(state: State<'_, Arc<AppState>>, id: String) {
    state.cancelled.lock().unwrap().insert(id);
}

#[tauri::command]
async fn transcribe(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
    id: String,
    path: String,
    engine: Option<String>,
) -> Result<TranscriptOut, String> {
    let backend =
        Backend::parse(engine.as_deref().unwrap_or("parakeet")).map_err(|e| e.to_string())?;
    let state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let emit = |stage, fraction, text: &str| {
            let _ = app.emit(
                "transcribe-progress",
                Progress {
                    id: id.clone(),
                    stage,
                    fraction,
                    text: text.to_string(),
                },
            );
        };
        let is_cancelled = || state.cancelled.lock().unwrap().contains(&id);
        let finish_cancelled = || TranscriptOut {
            text: String::new(),
            paragraphs: vec![],
            cancelled: true,
            language: None,
            audio_secs: 0.0,
            elapsed_secs: 0.0,
        };
        let result = (|| {
            let started = Instant::now();
            emit("decoding", 0.0, "");
            let pcm = audio::load_audio(&PathBuf::from(&path)).map_err(|e| format!("{e:#}"))?;
            let audio_secs = pcm.len() as f32 / audio::TARGET_RATE as f32;
            if is_cancelled() {
                return Ok(finish_cancelled());
            }

            emit("waiting", 0.0, "");
            let mut guard = state.engine.lock().unwrap();
            let engine = guard.as_mut().ok_or("The model isn't loaded yet.")?;
            if !engine.is_loaded(backend) {
                // First use of Whisper: load it now, reporting progress in the status line.
                let mut report = |message: String, progress: Option<f32>| {
                    set_status(
                        &app,
                        ModelStatus {
                            state: "loading",
                            message,
                            progress,
                        },
                    );
                };
                if let Err(e) = engine.ensure_loaded(backend, &mut report) {
                    let message = format!("{e:#}");
                    set_status(
                        &app,
                        ModelStatus {
                            state: "error",
                            message: message.clone(),
                            progress: None,
                        },
                    );
                    return Err(message);
                }
                if is_cancelled() {
                    return Ok(finish_cancelled());
                }
            }
            set_status(&app, ready_status(backend));
            emit("transcribing", 0.0, "");
            let t = engine
                .transcribe(backend, &pcm, &is_cancelled, &mut |f, text| {
                    emit("transcribing", f, text)
                })
                .map_err(|e| format!("{e:#}"))?;
            Ok(TranscriptOut {
                text: t.text,
                paragraphs: t.paragraphs,
                cancelled: t.cancelled,
                language: t.language,
                audio_secs,
                elapsed_secs: started.elapsed().as_secs_f32(),
            })
        })();
        state.cancelled.lock().unwrap().remove(&id);
        result
    })
    .await
    .map_err(|e| e.to_string())?
}

fn ready_status(backend: Backend) -> ModelStatus {
    ModelStatus {
        state: "ready",
        message: backend.label().to_string(),
        progress: None,
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Arc::new(AppState::default()))
        .invoke_handler(tauri::generate_handler![model_status, transcribe, cancel])
        .setup(|app| {
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                let mut report = |message: String, progress: Option<f32>| {
                    set_status(
                        &handle,
                        ModelStatus {
                            state: "loading",
                            message,
                            progress,
                        },
                    );
                };
                match Engine::load(&mut report) {
                    Ok(engine) => {
                        *handle.state::<Arc<AppState>>().engine.lock().unwrap() = Some(engine);
                        set_status(&handle, ready_status(Backend::Parakeet));
                    }
                    Err(e) => set_status(
                        &handle,
                        ModelStatus {
                            state: "error",
                            message: format!("{e:#}"),
                            progress: None,
                        },
                    ),
                }
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Drop to Text");
}
