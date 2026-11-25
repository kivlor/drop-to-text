//! Model file fetching: reuse an existing Hugging Face cache if present, otherwise download into
//! ~/Library/Application Support/<app>/models with resume, read timeouts and retries.

use anyhow::{anyhow, bail, Context, Result};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::Duration;

pub type StatusFn<'a> = &'a mut dyn FnMut(String, Option<f32>);

fn hf_cache_lookup(repo: &str, file: &str) -> Option<PathBuf> {
    let base = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub")))?;
    let snapshots = base.join(format!("models--{}", repo.replace('/', "--"))).join("snapshots");
    fs::read_dir(snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path().join(file))
        .find(|p| p.exists())
}

pub fn models_dir() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").ok_or_else(|| anyhow!("HOME not set"))?;
    Ok(PathBuf::from(home).join("Library/Application Support/com.droptotext.app/models"))
}

pub fn fetch(repo: &str, file: &str, status: StatusFn) -> Result<PathBuf> {
    if let Some(p) = hf_cache_lookup(repo, file) {
        return Ok(p);
    }
    let dir = models_dir()?.join(repo.replace('/', "--"));
    let dest = dir.join(file);
    if dest.exists() {
        return Ok(dest);
    }
    fs::create_dir_all(&dir)?;
    let part = dir.join(format!("{file}.part"));
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(30))
        .build();

    let mut last_err = None;
    for attempt in 0..8 {
        if attempt > 0 {
            status(format!("Download interrupted, retrying ({attempt}/7)…"), None);
            std::thread::sleep(Duration::from_secs(2u64.pow(attempt.min(4))));
        }
        match download_once(&agent, &url, &part, file, status) {
            Ok(()) => {
                fs::rename(&part, &dest)?;
                return Ok(dest);
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap()).with_context(|| format!("downloading {file}"))
}

fn download_once(
    agent: &ureq::Agent,
    url: &str,
    part: &PathBuf,
    file: &str,
    status: StatusFn,
) -> Result<()> {
    let have = fs::metadata(part).map(|m| m.len()).unwrap_or(0);
    let mut req = agent.get(url);
    if have > 0 {
        req = req.set("Range", &format!("bytes={have}-"));
    }
    let resp = req.call()?;
    let resumed = resp.status() == 206;
    let mut out = if resumed {
        OpenOptions::new().append(true).open(part)?
    } else {
        fs::File::create(part)? // server ignored Range: start over
    };
    let mut done = if resumed { have } else { 0 };
    let total = resp
        .header("Content-Length")
        .and_then(|v| v.parse::<u64>().ok())
        .map(|n| n + done);

    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let mut last_pct = -1i64;
    let mb = |b: u64| b as f64 / 1_048_576.0;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        done += n as u64;
        if let Some(t) = total {
            let pct = (done * 100 / t.max(1)) as i64;
            if pct != last_pct {
                last_pct = pct;
                status(
                    format!("Downloading {file} — {:.0} / {:.0} MB", mb(done), mb(t)),
                    Some(done as f32 / t as f32),
                );
            }
        }
    }
    out.flush()?;
    if let Some(t) = total {
        if done != t {
            bail!("connection closed early ({done} of {t} bytes)");
        }
    }
    Ok(())
}
