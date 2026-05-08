//! Concurrent + chunked download engine.
//!
//! Inputs: a list of [`FileSource`]s, a destination directory, and a
//! [`HubContext`] (metadata adapter + http transport). For each file the
//! engine:
//!
//! 1. Runs HEAD to discover size (unless the source already carries it).
//! 2. Pre-allocates the output file and loads/creates its `.progress`
//!    bitmap (byte-compatible with the Go CLI's format).
//! 3. Splits the file into chunks per [`chunk::plan_chunks`] and schedules
//!    the pending ones on a bounded task set, respecting both a
//!    file-level semaphore and a chunk-level semaphore.
//! 4. Streams each chunk into `dest_dir/<file>` at the right offset; on
//!    success flips the marker byte.
//!
//! Progress aggregation: per-file `AtomicU64` counters are incremented
//! by workers as bytes flow through a [`CountingSink`]. The main `run`
//! task polls them at a fixed interval via `tokio::select!` and invokes
//! the user callback from that single thread — no extra dispatcher task
//! needed, which means no `'static` callback bound required. Returning
//! `false` from the callback flips `cancel` and every worker bails on
//! its next chunk boundary.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::fs::OpenOptions;
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::error::{Error, Result};
use crate::hub::metadata::FileSource;
use crate::hub::{FileProgress, ProgressCallback};
use crate::transport::HttpTransport;

use super::chunk;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub file_concurrency: usize,
    pub chunk_concurrency: usize,
    pub progress_interval: Duration,
}

impl EngineConfig {
    /// Build a config from `GENIEX_DL_*` env overrides, falling back to
    /// the supplied default file concurrency (hubs signal their own
    /// comfort level — HF bumps it up with a token, local/S3 want 1).
    pub fn resolve(default_file_concurrency: usize) -> Self {
        let file_conc = env_usize("GENIEX_DL_FILE_CONCURRENCY").unwrap_or(default_file_concurrency);
        let chunk_conc = env_usize("GENIEX_DL_CHUNK_CONCURRENCY").unwrap_or(8);
        Self {
            file_concurrency: file_conc.max(1),
            chunk_concurrency: chunk_conc.max(1),
            progress_interval: Duration::from_millis(100),
        }
    }
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0)
}

struct FileState {
    name: String,
    total_bytes: AtomicU64,
    downloaded_bytes: AtomicU64,
}

impl FileState {
    fn snapshot(&self) -> FileProgress {
        let total = self.total_bytes.load(Ordering::Relaxed);
        FileProgress {
            file_name: self.name.clone(),
            downloaded_bytes: self.downloaded_bytes.load(Ordering::Relaxed) as i64,
            total_bytes: if total == u64::MAX { -1 } else { total as i64 },
        }
    }
}

pub struct Engine {
    transport: Arc<dyn HttpTransport>,
    cfg: EngineConfig,
}

impl Engine {
    pub fn new(transport: Arc<dyn HttpTransport>, default_file_concurrency: usize) -> Self {
        Self {
            cfg: EngineConfig::resolve(default_file_concurrency),
            transport,
        }
    }

    pub fn with_config(transport: Arc<dyn HttpTransport>, cfg: EngineConfig) -> Self {
        Self { transport, cfg }
    }

    pub async fn run(
        &self,
        sources: Vec<FileSource>,
        dest_dir: &Path,
        on_progress: Option<&ProgressCallback>,
    ) -> Result<()> {
        tokio::fs::create_dir_all(dest_dir).await?;

        let states: Vec<Arc<FileState>> = sources
            .iter()
            .map(|s| {
                Arc::new(FileState {
                    name: s.name.clone(),
                    total_bytes: AtomicU64::new(s.size.unwrap_or(u64::MAX)),
                    downloaded_bytes: AtomicU64::new(0),
                })
            })
            .collect();

        let cancel = Arc::new(AtomicBool::new(false));
        let file_sem = Arc::new(Semaphore::new(self.cfg.file_concurrency));
        let chunk_sem = Arc::new(Semaphore::new(self.cfg.chunk_concurrency));

        let mut file_tasks: JoinSet<Result<()>> = JoinSet::new();
        let transport = self.transport.clone();
        let dest_owned = dest_dir.to_path_buf();

        for (source, state) in sources.into_iter().zip(states.iter().cloned()) {
            let file_sem = file_sem.clone();
            let chunk_sem = chunk_sem.clone();
            let transport = transport.clone();
            let dest_dir = dest_owned.clone();
            let cancel = cancel.clone();
            file_tasks.spawn(async move {
                let _permit = file_sem
                    .acquire_owned()
                    .await
                    .map_err(|e| Error::Http(format!("file semaphore closed: {e}")))?;
                download_one(source, state, &dest_dir, transport, chunk_sem, cancel).await
            });
        }

        let mut first_err: Option<Error> = None;
        let mut ticker = tokio::time::interval(self.cfg.progress_interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Skip the immediate first tick so the first callback fires after
        // workers have actually started.
        ticker.tick().await;

        loop {
            tokio::select! {
                biased;

                maybe = file_tasks.join_next() => {
                    match maybe {
                        None => break,
                        Some(Ok(Ok(()))) => {}
                        Some(Ok(Err(e))) => {
                            cancel.store(true, Ordering::SeqCst);
                            first_err.get_or_insert(e);
                        }
                        Some(Err(join_err)) => {
                            cancel.store(true, Ordering::SeqCst);
                            first_err
                                .get_or_insert(Error::Http(format!("join: {join_err}")));
                        }
                    }
                }

                _ = ticker.tick() => {
                    if let Some(cb) = on_progress {
                        let snaps: Vec<FileProgress> =
                            states.iter().map(|s| s.snapshot()).collect();
                        if !(cb)(&snaps) {
                            cancel.store(true, Ordering::SeqCst);
                        }
                    }
                }
            }
        }

        // Terminal callback with the final counters, so the caller
        // always sees 100% (or the cancel state).
        if let Some(cb) = on_progress {
            let snaps: Vec<FileProgress> = states.iter().map(|s| s.snapshot()).collect();
            let _ = (cb)(&snaps);
        }

        if let Some(e) = first_err {
            return Err(e);
        }
        if cancel.load(Ordering::SeqCst) {
            return Err(Error::Cancelled);
        }
        Ok(())
    }
}

async fn download_one(
    source: FileSource,
    state: Arc<FileState>,
    dest_dir: &Path,
    transport: Arc<dyn crate::transport::HttpTransport>,
    chunk_sem: Arc<Semaphore>,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    let size = match source.size {
        Some(n) => n,
        None => {
            let info = transport.head(&source.url, source.auth.as_deref()).await?;
            info.size
        }
    };
    state.total_bytes.store(size, Ordering::Relaxed);

    let output_path = dest_dir.join(&source.name);
    let marker_path = PathBuf::from(format!(
        "{}{}",
        output_path.display(),
        crate::pull::PROGRESS_SUFFIX
    ));

    chunk::preallocate(&output_path, size)?;

    let plan = chunk::plan_chunks(size);
    let bitmap = chunk::load_or_init_bitmap(&marker_path, &plan)?;
    let already = chunk::bytes_already_done(&plan, &bitmap);
    state.downloaded_bytes.store(already, Ordering::Relaxed);

    let pending = chunk::pending_chunks(&plan, &bitmap);
    if pending.is_empty() {
        return Ok(());
    }

    let mut chunk_tasks: JoinSet<Result<()>> = JoinSet::new();
    for range in pending {
        if cancel.load(Ordering::SeqCst) {
            break;
        }
        let sem = chunk_sem.clone();
        let transport = transport.clone();
        let auth = source.auth.clone();
        let url = source.url.clone();
        let output_path = output_path.clone();
        let marker_path = marker_path.clone();
        let state = state.clone();
        let cancel = cancel.clone();
        chunk_tasks.spawn(async move {
            if cancel.load(Ordering::SeqCst) {
                return Err(Error::Cancelled);
            }
            let _permit = sem
                .acquire_owned()
                .await
                .map_err(|e| Error::Http(format!("chunk semaphore closed: {e}")))?;

            let mut file = OpenOptions::new()
                .write(true)
                .read(true)
                .open(&output_path)
                .await?;
            file.seek(std::io::SeekFrom::Start(range.offset)).await?;

            let mut counted = CountingSink::new(&mut file, state.clone());
            transport
                .get_range(&url, auth.as_deref(), range.offset, range.len, &mut counted)
                .await?;
            drop(counted);
            file.flush().await?;
            drop(file);

            chunk::mark_chunk_done(&marker_path, range.index)?;
            Ok(())
        });
    }

    let mut first_err: Option<Error> = None;
    while let Some(res) = chunk_tasks.join_next().await {
        match res {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                cancel.store(true, Ordering::SeqCst);
                first_err.get_or_insert(e);
            }
            Err(join_err) => {
                cancel.store(true, Ordering::SeqCst);
                first_err.get_or_insert(Error::Http(format!("chunk join: {join_err}")));
            }
        }
    }

    if let Some(e) = first_err {
        return Err(e);
    }
    if cancel.load(Ordering::SeqCst) {
        return Err(Error::Cancelled);
    }

    Ok(())
}

struct CountingSink<'a, W: tokio::io::AsyncWrite + Unpin + Send> {
    inner: &'a mut W,
    state: Arc<FileState>,
}

impl<'a, W: tokio::io::AsyncWrite + Unpin + Send> CountingSink<'a, W> {
    fn new(inner: &'a mut W, state: Arc<FileState>) -> Self {
        Self { inner, state }
    }
}

impl<'a, W: tokio::io::AsyncWrite + Unpin + Send> tokio::io::AsyncWrite for CountingSink<'a, W> {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        let state = self.state.clone();
        let inner = std::pin::Pin::new(&mut *self.inner);
        match inner.poll_write(cx, buf) {
            std::task::Poll::Ready(Ok(n)) => {
                state
                    .downloaded_bytes
                    .fetch_add(n as u64, Ordering::Relaxed);
                std::task::Poll::Ready(Ok(n))
            }
            other => other,
        }
    }

    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let inner_ref: &mut W = self.get_mut().inner;
        std::pin::Pin::new(inner_ref).poll_flush(cx)
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let inner_ref: &mut W = self.get_mut().inner;
        std::pin::Pin::new(inner_ref).poll_shutdown(cx)
    }
}
