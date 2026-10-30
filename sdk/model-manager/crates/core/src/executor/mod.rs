// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Byte-moving executor for [`crate::source::Plan`]s: materialises each
//! [`FileSpec`] into `dest_dir/<name>`. Raw byte ranges (HTTP or local)
//! share one chunked pipeline with `.progress` bitmap resume; DEFLATE
//! entries share one fetch-then-inflate pipeline; whole local files are
//! copied directly.

pub mod chunk;

use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::fs::OpenOptions;
use tokio::io::{AsyncSeekExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use url::Url;

use crate::error::{Error, Result};
use crate::resume::marker_path;
use crate::source::{BytesSource, FileSpec};
use crate::transport::{HttpTransport, LocalFileTransport};

pub use crate::resume::PROGRESS_SUFFIX;

/// Per-file download progress. `total_bytes == -1` means the total is unknown.
#[derive(Debug, Clone)]
pub struct FileProgress {
    pub file_name: String,
    pub downloaded_bytes: i64,
    pub total_bytes: i64,
}

/// Called periodically during a pull. Returning `false` flips the
/// cooperative cancel flag; every in-flight worker bails on its next
/// chunk / entry boundary.
pub type ProgressCallback = Box<dyn Fn(&[FileProgress]) -> bool + Send + Sync>;

#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    pub file_concurrency: usize,
    pub chunk_concurrency: usize,
    pub progress_interval: Duration,
}

impl ExecutorConfig {
    /// Resolve from `GENIEX_DL_*` env overrides.
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

#[derive(Clone)]
struct StopSignals {
    user: CancellationToken,
    failed: CancellationToken,
}

impl StopSignals {
    fn new() -> Self {
        Self {
            user: CancellationToken::new(),
            failed: CancellationToken::new(),
        }
    }

    fn halt_new_work(&self) -> bool {
        self.user.is_cancelled() || self.failed.is_cancelled()
    }

    async fn race(&self, fetch: impl Future<Output = Result<()>>) -> Result<()> {
        tokio::select! {
            biased;
            _ = self.user.cancelled() => Err(Error::Cancelled),
            res = fetch => res,
        }
    }
}

struct FileState {
    name: String,
    total_bytes: AtomicU64,
    downloaded_bytes: AtomicU64,
}

impl FileState {
    fn snapshot(&self) -> FileProgress {
        let total = self.total_bytes.load(Ordering::Relaxed);
        let downloaded = self.downloaded_bytes.load(Ordering::Relaxed);
        // Clamp: scaled DEFLATE counters can overshoot `total` by a
        // rounding byte or two before the post-decode pin snaps them
        // back. A UI showing "101%" is worse than showing "100%".
        let visible = if total == u64::MAX {
            downloaded
        } else {
            downloaded.min(total)
        };
        FileProgress {
            file_name: self.name.clone(),
            downloaded_bytes: visible as i64,
            total_bytes: if total == u64::MAX { -1 } else { total as i64 },
        }
    }
}

pub struct Executor {
    transport: Arc<dyn HttpTransport>,
    cfg: ExecutorConfig,
}

impl Executor {
    pub fn new(transport: Arc<dyn HttpTransport>, default_file_concurrency: usize) -> Self {
        Self {
            cfg: ExecutorConfig::resolve(default_file_concurrency),
            transport,
        }
    }

    pub fn with_config(transport: Arc<dyn HttpTransport>, cfg: ExecutorConfig) -> Self {
        Self { transport, cfg }
    }

    pub async fn run(
        &self,
        files: &[FileSpec],
        dest_dir: &Path,
        on_progress: Option<&ProgressCallback>,
    ) -> Result<()> {
        tokio::fs::create_dir_all(dest_dir).await?;

        let states: Vec<Arc<FileState>> = files
            .iter()
            .map(|s| {
                Arc::new(FileState {
                    name: s.name.clone(),
                    total_bytes: AtomicU64::new(if s.size == 0 { u64::MAX } else { s.size }),
                    downloaded_bytes: AtomicU64::new(0),
                })
            })
            .collect();

        let stop = StopSignals::new();
        let file_sem = Arc::new(Semaphore::new(self.cfg.file_concurrency));
        let chunk_sem = Arc::new(Semaphore::new(self.cfg.chunk_concurrency));

        let mut file_tasks: JoinSet<Result<()>> = JoinSet::new();
        let transport = self.transport.clone();
        let dest_owned = dest_dir.to_path_buf();

        for (spec, state) in files.iter().cloned().zip(states.iter().cloned()) {
            let file_sem = file_sem.clone();
            let chunk_sem = chunk_sem.clone();
            let transport = transport.clone();
            let dest_dir = dest_owned.clone();
            let stop = stop.clone();
            file_tasks.spawn(async move {
                let _permit = file_sem
                    .acquire_owned()
                    .await
                    .map_err(|e| Error::Http(format!("file semaphore closed: {e}")))?;
                run_one(spec, state, &dest_dir, transport, chunk_sem, stop).await
            });
        }

        let mut first_err: Option<Error> = None;
        let mut ticker = tokio::time::interval(self.cfg.progress_interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        ticker.tick().await;

        loop {
            tokio::select! {
                biased;
                maybe = file_tasks.join_next() => {
                    match maybe {
                        None => break,
                        Some(Ok(Ok(()))) => {}
                        Some(Ok(Err(e))) => {
                            stop.failed.cancel();
                            first_err.get_or_insert(e);
                        }
                        Some(Err(join_err)) => {
                            stop.failed.cancel();
                            first_err.get_or_insert(Error::Http(format!("join: {join_err}")));
                        }
                    }
                }
                _ = ticker.tick() => {
                    if let Some(cb) = on_progress {
                        let snaps: Vec<FileProgress> =
                            states.iter().map(|s| s.snapshot()).collect();
                        if !(cb)(&snaps) {
                            stop.user.cancel();
                        }
                    }
                }
            }
        }

        if let Some(cb) = on_progress {
            let snaps: Vec<FileProgress> = states.iter().map(|s| s.snapshot()).collect();
            let _ = (cb)(&snaps);
        }

        if let Some(e) = first_err {
            return Err(e);
        }
        if stop.user.is_cancelled() {
            return Err(Error::Cancelled);
        }
        Ok(())
    }
}

/// Where a range pipeline reads its bytes: an HTTP object, or a local file
/// served through [`LocalFileTransport`].
struct RangeSrc {
    transport: Arc<dyn HttpTransport>,
    url: Url,
    auth: Option<String>,
    offset: u64,
}

impl RangeSrc {
    fn http(
        transport: &Arc<dyn HttpTransport>,
        url: Url,
        auth: Option<String>,
        offset: u64,
    ) -> Self {
        Self {
            transport: transport.clone(),
            url,
            auth,
            offset,
        }
    }

    fn local(path: &Path, offset: u64) -> Result<Self> {
        Ok(Self {
            transport: Arc::new(LocalFileTransport::open(path)?),
            url: Url::parse("file:///local").expect("static url"),
            auth: None,
            offset,
        })
    }

    async fn fetch(
        &self,
        offset: u64,
        len: u64,
        sink: &mut (dyn AsyncWrite + Unpin + Send),
    ) -> Result<()> {
        self.transport
            .get_range(
                &self.url,
                self.auth.as_deref(),
                self.offset + offset,
                len,
                sink,
            )
            .await
    }
}

struct Job {
    name: String,
    state: Arc<FileState>,
    output: PathBuf,
    marker: PathBuf,
    chunk_sem: Arc<Semaphore>,
    stop: StopSignals,
}

async fn run_one(
    spec: FileSpec,
    state: Arc<FileState>,
    dest_dir: &Path,
    transport: Arc<dyn HttpTransport>,
    chunk_sem: Arc<Semaphore>,
    stop: StopSignals,
) -> Result<()> {
    let job = Job {
        output: dest_dir.join(&spec.name),
        marker: marker_path(dest_dir, &spec.name),
        name: spec.name,
        state,
        chunk_sem,
        stop,
    };
    let known = |n: u64| (n > 0).then_some(n);
    match spec.bytes {
        BytesSource::Http { url, auth } => {
            let src = RangeSrc::http(&transport, url, auth, 0);
            download_raw(&job, src, known(spec.size), false).await
        }
        BytesSource::HttpRange {
            url,
            auth,
            offset,
            len,
        } => {
            let src = RangeSrc::http(&transport, url, auth, offset);
            download_raw(&job, src, Some(len), false).await
        }
        BytesSource::LocalRange { path, offset, len } => {
            download_raw(&job, RangeSrc::local(&path, offset)?, Some(len), true).await
        }
        BytesSource::HttpDeflate {
            url,
            auth,
            offset,
            compressed_len,
        } => {
            let src = RangeSrc::http(&transport, url, auth, offset);
            download_deflate(&job, src, compressed_len, spec.size).await
        }
        BytesSource::LocalDeflate {
            path,
            offset,
            compressed_len,
        } => {
            let src = RangeSrc::local(&path, offset)?;
            download_deflate(&job, src, compressed_len, spec.size).await
        }
        BytesSource::Local { path } => copy_whole(&job, &path).await,
    }
}

fn finish_entry(job: &Job, size: u64) -> Result<()> {
    job.state.downloaded_bytes.store(size, Ordering::Relaxed);
    std::fs::write(&job.marker, [chunk::PROGRESS_DONE_BYTE])?;
    Ok(())
}

async fn copy_whole(job: &Job, src: &Path) -> Result<()> {
    if let Some(parent) = job.output.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let size = tokio::fs::copy(src, &job.output).await?;
    job.state.total_bytes.store(size, Ordering::Relaxed);
    finish_entry(job, size)
}

/// Chunked parallel range download with `.progress` bitmap resume.
/// `single_chunk` keeps local copies sequential with a one-byte marker.
async fn download_raw(
    job: &Job,
    src: RangeSrc,
    size: Option<u64>,
    single_chunk: bool,
) -> Result<()> {
    let size = match size {
        Some(s) => s,
        None => {
            src.transport
                .head(&src.url, src.auth.as_deref())
                .await?
                .size
        }
    };
    job.state.total_bytes.store(size, Ordering::Relaxed);

    let plan = if single_chunk {
        chunk::single_chunk_plan(size)
    } else {
        chunk::plan_chunks(size)
    };
    let mut bitmap = chunk::load_or_init_bitmap(&job.marker, &plan)?;
    let output_size = tokio::fs::metadata(&job.output).await.ok().map(|m| m.len());
    if chunk::bitmap_complete(&bitmap) && output_size != Some(size) {
        bitmap.fill(0);
        tokio::fs::write(&job.marker, &bitmap).await?;
    }
    chunk::preallocate(&job.output, size)?;
    job.state
        .downloaded_bytes
        .store(chunk::bytes_already_done(&plan, &bitmap), Ordering::Relaxed);

    let pending = chunk::pending_chunks(&plan, &bitmap);
    if pending.is_empty() {
        return Ok(());
    }
    let src = Arc::new(src);
    let mut tasks: JoinSet<Result<()>> = JoinSet::new();
    for range in pending {
        if job.stop.halt_new_work() {
            break;
        }
        let (sem, src, stop) = (job.chunk_sem.clone(), src.clone(), job.stop.clone());
        let (output, marker, state) = (job.output.clone(), job.marker.clone(), job.state.clone());
        tasks.spawn(async move {
            if stop.halt_new_work() {
                return Err(Error::Cancelled);
            }
            let _permit = sem
                .acquire_owned()
                .await
                .map_err(|e| Error::Http(format!("chunk semaphore closed: {e}")))?;
            if stop.halt_new_work() {
                return Err(Error::Cancelled);
            }
            let mut file = OpenOptions::new()
                .write(true)
                .read(true)
                .open(&output)
                .await?;
            file.seek(std::io::SeekFrom::Start(range.offset)).await?;
            let mut counted = CountingSink::new(&mut file, state, 1.0);
            stop.race(src.fetch(range.offset, range.len, &mut counted))
                .await?;
            file.flush().await?;
            chunk::mark_chunk_done(&marker, range.index)
        });
    }

    let mut first_err: Option<Error> = None;
    while let Some(res) = tasks.join_next().await {
        let err = match res {
            Ok(Ok(())) => continue,
            Ok(Err(e)) => e,
            Err(join_err) => Error::Http(format!("chunk join: {join_err}")),
        };
        job.stop.failed.cancel();
        first_err.get_or_insert(err);
    }
    if let Some(e) = first_err {
        return Err(e);
    }
    if job.stop.user.is_cancelled() {
        return Err(Error::Cancelled);
    }
    Ok(())
}

/// DEFLATE entries resume all-or-nothing: flate2 streams aren't seekable.
async fn download_deflate(job: &Job, src: RangeSrc, compressed_len: u64, size: u64) -> Result<()> {
    job.state.total_bytes.store(size, Ordering::Relaxed);
    if chunk::marker_complete(&job.marker, &job.output, size) {
        job.state.downloaded_bytes.store(size, Ordering::Relaxed);
        return Ok(());
    }
    if job.stop.halt_new_work() {
        return Err(Error::Cancelled);
    }

    // Credit the compressed fetch at the uncompressed scale the UI reports.
    let scale = if compressed_len == 0 {
        1.0
    } else {
        size as f64 / compressed_len as f64
    };
    let mut compressed = Vec::with_capacity(compressed_len as usize);
    let mut counted = CountingSink::new(&mut compressed, job.state.clone(), scale);
    job.stop
        .race(src.fetch(0, compressed_len, &mut counted))
        .await?;
    if job.stop.user.is_cancelled() {
        return Err(Error::Cancelled);
    }

    let decoded = inflate_to_file(compressed, job.output.clone()).await?;
    if decoded != size {
        return Err(Error::Hub(format!(
            "deflate short decode for {}: got {decoded} / expected {size}",
            job.name
        )));
    }
    finish_entry(job, size)
}

async fn inflate_to_file(compressed: Vec<u8>, output: PathBuf) -> Result<u64> {
    tokio::task::spawn_blocking(move || -> Result<u64> {
        use std::io::Write;
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::File::create(&output)?;
        let mut dec = flate2::write::DeflateDecoder::new(&mut file);
        dec.write_all(&compressed)
            .map_err(|e| Error::Hub(format!("inflate {}: {e}", output.display())))?;
        let written = dec.total_out();
        dec.finish()
            .map_err(|e| Error::Hub(format!("inflate finish {}: {e}", output.display())))?;
        file.sync_all()?;
        Ok(written)
    })
    .await
    .map_err(|e| Error::Http(format!("deflate join: {e}")))?
}

/// Credits `written * scale` bytes to `downloaded_bytes` on every write.
struct CountingSink<'a, W: AsyncWrite + Unpin + Send> {
    inner: &'a mut W,
    state: Arc<FileState>,
    scale: f64,
}

impl<'a, W: AsyncWrite + Unpin + Send> CountingSink<'a, W> {
    fn new(inner: &'a mut W, state: Arc<FileState>, scale: f64) -> Self {
        Self {
            inner,
            state,
            scale,
        }
    }
}

impl<W: AsyncWrite + Unpin + Send> AsyncWrite for CountingSink<'_, W> {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        let state = self.state.clone();
        let scale = self.scale;
        let inner = std::pin::Pin::new(&mut *self.inner);
        match inner.poll_write(cx, buf) {
            std::task::Poll::Ready(Ok(n)) => {
                let credited = ((n as f64) * scale).round() as u64;
                state
                    .downloaded_bytes
                    .fetch_add(credited, Ordering::Relaxed);
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
