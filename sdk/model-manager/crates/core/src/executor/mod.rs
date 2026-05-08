//! Byte-moving executor for [`crate::source::Plan`]s.
//!
//! The executor is hub-agnostic: it takes a slice of [`FileSpec`]s and
//! materialises each one into `dest_dir/<name>`. Four branches cover
//! every hub we support today:
//!
//! - [`BytesSource::Http`] — chunked parallel download with `.progress`
//!   bitmap resume. One URL per file (HF).
//! - [`BytesSource::HttpRange`] — same chunking/resume machinery but
//!   every range request is offset into a larger remote object (a
//!   STORED zip entry inside an AI Hub asset).
//! - [`BytesSource::HttpDeflate`] — single-range fetch + inline flate2
//!   decode. Resume is entry-granular because DEFLATE is not seekable.
//! - [`BytesSource::Local`] — plain copy for `LocalFsSource`.

pub mod chunk;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::fs::OpenOptions;
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::error::{Error, Result};
use crate::source::{BytesSource, FileSpec};
use crate::transport::HttpTransport;

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

        let cancel = Arc::new(AtomicBool::new(false));
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
            let cancel = cancel.clone();
            file_tasks.spawn(async move {
                let _permit = file_sem
                    .acquire_owned()
                    .await
                    .map_err(|e| Error::Http(format!("file semaphore closed: {e}")))?;
                run_one(spec, state, &dest_dir, transport, chunk_sem, cancel).await
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
                            cancel.store(true, Ordering::SeqCst);
                            first_err.get_or_insert(e);
                        }
                        Some(Err(join_err)) => {
                            cancel.store(true, Ordering::SeqCst);
                            first_err.get_or_insert(Error::Http(format!("join: {join_err}")));
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

async fn run_one(
    spec: FileSpec,
    state: Arc<FileState>,
    dest_dir: &Path,
    transport: Arc<dyn HttpTransport>,
    chunk_sem: Arc<Semaphore>,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    match spec.bytes.clone() {
        BytesSource::Http { url, auth } => {
            download_range_based(
                &spec.name,
                spec.size,
                state,
                dest_dir,
                transport,
                chunk_sem,
                cancel,
                url,
                auth,
                /*base_offset*/ 0,
                /*size_provided*/ spec.size > 0,
            )
            .await
        }
        BytesSource::HttpRange {
            url,
            auth,
            offset,
            len,
        } => {
            let size = if len > 0 { len } else { spec.size };
            download_range_based(
                &spec.name, size, state, dest_dir, transport, chunk_sem, cancel, url, auth, offset,
                true,
            )
            .await
        }
        BytesSource::HttpDeflate {
            url,
            auth,
            offset,
            compressed_len,
        } => {
            download_http_deflate(
                &spec.name,
                spec.size,
                compressed_len,
                offset,
                state,
                dest_dir,
                transport,
                cancel,
                url,
                auth,
            )
            .await
        }
        BytesSource::Local { path } => {
            let dest = dest_dir.join(&spec.name);
            if let Some(parent) = dest.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::copy(&path, &dest).await?;
            let size = tokio::fs::metadata(&dest)
                .await
                .map(|m| m.len())
                .unwrap_or(0);
            state.total_bytes.store(size, Ordering::Relaxed);
            state.downloaded_bytes.store(size, Ordering::Relaxed);
            let marker = PathBuf::from(format!("{}{}", dest.display(), PROGRESS_SUFFIX));
            tokio::fs::write(&marker, [chunk::PROGRESS_DONE_BYTE]).await?;
            Ok(())
        }
    }
}

/// Chunked parallel download for `Http` and `HttpRange`. The only
/// difference between them is the `base_offset` added to every range
/// request: `Http` uses 0, `HttpRange` uses the entry's start inside
/// the outer zip.
#[allow(clippy::too_many_arguments)]
async fn download_range_based(
    name: &str,
    size_hint: u64,
    state: Arc<FileState>,
    dest_dir: &Path,
    transport: Arc<dyn HttpTransport>,
    chunk_sem: Arc<Semaphore>,
    cancel: Arc<AtomicBool>,
    url: url::Url,
    auth: Option<String>,
    base_offset: u64,
    size_provided: bool,
) -> Result<()> {
    let size = if size_provided && size_hint > 0 {
        size_hint
    } else {
        let info = transport.head(&url, auth.as_deref()).await?;
        info.size
    };
    state.total_bytes.store(size, Ordering::Relaxed);

    let output_path = dest_dir.join(name);
    let marker_path = PathBuf::from(format!("{}{}", output_path.display(), PROGRESS_SUFFIX));

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
        let auth = auth.clone();
        let url = url.clone();
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
                .get_range(
                    &url,
                    auth.as_deref(),
                    base_offset + range.offset,
                    range.len,
                    &mut counted,
                )
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

/// DEFLATE-decoding download for AI Hub `.bin` shards. Entry-granular
/// resume: any partial state from a previous crash is discarded and
/// the whole entry is refetched, because flate2 streams aren't
/// seekable mid-entry.
#[allow(clippy::too_many_arguments)]
async fn download_http_deflate(
    name: &str,
    uncompressed_size: u64,
    compressed_len: u64,
    offset: u64,
    state: Arc<FileState>,
    dest_dir: &Path,
    transport: Arc<dyn HttpTransport>,
    cancel: Arc<AtomicBool>,
    url: url::Url,
    auth: Option<String>,
) -> Result<()> {
    state
        .total_bytes
        .store(uncompressed_size, Ordering::Relaxed);
    let output_path = dest_dir.join(name);
    let marker_path = PathBuf::from(format!("{}{}", output_path.display(), PROGRESS_SUFFIX));

    // DEFLATE entries are entry-granular: a 0x01 marker means the file
    // is fully decoded, anything else (partial, missing, stale) means
    // refetch the whole entry — flate2 streams aren't seekable, so
    // mid-entry resume isn't possible.
    if let Ok(data) = std::fs::read(&marker_path) {
        if data.first().copied() == Some(chunk::PROGRESS_DONE_BYTE)
            && output_path.exists()
            && std::fs::metadata(&output_path)
                .map(|m| m.len())
                .unwrap_or(0)
                == uncompressed_size
        {
            state
                .downloaded_bytes
                .store(uncompressed_size, Ordering::Relaxed);
            return Ok(());
        }
    }

    if cancel.load(Ordering::SeqCst) {
        return Err(Error::Cancelled);
    }

    // Buffering the compressed slice caps RAM at `compressed_len` per
    // in-flight deflate entry; `file_concurrency` bounds how many run
    // in parallel. Streaming decode on `spawn_blocking` because flate2
    // is sync.
    let mut compressed: Vec<u8> = Vec::with_capacity(compressed_len as usize);
    transport
        .get_range(
            &url,
            auth.as_deref(),
            offset,
            compressed_len,
            &mut compressed,
        )
        .await?;

    if cancel.load(Ordering::SeqCst) {
        return Err(Error::Cancelled);
    }

    if let Some(parent) = output_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let out_path = output_path.clone();
    let state_cl = state.clone();
    let decoded_size = tokio::task::spawn_blocking(move || -> Result<u64> {
        use flate2::write::DeflateDecoder;
        use std::io::Write as IoWrite;
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&out_path)?;
        let mut dec = DeflateDecoder::new(CountingWriter {
            inner: file,
            state: state_cl,
        });
        dec.write_all(&compressed)
            .map_err(|e| Error::Hub(format!("inflate {name:?}: {e}", name = out_path.display())))?;
        let counted = dec
            .finish()
            .map_err(|e| Error::Hub(format!("inflate finish {:?}: {e}", out_path.display())))?;
        counted.inner.sync_all()?;
        Ok(counted.state.downloaded_bytes.load(Ordering::Relaxed))
    })
    .await
    .map_err(|e| Error::Http(format!("deflate join: {e}")))??;

    if decoded_size != uncompressed_size {
        return Err(Error::Hub(format!(
            "deflate short decode for {}: got {decoded_size} / expected {uncompressed_size}",
            name
        )));
    }

    std::fs::write(&marker_path, [chunk::PROGRESS_DONE_BYTE])?;

    Ok(())
}

struct CountingWriter<W: std::io::Write> {
    inner: W,
    state: Arc<FileState>,
}

impl<W: std::io::Write> std::io::Write for CountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.state
            .downloaded_bytes
            .fetch_add(n as u64, Ordering::Relaxed);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
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
