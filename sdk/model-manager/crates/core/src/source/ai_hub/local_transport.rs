//! [`HttpTransport`] adapter backed by a single local file.
//!
//! [`fetch_central_directory`](super::remote_zip::fetch_central_directory)
//! is written against the [`HttpTransport`] trait so any byte source —
//! reqwest, wiremock, or a local file — can drive it. This module
//! provides the third option: when the user pulls from a `.zip` already
//! sitting on disk, we wrap the file as a transport whose `head`
//! returns its length and whose `get_range` becomes a `seek` + `read`.
//! The `Url` parameter is ignored.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use tokio::io::{AsyncSeekExt, AsyncWrite, AsyncWriteExt};
use url::Url;

use crate::error::{Error, Result};
use crate::transport::{HeadInfo, HttpTransport};

#[derive(Debug)]
pub struct LocalFileTransport {
    path: PathBuf,
    len: u64,
}

impl LocalFileTransport {
    pub fn open(path: &Path) -> Result<Self> {
        let meta = std::fs::metadata(path).map_err(|e| {
            Error::Hub(format!(
                "local archive {} is not accessible: {e}",
                path.display()
            ))
        })?;
        if !meta.is_file() {
            return Err(Error::Hub(format!(
                "local archive {} is not a regular file",
                path.display()
            )));
        }
        Ok(Self {
            path: path.to_path_buf(),
            len: meta.len(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[async_trait]
impl HttpTransport for LocalFileTransport {
    async fn head(&self, _url: &Url, _auth: Option<&str>) -> Result<HeadInfo> {
        Ok(HeadInfo {
            size: self.len,
            accepts_ranges: true,
            etag: None,
        })
    }

    async fn get_range(
        &self,
        _url: &Url,
        _auth: Option<&str>,
        offset: u64,
        len: u64,
        sink: &mut (dyn AsyncWrite + Unpin + Send),
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        if offset
            .checked_add(len)
            .map(|end| end > self.len)
            .unwrap_or(true)
        {
            return Err(Error::Hub(format!(
                "local range {offset}+{len} exceeds archive size {}",
                self.len
            )));
        }

        let mut file = tokio::fs::File::open(&self.path).await?;
        file.seek(std::io::SeekFrom::Start(offset)).await?;
        let mut remaining = len;
        let mut buf = vec![0u8; 64 * 1024];
        while remaining > 0 {
            let want = remaining.min(buf.len() as u64) as usize;
            let n = tokio::io::AsyncReadExt::read(&mut file, &mut buf[..want]).await?;
            if n == 0 {
                return Err(Error::Hub(format!(
                    "local archive {}: unexpected EOF at offset {}",
                    self.path.display(),
                    offset + (len - remaining)
                )));
            }
            sink.write_all(&buf[..n])
                .await
                .map_err(|e| Error::Http(format!("write sink: {e}")))?;
            remaining -= n as u64;
        }
        sink.flush()
            .await
            .map_err(|e| Error::Http(format!("flush sink: {e}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    async fn read_all(t: &Arc<dyn HttpTransport>, off: u64, len: u64) -> Vec<u8> {
        let dummy = Url::parse("file:///dummy").unwrap();
        let mut buf: Vec<u8> = Vec::new();
        t.get_range(&dummy, None, off, len, &mut buf).await.unwrap();
        buf
    }

    #[tokio::test]
    async fn head_returns_file_length() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.bin");
        std::fs::write(&p, b"abcdef").unwrap();
        let t = LocalFileTransport::open(&p).unwrap();
        let dummy = Url::parse("file:///dummy").unwrap();
        let info = t.head(&dummy, None).await.unwrap();
        assert_eq!(info.size, 6);
        assert!(info.accepts_ranges);
    }

    #[tokio::test]
    async fn get_range_returns_exact_slice() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.bin");
        std::fs::write(&p, b"abcdef").unwrap();
        let t: Arc<dyn HttpTransport> = Arc::new(LocalFileTransport::open(&p).unwrap());
        assert_eq!(read_all(&t, 0, 3).await, b"abc");
        assert_eq!(read_all(&t, 2, 4).await, b"cdef");
        assert_eq!(read_all(&t, 5, 1).await, b"f");
    }

    #[tokio::test]
    async fn out_of_bounds_range_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.bin");
        std::fs::write(&p, b"abcdef").unwrap();
        let t = LocalFileTransport::open(&p).unwrap();
        let dummy = Url::parse("file:///dummy").unwrap();
        let mut buf: Vec<u8> = Vec::new();
        let err = t
            .get_range(&dummy, None, 5, 10, &mut buf)
            .await
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("exceeds"), "msg: {msg}");
    }

    #[tokio::test]
    async fn missing_file_rejected_at_open() {
        let err = LocalFileTransport::open(Path::new("/nonexistent/zzz.zip")).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not accessible"), "msg: {msg}");
    }
}
