use std::fs;
use std::io::Read;
use std::path::PathBuf;

use fs2::FileExt;

use crate::config::StoreConfig;
use crate::error::{Error, Result};
use crate::manifest::{ModelManifest, ModelType};
use crate::mapping::canonicalize_model_name;
use crate::paths::{resolve_model_paths, ModelPaths};
use crate::validation::{validate_model_name, validate_relative_file};

pub const MANIFEST_FILE: &str = "geniex.json";

/// Maximum allowed size for a geniex.json manifest, to prevent OOM via
/// a maliciously-crafted file. Real manifests are typically < 10 KB.
const MANIFEST_MAX_BYTES: u64 = 1024 * 1024;

/// On-disk filename used for the per-model exclusive lock.
const LOCK_FILE: &str = ".lock";

/// Sentinel directory name used during an in-progress pull; its presence
/// marks a model as incomplete so `list()` can filter it out.
pub const INFLIGHT_DIR: &str = ".inflight";

pub struct Store {
    cfg: StoreConfig,
}

impl Store {
    pub fn new(cfg: StoreConfig) -> Result<Self> {
        fs::create_dir_all(cfg.models_dir())?;
        Ok(Self { cfg })
    }

    pub fn config(&self) -> &StoreConfig {
        &self.cfg
    }

    /// Run `f` while holding an exclusive file lock for this model.
    ///
    /// The lock is cross-process (uses `flock(LOCK_EX)` on Unix /
    /// LockFileEx on Windows via `fs2`). This matches the Go CLI which
    /// uses `github.com/gofrs/flock` for the same purpose.
    pub fn with_model_lock<T>(&self, name: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let lock_file = self.acquire_model_lock(name)?;
        let result = f();
        drop(lock_file); // explicit for clarity; Drop releases the OS lock
        result
    }

    /// Async variant of [`Self::with_model_lock`]. Acquires the OS lock
    /// synchronously (it's a single syscall + file create, fast enough
    /// on a tokio worker thread), then runs the async body while
    /// holding it. Lock is released when the returned future completes
    /// and the `lock_file` RAII handle drops.
    pub async fn with_model_lock_async<F, Fut, T>(&self, name: &str, f: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let lock_file = self.acquire_model_lock(name)?;
        let result = f().await;
        drop(lock_file);
        result
    }

    fn acquire_model_lock(&self, name: &str) -> Result<fs::File> {
        validate_model_name(name)?;
        let dir = self.cfg.model_dir(name);
        fs::create_dir_all(&dir)?;
        let lock_path = dir.join(LOCK_FILE);
        let lock_file = fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&lock_path)?;
        lock_file.lock_exclusive()?;
        Ok(lock_file)
    }

    pub fn list(&self) -> Result<Vec<ModelManifest>> {
        let models_dir = self.cfg.models_dir();
        let mut manifests = Vec::new();

        // Walk two levels: org/ and repo/
        let org_iter = match fs::read_dir(&models_dir) {
            Ok(it) => it,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(manifests),
            Err(e) => return Err(e.into()),
        };

        for org_entry in org_iter.flatten() {
            if !org_entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let repo_iter = match fs::read_dir(org_entry.path()) {
                Ok(it) => it,
                Err(_) => continue,
            };
            for repo_entry in repo_iter.flatten() {
                if !repo_entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    continue;
                }
                // Skip in-progress pulls.
                if repo_entry.path().join(INFLIGHT_DIR).exists() {
                    continue;
                }
                let manifest_path = repo_entry.path().join(MANIFEST_FILE);
                if !manifest_path.exists() {
                    continue;
                }
                match read_manifest(&manifest_path) {
                    Ok(m) => manifests.push(m),
                    Err(e) => {
                        // Log + skip corrupted manifests rather than failing the
                        // whole listing.
                        eprintln!(
                            "[model-manager] skipping corrupted manifest {}: {}",
                            manifest_path.display(),
                            e
                        );
                    }
                }
            }
        }

        Ok(manifests)
    }

    pub fn get_manifest(&self, name: &str) -> Result<ModelManifest> {
        validate_model_name(name)?;
        let path = self.cfg.model_dir(name).join(MANIFEST_FILE);
        if !path.exists() {
            return Err(Error::ModelNotFound(name.to_string()));
        }
        read_manifest(&path)
    }

    pub fn write_manifest(&self, manifest: &ModelManifest) -> Result<()> {
        validate_model_name(&manifest.name)?;
        let dir = self.cfg.model_dir(&manifest.name);
        fs::create_dir_all(&dir)?;
        let path = dir.join(MANIFEST_FILE);
        let data = serde_json::to_string(manifest)?;
        fs::write(&path, data)?;
        Ok(())
    }

    pub fn remove(&self, name: &str) -> Result<()> {
        validate_model_name(name)?;
        self.with_model_lock(name, || {
            let dir = self.cfg.model_dir(name);
            if dir.exists() {
                fs::remove_dir_all(&dir)?;
            }
            Ok(())
        })
    }

    /// Remove all cached models. Returns the count of removed model directories.
    pub fn clean(&self) -> Result<i32> {
        let manifests = self.list()?;
        let mut count = 0i32;
        for m in manifests {
            self.remove(&m.name)?;
            count += 1;
        }
        Ok(count)
    }

    pub fn get_model_type(&self, name: &str) -> Result<ModelType> {
        Ok(self.get_manifest(name)?.model_type)
    }

    /// Resolve a model name (with optional ":quant" suffix) to ModelPaths.
    ///
    /// Accepts a bare name (no '/') and canonicalises it to `aihub/<name>`
    /// so callers can refer to AI Hub models without the prefix.
    pub fn get_paths(&self, name_with_quant: &str) -> Result<(String, ModelPaths)> {
        let (name, quant) = split_quant(name_with_quant);
        let name = canonicalize_model_name(name);
        validate_model_name(&name)?;
        let manifest = self.get_manifest(&name)?;
        let base_dir = self.cfg.model_dir(&name);
        resolve_model_paths(&manifest, &base_dir, quant.as_deref())
    }

    pub fn model_file_path(&self, name: &str, file: &str) -> Result<PathBuf> {
        validate_model_name(name)?;
        if !file.is_empty() {
            validate_relative_file(file)?;
        }
        Ok(self.cfg.model_file_path(name, file))
    }
}

/// Read and parse `geniex.json` with a hard size cap.
fn read_manifest(path: &std::path::Path) -> Result<ModelManifest> {
    let file = fs::File::open(path)?;
    let mut reader = file.take(MANIFEST_MAX_BYTES);
    let mut data = String::new();
    reader.read_to_string(&mut data)?;
    Ok(serde_json::from_str(&data)?)
}

/// Split "org/repo:quant" into ("org/repo", Some("quant")) or ("org/repo", None).
fn split_quant(s: &str) -> (&str, Option<String>) {
    if let Some(pos) = s.rfind(':') {
        let name = &s[..pos];
        let quant = &s[pos + 1..];
        if quant.is_empty() {
            (name, None)
        } else {
            (name, Some(quant.to_string()))
        }
    } else {
        (s, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ModelFileInfo, ModelType};
    use std::collections::HashMap;

    fn make_store() -> Store {
        // leak the TempDir so the directory persists for the test duration
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().to_path_buf();
        std::mem::forget(tmp);
        let cfg = StoreConfig::new(path);
        Store::new(cfg).unwrap()
    }

    fn sample_manifest(name: &str) -> ModelManifest {
        let mut model_file = HashMap::new();
        model_file.insert(
            "Q4_K_M".to_string(),
            ModelFileInfo {
                name: "model-Q4_K_M.gguf".to_string(),
                downloaded: true,
                size: 100,
            },
        );
        ModelManifest {
            name: name.to_string(),
            model_name: "test-1b".to_string(),
            model_type: ModelType::Llm,
            plugin_id: "llama_cpp".to_string(),
            precision: String::new(),
            model_file,
            mmproj_file: ModelFileInfo::default(),
            tokenizer_file: ModelFileInfo::default(),
            extra_files: vec![],
        }
    }

    #[test]
    fn roundtrip_manifest() {
        let store = make_store();
        let m = sample_manifest("TestOrg/TestRepo");
        store.write_manifest(&m).unwrap();
        let loaded = store.get_manifest("TestOrg/TestRepo").unwrap();
        assert_eq!(loaded.name, "TestOrg/TestRepo");
    }

    #[test]
    fn list_returns_written_manifests() {
        let store = make_store();
        store.write_manifest(&sample_manifest("Org/A")).unwrap();
        store.write_manifest(&sample_manifest("Org/B")).unwrap();
        let list = store.list().unwrap();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn remove_deletes_directory() {
        let store = make_store();
        store.write_manifest(&sample_manifest("Org/C")).unwrap();
        store.remove("Org/C").unwrap();
        assert!(!store.cfg.model_dir("Org/C").exists());
    }

    #[test]
    fn clean_returns_count() {
        let store = make_store();
        store.write_manifest(&sample_manifest("Org/D")).unwrap();
        store.write_manifest(&sample_manifest("Org/E")).unwrap();
        let n = store.clean().unwrap();
        assert_eq!(n, 2);
    }

    #[test]
    fn list_skips_inflight() {
        let store = make_store();
        // write a valid manifest + an inflight marker in the same dir
        store.write_manifest(&sample_manifest("Org/F")).unwrap();
        let dir = store.cfg.model_dir("Org/F");
        fs::create_dir_all(dir.join(INFLIGHT_DIR)).unwrap();
        let list = store.list().unwrap();
        assert!(list.iter().all(|m| m.name != "Org/F"));
    }

    #[test]
    fn list_skips_corrupted() {
        let store = make_store();
        store.write_manifest(&sample_manifest("Org/Good")).unwrap();
        let bad_dir = store.cfg.model_dir("Org/Bad");
        fs::create_dir_all(&bad_dir).unwrap();
        fs::write(bad_dir.join(MANIFEST_FILE), "not valid json").unwrap();
        let list = store.list().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "Org/Good");
    }

    #[test]
    fn rejects_invalid_names() {
        let store = make_store();
        assert!(store.get_manifest("../etc").is_err());
        assert!(store.remove("a/..").is_err());
        assert!(store.model_file_path("Org/Foo", "../outside").is_err());
    }

    #[test]
    fn rejects_oversized_manifest() {
        let store = make_store();
        let dir = store.cfg.model_dir("Org/Huge");
        fs::create_dir_all(&dir).unwrap();
        let big = "x".repeat((MANIFEST_MAX_BYTES + 1) as usize);
        fs::write(dir.join(MANIFEST_FILE), big).unwrap();
        // Size-capped reader will truncate, then JSON parse fails — either way, Err.
        assert!(store.get_manifest("Org/Huge").is_err());
    }
}
