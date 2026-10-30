// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use fs2::FileExt;

use crate::config::StoreConfig;
use crate::error::{Error, Result};
use crate::manifest::infer::quant_sort_key;
use crate::manifest::{ModelManifest, ModelType};
use crate::model_ref::{validate_name, ModelRef};

#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub model_path: PathBuf,
    pub mmproj_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub model_dir: PathBuf,
    pub model_name: String,
    pub plugin_id: String,
    pub model_type: ModelType,
}

pub const MANIFEST_FILE: &str = "geniex.json";

/// Caps a crafted manifest's allocation; real ones are < 10 KB.
const MANIFEST_MAX_BYTES: u64 = 1024 * 1024;

const LOCK_FILE: &str = ".lock";

/// Present while a pull is in progress; `list()` skips such models.
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

    /// Cross-process lock, compatible with the Go CLI's `gofrs/flock` on the same file.
    pub fn with_model_lock<T>(&self, name: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let _lock = self.acquire_model_lock(name)?;
        f()
    }

    pub async fn with_model_lock_async<F, Fut, T>(&self, name: &str, f: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let _lock = self.acquire_model_lock(name)?;
        f().await
    }

    fn acquire_model_lock(&self, name: &str) -> Result<fs::File> {
        validate_name(name)?;
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
                        crate::logging::warn(&format!(
                            "skipping corrupted manifest {}: {}",
                            manifest_path.display(),
                            e
                        ));
                    }
                }
            }
        }

        Ok(manifests)
    }

    pub fn get_manifest(&self, name: &str) -> Result<ModelManifest> {
        validate_name(name)?;
        let path = self.cfg.model_dir(name).join(MANIFEST_FILE);
        if !path.exists() {
            return Err(Error::ModelNotFound(name.to_string()));
        }
        read_manifest(&path)
    }

    pub fn write_manifest(&self, manifest: &ModelManifest) -> Result<()> {
        validate_name(&manifest.name)?;
        let dir = self.cfg.model_dir(&manifest.name);
        fs::create_dir_all(&dir)?;
        let path = dir.join(MANIFEST_FILE);
        let data = serde_json::to_string(manifest)?;
        fs::write(&path, data)?;
        Ok(())
    }

    pub fn remove(&self, name_with_quant: &str) -> Result<()> {
        // With ":quant", keep the dir unless it was the last quant: extras are shared.
        let ModelRef { name, quant } = ModelRef::parse(name_with_quant)?;
        let name = name.as_str();
        self.with_model_lock(name, || {
            let dir = self.cfg.model_dir(name);
            let Some(quant) = quant else {
                if dir.exists() {
                    fs::remove_dir_all(&dir)?;
                }
                return Ok(());
            };

            let manifest_path = dir.join(MANIFEST_FILE);
            if !manifest_path.exists() {
                return Err(Error::ModelNotFound(name.to_string()));
            }
            let mut manifest = read_manifest(&manifest_path)?;
            let file_info = manifest
                .model_file
                .get(&quant)
                .ok_or_else(|| Error::QuantNotFound(quant.clone(), name.to_string()))?
                .clone();

            if manifest.model_file.len() == 1 {
                fs::remove_dir_all(&dir)?;
                return Ok(());
            }

            if !file_info.name.is_empty() {
                let file_path = dir.join(&file_info.name);
                match fs::remove_file(&file_path) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(e.into()),
                }
            }
            manifest.model_file.remove(&quant);
            let data = serde_json::to_string(&manifest)?;
            fs::write(&manifest_path, data)?;
            Ok(())
        })
    }

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
        Ok(self.resolve_manifest(name)?.model_type)
    }

    pub fn set_model_type(&self, name: &str, model_type: ModelType) -> Result<()> {
        let name = ModelRef::parse(name)?.name;
        self.with_model_lock(&name, || {
            let mut manifest = self.get_manifest(&name)?;
            manifest.model_type = model_type;
            self.write_manifest(&manifest)
        })
    }

    pub fn resolve_manifest(&self, name_with_quant: &str) -> Result<ModelManifest> {
        self.get_manifest(&ModelRef::parse(name_with_quant)?.name)
    }

    /// Skips in-progress pulls to agree with [`Self::list`]; the load path does not.
    pub fn resolve_detail_manifest(&self, name_with_quant: &str) -> Result<ModelManifest> {
        let manifest = self.resolve_manifest(name_with_quant)?;
        if self
            .cfg
            .model_dir(&manifest.name)
            .join(INFLIGHT_DIR)
            .exists()
        {
            return Err(Error::ModelNotFound(manifest.name));
        }
        Ok(manifest)
    }

    pub fn get_paths(&self, name_with_quant: &str) -> Result<(String, ModelPaths)> {
        let ModelRef { name, quant } = ModelRef::parse(name_with_quant)?;
        let manifest = self.get_manifest(&name)?;
        let base_dir = self.cfg.model_dir(&name);
        resolve_model_paths(&manifest, &base_dir, quant.as_deref())
    }

    pub fn model_dir(&self, name: &str) -> Result<PathBuf> {
        validate_name(name)?;
        Ok(self.cfg.model_dir(name))
    }
}

/// `quant: None` picks the best quant already on disk; auto-load passes bare names.
pub(crate) fn resolve_model_paths(
    manifest: &ModelManifest,
    base_dir: &Path,
    quant: Option<&str>,
) -> Result<(String, ModelPaths)> {
    let quant = match quant {
        Some(q) => q.to_string(),
        None => pick_downloaded_by_priority(manifest)?,
    };
    let fi = manifest
        .model_file
        .get(&quant)
        .ok_or_else(|| Error::QuantNotFound(quant.clone(), manifest.name.clone()))?;
    if !fi.downloaded {
        return Err(Error::QuantNotDownloaded(quant, manifest.name.clone()));
    }
    let model_path = base_dir.join(&fi.name);
    let mmproj_path =
        (!manifest.mmproj_file.name.is_empty()).then(|| base_dir.join(&manifest.mmproj_file.name));
    let tokenizer_path = (!manifest.tokenizer_file.name.is_empty())
        .then(|| base_dir.join(&manifest.tokenizer_file.name));
    Ok((
        quant,
        ModelPaths {
            model_path,
            mmproj_path,
            tokenizer_path,
            model_dir: base_dir.to_path_buf(),
            model_name: manifest.model_name.clone(),
            plugin_id: manifest.plugin_id.clone(),
            model_type: manifest.model_type.clone(),
        },
    ))
}

fn pick_downloaded_by_priority(manifest: &ModelManifest) -> Result<String> {
    manifest
        .model_file
        .iter()
        .filter(|(_, v)| v.downloaded)
        .map(|(k, _)| k.as_str())
        .min_by(|a, b| quant_sort_key(a).cmp(&quant_sort_key(b)))
        .map(str::to_string)
        .ok_or_else(|| Error::ModelNotFound(manifest.name.clone()))
}

fn read_manifest(path: &std::path::Path) -> Result<ModelManifest> {
    let file = fs::File::open(path)?;
    let mut reader = file.take(MANIFEST_MAX_BYTES);
    let mut data = String::new();
    reader.read_to_string(&mut data)?;
    let mut manifest: ModelManifest = serde_json::from_str(&data)?;
    if migrate_legacy_qairt_precision(&mut manifest) {
        if let Ok(rewritten) = serde_json::to_string(&manifest) {
            let _ = fs::write(path, rewritten);
        }
    }
    Ok(manifest)
}

// Older QAIRT manifests keyed the model under "N/A" with a top-level precision.
fn migrate_legacy_qairt_precision(m: &mut ModelManifest) -> bool {
    if m.plugin_id != "qairt" || m.precision.is_empty() {
        return false;
    }
    if m.model_file.len() != 1 || !m.model_file.contains_key("N/A") {
        return false;
    }
    let entry = m.model_file.remove("N/A").unwrap();
    let key = std::mem::take(&mut m.precision);
    m.model_file.insert(key, entry);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::ModelFileInfo;
    use crate::test_support::{file_info, manifest, temp_store};

    fn sample(name: &str) -> ModelManifest {
        manifest(name, [("Q4_K_M", file_info("model-Q4_K_M.gguf", 100))])
    }

    fn stored(names: &[&str]) -> (Store, tempfile::TempDir) {
        let (store, tmp) = temp_store();
        for name in names {
            store.write_manifest(&sample(name)).unwrap();
        }
        (store, tmp)
    }

    #[test]
    fn bare_name_pick_is_the_head_of_the_reported_order() {
        let mut m = sample("Org/Repo");
        m.model_file
            .insert("IQ4_XS".into(), file_info("model-IQ4_XS.gguf", 50));
        m.model_file.insert(
            "Q8_0".into(),
            ModelFileInfo {
                downloaded: false,
                ..file_info("model-Q8_0.gguf", 200)
            },
        );
        // Unlisted IQ4_XS sorts after Q4_K_M; Q8_0 is not downloaded.
        let mut reported: Vec<&str> = m
            .model_file
            .iter()
            .filter(|(_, f)| f.downloaded)
            .map(|(q, _)| q.as_str())
            .collect();
        reported.sort_by_key(|q| quant_sort_key(q));
        assert_eq!(reported, ["Q4_K_M", "IQ4_XS"]);
        assert_eq!(pick_downloaded_by_priority(&m).unwrap(), reported[0]);
    }

    #[test]
    fn roundtrip_and_list() {
        let (store, _tmp) = stored(&["TestOrg/TestRepo", "Org/B"]);
        assert_eq!(
            store.get_manifest("TestOrg/TestRepo").unwrap().name,
            "TestOrg/TestRepo"
        );
        assert_eq!(store.list().unwrap().len(), 2);
    }

    #[test]
    fn resolve_manifest_accepts_loose_names() {
        let (store, _tmp) = stored(&["qualcomm/Bare", "ggml-org/Qwen3"]);
        for (input, want) in [
            ("Bare", "qualcomm/Bare"),
            ("Bare:Q4_K_M", "qualcomm/Bare"),
            ("https://huggingface.co/ggml-org/Qwen3", "ggml-org/Qwen3"),
        ] {
            assert_eq!(store.resolve_manifest(input).unwrap().name, want, "{input}");
        }
        assert!(store.resolve_manifest("Org/Missing").is_err());
        assert_eq!(store.get_model_type("Bare:Q4_K_M").unwrap(), ModelType::Llm);
    }

    #[test]
    fn every_operation_accepts_loose_names() {
        let (store, _tmp) = stored(&["qualcomm/Bare", "ggml-org/Qwen3"]);
        let url = "https://huggingface.co/ggml-org/Qwen3";
        for input in [url, &format!("{url}:Q4_K_M")] {
            let (quant, paths) = store.get_paths(input).unwrap();
            assert_eq!(
                (quant.as_str(), paths.model_name.as_str()),
                ("Q4_K_M", "Qwen3"),
                "{input}"
            );
        }
        store.set_model_type("Bare:Q4_K_M", ModelType::Vlm).unwrap();
        assert_eq!(
            store.get_model_type("qualcomm/Bare").unwrap(),
            ModelType::Vlm
        );
        store.remove("Bare").unwrap();
        assert!(!store.cfg.model_dir("qualcomm/Bare").exists());
    }

    #[test]
    fn inflight_models_are_hidden_from_listing_but_loadable() {
        let (store, _tmp) = stored(&["Org/Pulling", "Org/Done"]);
        fs::create_dir_all(store.cfg.model_dir("Org/Pulling").join(INFLIGHT_DIR)).unwrap();
        assert!(store.resolve_detail_manifest("Org/Pulling").is_err());
        assert!(store.resolve_manifest("Org/Pulling").is_ok());
        let listed: Vec<String> = store.list().unwrap().into_iter().map(|m| m.name).collect();
        assert_eq!(listed, ["Org/Done"]);
    }

    #[test]
    fn list_skips_corrupted() {
        let (store, _tmp) = stored(&["Org/Good"]);
        let bad_dir = store.cfg.model_dir("Org/Bad");
        fs::create_dir_all(&bad_dir).unwrap();
        fs::write(bad_dir.join(MANIFEST_FILE), "not valid json").unwrap();
        let listed: Vec<String> = store.list().unwrap().into_iter().map(|m| m.name).collect();
        assert_eq!(listed, ["Org/Good"]);
    }

    #[test]
    fn set_model_type_roundtrips_and_requires_the_model() {
        let (store, _tmp) = stored(&["Org/Typed"]);
        assert_eq!(store.get_model_type("Org/Typed").unwrap(), ModelType::Llm);
        store.set_model_type("Org/Typed", ModelType::Vlm).unwrap();
        assert_eq!(store.get_model_type("Org/Typed").unwrap(), ModelType::Vlm);
        let err = store
            .set_model_type("Org/Absent", ModelType::Vlm)
            .unwrap_err();
        assert!(matches!(err, Error::ModelNotFound(_)), "{err:?}");
    }

    #[test]
    fn remove_whole_model_or_its_last_quant_deletes_the_dir() {
        // A colon must never reach a Windows path (os error 267).
        let (store, _tmp) = stored(&["Org/C", "Org/WithQuant"]);
        for (name, arg) in [
            ("Org/C", "Org/C"),
            ("Org/WithQuant", "Org/WithQuant:Q4_K_M"),
        ] {
            store.remove(arg).unwrap();
            assert!(!store.cfg.model_dir(name).exists(), "{arg}");
        }
    }

    #[test]
    fn remove_quant_keeps_other_quants_and_shared_files() {
        let (store, _tmp) = temp_store();
        let mut m = sample("Org/Multi");
        m.model_file
            .insert("Q8_0".into(), file_info("model-Q8_0.gguf", 200));
        m.mmproj_file = file_info("mmproj.gguf", 50);
        store.write_manifest(&m).unwrap();
        let dir = store.cfg.model_dir("Org/Multi");
        for f in ["model-Q4_K_M.gguf", "model-Q8_0.gguf", "mmproj.gguf"] {
            fs::write(dir.join(f), b"x").unwrap();
        }

        let err = store.remove("Org/Multi:Q2_K").unwrap_err();
        assert!(matches!(err, Error::QuantNotFound(..)), "{err:?}");
        assert!(dir.join(MANIFEST_FILE).exists());

        store.remove("Org/Multi:Q4_K_M").unwrap();
        assert!(!dir.join("model-Q4_K_M.gguf").exists());
        assert!(dir.join("model-Q8_0.gguf").exists());
        assert!(dir.join("mmproj.gguf").exists());
        let quants: Vec<String> = store
            .get_manifest("Org/Multi")
            .unwrap()
            .model_file
            .into_keys()
            .collect();
        assert_eq!(quants, ["Q8_0"]);
    }

    #[test]
    fn clean_returns_count() {
        let (store, _tmp) = stored(&["Org/D", "Org/E"]);
        assert_eq!(store.clean().unwrap(), 2);
    }

    #[test]
    fn rejects_invalid_names() {
        let (store, _tmp) = temp_store();
        assert!(store.get_manifest("../etc").is_err());
        assert!(store.remove("a/..").is_err());
        assert!(store.model_dir("../etc").is_err());
    }

    #[test]
    fn rejects_oversized_manifest() {
        let (store, _tmp) = temp_store();
        let dir = store.cfg.model_dir("Org/Huge");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join(MANIFEST_FILE),
            "x".repeat(MANIFEST_MAX_BYTES as usize + 1),
        )
        .unwrap();
        assert!(store.get_manifest("Org/Huge").is_err());
    }

    #[test]
    fn legacy_qairt_manifest_migrates_on_read_and_rewrites_disk() {
        let (store, _tmp) = temp_store();
        let dir = store.cfg.model_dir("qualcomm/Legacy");
        fs::create_dir_all(&dir).unwrap();
        let legacy = r#"{
            "Name":"qualcomm/Legacy","ModelName":"legacy","ModelType":"llm",
            "PluginId":"qairt","Precision":"W4A16",
            "ModelFile":{"N/A":{"Name":"model-00.bin","Downloaded":true,"Size":10}}
        }"#;
        fs::write(dir.join(MANIFEST_FILE), legacy).unwrap();

        let loaded = store.get_manifest("qualcomm/Legacy").unwrap();
        assert_eq!(loaded.precision, "");
        assert_eq!(loaded.model_file.keys().collect::<Vec<_>>(), ["W4A16"]);

        let on_disk = fs::read_to_string(dir.join(MANIFEST_FILE)).unwrap();
        assert!(on_disk.contains(r#""W4A16":"#), "{on_disk}");
        for gone in [r#""N/A""#, r#""Precision""#] {
            assert!(!on_disk.contains(gone), "{gone} in {on_disk}");
        }
    }
}
