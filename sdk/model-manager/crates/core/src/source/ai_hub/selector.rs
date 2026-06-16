//! Pick one asset out of `release-assets.json` for a given chipset.
//!
//! Matches the Go CLI's selection logic in
//! `cli/internal/model_hub/aihub/selector.go`:
//!
//! - Resolve the user-supplied chipset against `platform.json` aliases.
//! - Filter to assets whose `runtime == RUNTIME_GENIE`. The CLI only
//!   supports the Genie runtime (LLM / VLM domains); other assets are
//!   rejected up front.
//! - Filter to assets whose canonical chipset matches.
//! - If multiple precisions survive, pick the lex-first precision string
//!   for a deterministic default. The Go CLI sorts by enum value, which
//!   produces the same order for the common precisions we see in practice
//!   (`PRECISION_FP16` before `PRECISION_W4A16`, etc.). This is a
//!   best-effort tiebreaker; precision selection is not exposed over FFI.

use super::manifest::{AssetDetails, ModelReleaseAssets, PlatformInfo};
use crate::error::{Error, Result};

/// Runtime string the public bucket uses for Genie-compatible assets.
/// Matches `qaihm.Runtime_RUNTIME_GENIE`.
const RUNTIME_GENIE: &str = "RUNTIME_GENIE";

/// Domains the SDK currently supports — same subset the Go CLI accepts
/// in `RuntimeForDomain`.
const SUPPORTED_DOMAINS: &[&str] = &["MODEL_DOMAIN_GENERATIVE_AI", "MODEL_DOMAIN_MULTIMODAL"];

pub fn is_domain_supported(domain: &str) -> bool {
    SUPPORTED_DOMAINS.contains(&domain)
}

/// Match a chipset alias (case-insensitive, trimmed) to its canonical name
/// from `platform.json`. Returns an error if the chipset is not in the list.
pub fn resolve_chipset(plat: &PlatformInfo, chipset: &str) -> Result<String> {
    let target = chipset.trim().to_ascii_lowercase();
    if target.is_empty() {
        return Err(Error::Hub("empty chipset".to_string()));
    }
    for cs in &plat.chipsets {
        if cs.name.to_ascii_lowercase() == target
            || cs.reference_device.to_ascii_lowercase() == target
        {
            return Ok(cs.name.clone());
        }
        for a in &cs.aliases {
            if a.to_ascii_lowercase() == target {
                return Ok(cs.name.clone());
            }
        }
    }
    Err(Error::Hub(format!(
        "chipset {chipset:?} not found in platform.json"
    )))
}

/// Pick one asset matching `(chipset, RUNTIME_GENIE)`. Returns the list of
/// supported chipsets (for actionable error messages) when nothing matches.
pub fn match_asset<'a>(
    ra: &'a ModelReleaseAssets,
    plat: &PlatformInfo,
    chipset: &str,
) -> std::result::Result<&'a AssetDetails, UnavailableChipset> {
    if ra.assets.is_empty() {
        return Err(UnavailableChipset {
            requested: chipset.to_string(),
            available: Vec::new(),
        });
    }

    let canonical = match resolve_chipset(plat, chipset) {
        Ok(c) => c,
        Err(_) => {
            return Err(UnavailableChipset {
                requested: chipset.to_string(),
                available: collect_chipsets(ra, plat),
            });
        }
    };

    let mut candidates: Vec<&AssetDetails> = ra
        .assets
        .iter()
        .filter(|a| a.runtime == RUNTIME_GENIE && a.chipset.as_deref() == Some(canonical.as_str()))
        .collect();

    if candidates.is_empty() {
        return Err(UnavailableChipset {
            requested: chipset.to_string(),
            available: collect_chipsets(ra, plat),
        });
    }

    // Deterministic tiebreak when multiple precisions match.
    candidates.sort_by(|a, b| a.precision.cmp(&b.precision));
    Ok(candidates[0])
}

/// Collect the chipsets a model ships assets for, rendered as the
/// reference device names callers see in `list_supported_chipsets` (so
/// error messages match the picker). Falls back to the canonical id when
/// `platform.json` has no reference device for it.
fn collect_chipsets(ra: &ModelReleaseAssets, plat: &PlatformInfo) -> Vec<String> {
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for a in &ra.assets {
        if let Some(cs) = a.chipset.as_deref() {
            let display = plat
                .chipsets
                .iter()
                .find(|c| c.name == cs && !c.reference_device.is_empty())
                .map(|c| c.reference_device.clone())
                .unwrap_or_else(|| cs.to_string());
            seen.insert(display);
        }
    }
    seen.into_iter().collect()
}

#[derive(Debug)]
pub struct UnavailableChipset {
    pub requested: String,
    pub available: Vec<String>,
}

impl std::fmt::Display for UnavailableChipset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "device {:?} not available; model supports: {}",
            self.requested,
            self.available.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::ai_hub::manifest::ChipsetInfo;

    fn platform(entries: &[(&str, &[&str])]) -> PlatformInfo {
        PlatformInfo {
            chipsets: entries
                .iter()
                .map(|(name, aliases)| ChipsetInfo {
                    name: (*name).to_string(),
                    reference_device: String::new(),
                    aliases: aliases.iter().map(|a| (*a).to_string()).collect(),
                })
                .collect(),
        }
    }

    #[test]
    fn resolves_reference_device() {
        let plat = PlatformInfo {
            chipsets: vec![ChipsetInfo {
                name: "qualcomm-snapdragon-x-elite".to_string(),
                reference_device: "Snapdragon X Elite CRD".to_string(),
                aliases: vec![],
            }],
        };
        assert_eq!(
            resolve_chipset(&plat, "snapdragon x elite crd").unwrap(),
            "qualcomm-snapdragon-x-elite"
        );
    }

    fn asset(chipset: &str, runtime: &str, precision: &str) -> AssetDetails {
        AssetDetails {
            chipset: Some(chipset.to_string()),
            runtime: runtime.to_string(),
            precision: precision.to_string(),
            download_url: format!("https://example.invalid/{chipset}-{precision}.zip"),
            uncompressed_size: Some(1),
        }
    }

    #[test]
    fn resolves_alias_case_insensitive() {
        let plat = platform(&[("Snapdragon 8 Gen 3", &["sm8650", "SD8G3"])]);
        assert_eq!(
            resolve_chipset(&plat, "sm8650").unwrap(),
            "Snapdragon 8 Gen 3"
        );
        assert_eq!(
            resolve_chipset(&plat, "SD8G3").unwrap(),
            "Snapdragon 8 Gen 3"
        );
        assert!(resolve_chipset(&plat, "unknown").is_err());
    }

    #[test]
    fn picks_matching_runtime_and_chipset() {
        let plat = platform(&[("SD8G3", &["sm8650"])]);
        let ra = ModelReleaseAssets {
            model_id: "m".into(),
            assets: vec![
                asset("SD8G3", "RUNTIME_TFLITE", "PRECISION_FP16"),
                asset("SD8G3", "RUNTIME_GENIE", "PRECISION_W4A16"),
                asset("Other", "RUNTIME_GENIE", "PRECISION_FP16"),
            ],
        };
        let got = match_asset(&ra, &plat, "sm8650").unwrap();
        assert_eq!(got.chipset.as_deref(), Some("SD8G3"));
        assert_eq!(got.runtime, "RUNTIME_GENIE");
    }

    #[test]
    fn reports_available_when_chipset_missing() {
        let plat = platform(&[("A", &[]), ("B", &[])]);
        let ra = ModelReleaseAssets {
            model_id: "m".into(),
            assets: vec![
                asset("A", "RUNTIME_GENIE", "PRECISION_FP16"),
                asset("B", "RUNTIME_GENIE", "PRECISION_FP16"),
            ],
        };
        let err = match_asset(&ra, &plat, "C").unwrap_err();
        assert_eq!(err.available, vec!["A".to_string(), "B".to_string()]);
    }

    #[test]
    fn available_uses_reference_device_names() {
        let plat = PlatformInfo {
            chipsets: vec![ChipsetInfo {
                name: "qualcomm-snapdragon-x-elite".to_string(),
                reference_device: "Snapdragon X Elite CRD".to_string(),
                aliases: vec![],
            }],
        };
        let ra = ModelReleaseAssets {
            model_id: "m".into(),
            assets: vec![asset(
                "qualcomm-snapdragon-x-elite",
                "RUNTIME_GENIE",
                "PRECISION_FP16",
            )],
        };
        let err = match_asset(&ra, &plat, "unknown").unwrap_err();
        assert_eq!(err.available, vec!["Snapdragon X Elite CRD".to_string()]);
    }

    #[test]
    fn tiebreaks_multiple_precisions_deterministically() {
        let plat = platform(&[("A", &[])]);
        let ra = ModelReleaseAssets {
            model_id: "m".into(),
            assets: vec![
                asset("A", "RUNTIME_GENIE", "PRECISION_W4A16"),
                asset("A", "RUNTIME_GENIE", "PRECISION_FP16"),
            ],
        };
        let picked = match_asset(&ra, &plat, "A").unwrap();
        assert_eq!(picked.precision, "PRECISION_FP16");
    }
}
