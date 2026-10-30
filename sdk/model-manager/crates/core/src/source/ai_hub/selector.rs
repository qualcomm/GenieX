// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Pick one asset out of `release-assets.json` for a given chipset.
//!
//! Matches the Go CLI's selection logic in
//! `cli/internal/model_hub/aihub/selector.go`:
//!
//! - Resolve the user-supplied chipset against `platform.json` aliases.
//! - Filter to assets whose `runtime` is one of [`QAIRT_RUNTIMES`]. The
//!   SDK only runs QAIRT (NPU) archives through this source; other assets
//!   (ONNX, TFLite, QNN_DLC, llama.cpp GGUF) are rejected up front.
//! - Filter to assets whose canonical chipset matches.
//! - If multiple assets survive, prefer the newer `RUNTIME_GENIEX_QAIRT`
//!   entry, then pick the lex-first precision string for a deterministic
//!   default. Best-effort tiebreaker; precision is not exposed over FFI.

use super::dto::{AssetDetails, ModelReleaseAssets, PlatformInfo};
use crate::error::{Error, Result};

/// Runtime strings the public bucket uses for QAIRT (NPU) assets, in
/// preference order.
///
/// AI Hub is mid-migration from the legacy `RUNTIME_GENIE` enum to the
/// namespaced `RUNTIME_GENIEX_QAIRT`. Most models publish both, but neither
/// alone is sufficient (bucket v0.60.0): `Gemma-4-E4B-it` ships only the
/// new name, `Phi-3.5-Mini-Instruct` only the legacy one. Chipset coverage
/// can also differ between the two for the same model, so both are kept as
/// match candidates rather than picking one per model.
const QAIRT_RUNTIMES: &[&str] = &["RUNTIME_GENIEX_QAIRT", "RUNTIME_GENIE"];

/// Rank of `runtime` within [`QAIRT_RUNTIMES`] for the preference sort;
/// non-QAIRT runtimes sort last (they're filtered out before this runs).
fn runtime_rank(runtime: &str) -> usize {
    QAIRT_RUNTIMES
        .iter()
        .position(|r| *r == runtime)
        .unwrap_or(QAIRT_RUNTIMES.len())
}

/// Whether `runtime` names a QAIRT archive this source can install.
pub fn is_qairt_runtime(runtime: &str) -> bool {
    QAIRT_RUNTIMES.contains(&runtime)
}

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

/// Resolve a chipset alias (case-insensitive, trimmed) to the reference
/// device name AI Hub displays for it, e.g. `"qualcomm-snapdragon-x-elite"`
/// or `"sm8650"` → `"Snapdragon X Elite CRD"`. Falls back to the canonical
/// id when the bucket omits a reference device. Returns `None` when the
/// chipset is not in `platform.json`.
pub fn resolve_chipset_display(plat: &PlatformInfo, chipset: &str) -> Option<String> {
    let target = chipset.trim().to_ascii_lowercase();
    if target.is_empty() {
        return None;
    }
    for cs in &plat.chipsets {
        let hit = cs.name.to_ascii_lowercase() == target
            || cs.reference_device.to_ascii_lowercase() == target
            || cs.aliases.iter().any(|a| a.to_ascii_lowercase() == target);
        if hit {
            if cs.reference_device.is_empty() {
                return Some(cs.name.clone());
            }
            return Some(cs.reference_device.clone());
        }
    }
    None
}

/// Pick one asset matching `(chipset, QAIRT runtime)`. Returns the list of
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
        .filter(|a| {
            is_qairt_runtime(&a.runtime) && a.chipset.as_deref() == Some(canonical.as_str())
        })
        .collect();

    if candidates.is_empty() {
        return Err(UnavailableChipset {
            requested: chipset.to_string(),
            available: collect_chipsets(ra, plat),
        });
    }

    // Prefer the newer runtime, then tiebreak on precision so the pick is
    // deterministic when a model publishes the same chipset under both.
    candidates.sort_by(|a, b| {
        runtime_rank(&a.runtime)
            .cmp(&runtime_rank(&b.runtime))
            .then_with(|| a.precision.cmp(&b.precision))
    });
    Ok(candidates[0])
}

/// Collect the chipsets a model ships *installable* (QAIRT) assets for,
/// rendered as the reference device names callers see in
/// `list_supported_chipsets` (so error messages match the picker). Falls
/// back to the canonical id when `platform.json` has no reference device
/// for it.
///
/// Only accepted runtimes are counted: listing every chipset in
/// `release-assets.json` regardless of runtime produced self-contradictory
/// errors that named the very chipset the caller had just requested.
fn collect_chipsets(ra: &ModelReleaseAssets, plat: &PlatformInfo) -> Vec<String> {
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for a in ra.assets.iter().filter(|a| is_qairt_runtime(&a.runtime)) {
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
    use crate::source::ai_hub::dto::ChipsetInfo;

    const QAIRT: &str = "RUNTIME_GENIEX_QAIRT";
    const GENIE: &str = "RUNTIME_GENIE";
    const X_ELITE: &str = "qualcomm-snapdragon-x-elite";
    const CRD: &str = "Snapdragon X Elite CRD";

    /// `(name, reference_device, aliases)` per chipset.
    fn platform(entries: &[(&str, &str, &[&str])]) -> PlatformInfo {
        PlatformInfo {
            chipsets: entries
                .iter()
                .map(|(name, reference, aliases)| ChipsetInfo {
                    name: name.to_string(),
                    reference_device: reference.to_string(),
                    aliases: aliases.iter().map(|a| a.to_string()).collect(),
                })
                .collect(),
            devices: Vec::new(),
        }
    }

    /// `(chipset, runtime, precision)`.
    type Asset<'a> = (&'a str, &'a str, &'a str);

    fn assets(entries: &[Asset]) -> ModelReleaseAssets {
        ModelReleaseAssets {
            model_id: "m".into(),
            assets: entries
                .iter()
                .map(|(chipset, runtime, precision)| AssetDetails {
                    chipset: Some(chipset.to_string()),
                    runtime: runtime.to_string(),
                    precision: precision.to_string(),
                    download_url: format!("https://example.invalid/{chipset}-{precision}.zip"),
                    uncompressed_size: Some(1),
                })
                .collect(),
        }
    }

    #[test]
    fn resolve_chipset_matches_aliases_and_reference_device() {
        let plat = platform(&[
            ("Snapdragon 8 Gen 3", "", &["sm8650", "SD8G3"]),
            (X_ELITE, CRD, &[]),
        ]);
        for (input, want) in [
            ("sm8650", "Snapdragon 8 Gen 3"),
            ("SD8G3", "Snapdragon 8 Gen 3"),
            ("snapdragon x elite crd", X_ELITE),
        ] {
            assert_eq!(resolve_chipset(&plat, input).unwrap(), want, "{input}");
        }
        assert!(resolve_chipset(&plat, "unknown").is_err());
    }

    #[test]
    fn resolve_chipset_display_prefers_reference_device() {
        let plat = platform(&[
            (X_ELITE, CRD, &["x1e80100"]),
            ("qualcomm-qcs9075", "", &["sa8775p"]),
        ]);
        for (input, want) in [
            (X_ELITE, Some(CRD)),
            ("X1E80100", Some(CRD)),
            ("snapdragon x elite crd", Some(CRD)),
            ("sa8775p", Some("qualcomm-qcs9075")),
            ("unknown", None),
            ("", None),
        ] {
            assert_eq!(
                resolve_chipset_display(&plat, input).as_deref(),
                want,
                "{input:?}"
            );
        }
    }

    #[test]
    fn match_asset_picks() {
        // Gemma-4-E4B-it publishes only GENIEX_QAIRT, Phi-3.5-Mini-Instruct only legacy GENIE;
        // a dual-published chipset prefers GENIEX_QAIRT though GENIE sorts first.
        let cases: [(&[Asset], &str, Asset); 5] = [
            (
                &[
                    ("SD8G3", "RUNTIME_TFLITE", "PRECISION_FP16"),
                    ("SD8G3", QAIRT, "PRECISION_W4A16"),
                    ("Other", QAIRT, "PRECISION_FP16"),
                ],
                "sm8650",
                ("SD8G3", QAIRT, "PRECISION_W4A16"),
            ),
            (
                &[(X_ELITE, QAIRT, "PRECISION_W4A16")],
                "x1e80100",
                (X_ELITE, QAIRT, "PRECISION_W4A16"),
            ),
            (
                &[(X_ELITE, GENIE, "PRECISION_W4A16")],
                "x1e80100",
                (X_ELITE, GENIE, "PRECISION_W4A16"),
            ),
            (
                &[
                    ("A", GENIE, "PRECISION_W4A16"),
                    ("A", QAIRT, "PRECISION_W4A16"),
                ],
                "A",
                ("A", QAIRT, "PRECISION_W4A16"),
            ),
            (
                &[
                    ("A", QAIRT, "PRECISION_W4A16"),
                    ("A", QAIRT, "PRECISION_FP16"),
                ],
                "A",
                ("A", QAIRT, "PRECISION_FP16"),
            ),
        ];
        let plat = platform(&[
            ("SD8G3", "", &["sm8650"]),
            (X_ELITE, "", &["x1e80100"]),
            ("A", "", &[]),
        ]);
        for (published, chipset, (want_chip, want_rt, want_prec)) in cases {
            let ra = assets(published);
            let got = match_asset(&ra, &plat, chipset).unwrap();
            assert_eq!(
                (
                    got.chipset.as_deref(),
                    got.runtime.as_str(),
                    got.precision.as_str()
                ),
                (Some(want_chip), want_rt, want_prec),
                "{published:?}"
            );
        }
    }

    #[test]
    fn match_asset_reports_installable_chipsets_on_miss() {
        // Non-QAIRT runtimes must not appear, or the hint names an uninstallable chipset.
        let cases: [(PlatformInfo, &[Asset], &[&str]); 3] = [
            (
                platform(&[("A", "", &[]), ("B", "", &[])]),
                &[
                    ("A", QAIRT, "PRECISION_W4A16"),
                    ("B", "RUNTIME_TFLITE", "PRECISION_FP16"),
                ],
                &["A"],
            ),
            (
                platform(&[("A", "", &[]), ("B", "", &[])]),
                &[
                    ("A", QAIRT, "PRECISION_FP16"),
                    ("B", GENIE, "PRECISION_FP16"),
                ],
                &["A", "B"],
            ),
            (
                platform(&[(X_ELITE, CRD, &[])]),
                &[(X_ELITE, QAIRT, "PRECISION_FP16")],
                &[CRD],
            ),
        ];
        for (plat, published, want) in cases {
            let err = match_asset(&assets(published), &plat, "unknown").unwrap_err();
            assert_eq!(err.available, want, "{published:?}");
        }
    }
}
