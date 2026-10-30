// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Pick one QAIRT asset out of `release-assets.json` for a given chipset.

use super::dto::{AssetDetails, ChipsetInfo, ModelReleaseAssets, PlatformInfo};
use crate::error::{Error, Result};

/// Preference order. AI Hub is migrating `RUNTIME_GENIE` to `RUNTIME_GENIEX_QAIRT`;
/// some models ship only one, and chipset coverage can differ, so both match.
const QAIRT_RUNTIMES: &[&str] = &["RUNTIME_GENIEX_QAIRT", "RUNTIME_GENIE"];

fn runtime_rank(runtime: &str) -> usize {
    QAIRT_RUNTIMES
        .iter()
        .position(|r| *r == runtime)
        .unwrap_or(QAIRT_RUNTIMES.len())
}

pub fn is_qairt_runtime(runtime: &str) -> bool {
    QAIRT_RUNTIMES.contains(&runtime)
}

const SUPPORTED_DOMAINS: &[&str] = &["MODEL_DOMAIN_GENERATIVE_AI", "MODEL_DOMAIN_MULTIMODAL"];

pub fn is_domain_supported(domain: &str) -> bool {
    SUPPORTED_DOMAINS.contains(&domain)
}

fn find_chipset<'a>(plat: &'a PlatformInfo, chipset: &str) -> Option<&'a ChipsetInfo> {
    let target = chipset.trim();
    if target.is_empty() {
        return None;
    }
    plat.chipsets.iter().find(|cs| {
        std::iter::once(&cs.name)
            .chain([&cs.reference_device])
            .chain(&cs.aliases)
            .any(|n| n.eq_ignore_ascii_case(target))
    })
}

pub fn resolve_chipset(plat: &PlatformInfo, chipset: &str) -> Result<String> {
    find_chipset(plat, chipset)
        .map(|cs| cs.name.clone())
        .ok_or_else(|| Error::Hub(format!("chipset {chipset:?} not found in platform.json")))
}

pub fn resolve_chipset_display(plat: &PlatformInfo, chipset: &str) -> Option<String> {
    find_chipset(plat, chipset).map(|cs| cs.display_name().to_string())
}

pub fn match_asset<'a>(
    ra: &'a ModelReleaseAssets,
    plat: &PlatformInfo,
    chipset: &str,
) -> Result<&'a AssetDetails> {
    let canonical = find_chipset(plat, chipset).map(|cs| cs.name.as_str());
    let mut candidates: Vec<&AssetDetails> = ra
        .assets
        .iter()
        .filter(|a| {
            is_qairt_runtime(&a.runtime) && canonical.is_some() && a.chipset.as_deref() == canonical
        })
        .collect();
    candidates.sort_by(|a, b| {
        runtime_rank(&a.runtime)
            .cmp(&runtime_rank(&b.runtime))
            .then_with(|| a.precision.cmp(&b.precision))
    });
    candidates
        .first()
        .copied()
        .ok_or_else(|| Error::ChipsetUnavailable {
            requested: chipset.to_string(),
            available: installable_chipsets(ra, plat),
        })
}

fn installable_chipsets(ra: &ModelReleaseAssets, plat: &PlatformInfo) -> Vec<String> {
    let names: std::collections::BTreeSet<String> = ra
        .assets
        .iter()
        .filter(|a| is_qairt_runtime(&a.runtime))
        .filter_map(|a| a.chipset.as_deref())
        .map(|cs| {
            plat.chipsets
                .iter()
                .find(|c| c.name == cs)
                .map_or(cs, |c| c.display_name())
                .to_string()
        })
        .collect();
    names.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const QAIRT: &str = "RUNTIME_GENIEX_QAIRT";
    const GENIE: &str = "RUNTIME_GENIE";
    const X_ELITE: &str = "qualcomm-snapdragon-x-elite";
    const CRD: &str = "Snapdragon X Elite CRD";

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
        // A dual-published chipset prefers GENIEX_QAIRT though GENIE sorts first.
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
            let Error::ChipsetUnavailable { available, .. } = err else {
                panic!("{err:?}");
            };
            assert_eq!(available, want, "{published:?}");
        }
    }
}
