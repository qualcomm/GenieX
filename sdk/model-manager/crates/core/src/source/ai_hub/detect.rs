//! Host chipset auto-detection for AI Hub pulls.
//!
//! AI Hub packages qairt assets per (chipset, HTP arch) pair — for
//! example `snapdragon-x-elite` and `snapdragon-x2-elite` are two
//! different zips even though both are Qualcomm Compute laptops. Asking
//! every SDK caller to hard-code a chipset string is painful, so this
//! module takes a best-effort guess from the host.
//!
//! Current coverage:
//!   * Windows on Snapdragon (X Elite / X Plus / X2 Elite) — parsed
//!     from the CPU brand string via a `reg query` probe. This is the
//!     95% case for Genie runtime users today.
//!   * Linux on Qualcomm Dragonwing boards (QCS6490 / QCS9075) —
//!     parsed from `/sys/firmware/devicetree/base/compatible`.
//!   * Android on Snapdragon — parsed from the `ro.soc.model`
//!     system property (e.g. `SM8750` → `qualcomm-snapdragon-8-elite`
//!     via AI Hub's alias table).
//!   * Everything else — returns `None`. Callers must then pass
//!     `chipset` explicitly.

/// Probe the host for a chipset identifier that can be fed to
/// `resolve_chipset`. Returns `None` when detection is not supported
/// on this platform or when the probe fails.
pub fn detect_host_chipset() -> Option<String> {
    #[cfg(target_os = "windows")]
    {
        windows::detect_cpu_brand().and_then(cpu_name_to_chipset_alias)
    }
    #[cfg(target_os = "linux")]
    {
        linux::detect_dt_chipset()
    }
    #[cfg(target_os = "android")]
    {
        android::detect_soc_chipset()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "android")))]
    {
        None
    }
}

/// Map a Qualcomm Oryon SKU number (extracted from the CPU brand
/// string) to the matching AI Hub chipset alias. Covers the SKUs
/// published as of QAIRT 2.45; unknown SKUs return `None` so we
/// fall back to the caller-supplied chipset.
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
pub(crate) fn cpu_name_to_chipset_alias(brand: String) -> Option<String> {
    let sku = extract_oryon_sku(&brand)?;
    match &sku[..3] {
        "X1E" => Some("qualcomm-snapdragon-x-elite".to_string()),
        "X1P" => Some("qualcomm-snapdragon-x-plus-8-core".to_string()),
        "X2E" => Some("qualcomm-snapdragon-x2-elite".to_string()),
        _ => None,
    }
}

#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
fn extract_oryon_sku(brand: &str) -> Option<String> {
    for tok in brand.split(|c: char| !(c.is_ascii_alphanumeric())) {
        if is_oryon_sku(tok) {
            return Some(tok.to_ascii_uppercase());
        }
    }
    None
}

#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
fn is_oryon_sku(tok: &str) -> bool {
    let bytes = tok.as_bytes();
    if bytes.len() < 6 {
        return false;
    }
    if !bytes[0].eq_ignore_ascii_case(&b'X') {
        return false;
    }
    if !bytes[1].is_ascii_digit() {
        return false;
    }
    let c = bytes[2].to_ascii_uppercase();
    if c != b'E' && c != b'P' {
        return false;
    }
    bytes[3..].iter().all(|b| b.is_ascii_digit())
}

#[cfg(target_os = "windows")]
mod windows {
    //! We read the CPU brand string from
    //! `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0\ProcessorNameString`
    //! via `reg query`; it's been stable since Windows 2000 and
    //! doesn't depend on `wmic.exe`, which was deprecated and is
    //! missing on recent Windows 11 builds (24H2+).

    use std::process::Command;

    const KEY: &str = r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0";
    const VALUE: &str = "ProcessorNameString";

    pub(super) fn detect_cpu_brand() -> Option<String> {
        let out = Command::new("reg")
            .args(["query", KEY, "/v", VALUE])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        parse_reg_query(&String::from_utf8_lossy(&out.stdout))
    }

    fn parse_reg_query(stdout: &str) -> Option<String> {
        for line in stdout.lines() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with(VALUE) {
                continue;
            }
            if let Some((_, value)) = trimmed.split_once("REG_SZ") {
                let v = value.trim();
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
        None
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parses_xelite1_reg_output() {
            let stdout = "\
HKEY_LOCAL_MACHINE\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0\r\n    \
ProcessorNameString    REG_SZ    Snapdragon(R) X 12-core X1E80100 @ 3.40 GHz\r\n\r\n";
            let brand = parse_reg_query(stdout).expect("parse");
            assert_eq!(brand, "Snapdragon(R) X 12-core X1E80100 @ 3.40 GHz");
        }

        #[test]
        fn returns_none_on_empty_output() {
            assert!(parse_reg_query("").is_none());
        }
    }
}

#[cfg(target_os = "linux")]
mod linux {
    //! OpenEmbedded-based Qualcomm Linux images always expose the boot
    //! Device Tree at `/sys/firmware/devicetree/base/compatible` — a
    //! NUL-separated list of `vendor,model` strings ordered from most
    //! specific (board) to most generic (SoC family). On QCS9075 EVK
    //! the three entries are:
    //!
    //! ```text
    //! qcom,qcs9075-addons-iq-9075-evk\0qcom,qcs9075\0qcom,sa8775p\0
    //! ```
    //!
    //! We walk that list in order and map the first entry whose SoC
    //! suffix is in our explicit table to an AI Hub chipset id. Board
    //! rows like `qcs9075-addons-iq-9075-evk` and SoC-family rows like
    //! `sa8775p` are not in the table and fall through, so a hit is
    //! only possible on a SoC we have AI Hub assets for.
    //!
    //! `/proc/cpuinfo` is deliberately not consulted — the aarch64
    //! kernel on these images exposes only ARM-standard fields
    //! (`CPU part: 0xd4b`) with no Qualcomm SoC branding.

    use std::fs;

    const DT_COMPATIBLE: &str = "/sys/firmware/devicetree/base/compatible";

    // Map a Device Tree `qcom,<soc>` suffix to the AI Hub chipset id
    // that owns the qairt assets for it. Only SoCs currently present
    // in AI Hub's `platform.json` are listed; unknown SoCs return
    // `None` so callers keep the existing "pass --chipset explicitly"
    // error path.
    const SOC_TO_CHIPSET: &[(&str, &str)] = &[
        ("qcs6490", "qualcomm-qcs6490"),
        ("qcs9075", "qualcomm-qcs9075"),
    ];

    pub(super) fn detect_dt_chipset() -> Option<String> {
        let bytes = fs::read(DT_COMPATIBLE).ok()?;
        map_compatible(&bytes).map(|s| s.to_string())
    }

    fn map_compatible(bytes: &[u8]) -> Option<&'static str> {
        for entry in bytes.split(|b| *b == 0) {
            let Ok(s) = std::str::from_utf8(entry) else {
                continue;
            };
            let Some(soc) = s.strip_prefix("qcom,") else {
                continue;
            };
            if let Some((_, chipset)) = SOC_TO_CHIPSET.iter().find(|(k, _)| *k == soc) {
                return Some(*chipset);
            }
        }
        None
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn maps_qcs9075_evk_real_fixture() {
            // Captured from `od -c /sys/firmware/devicetree/base/compatible`
            // on Qualcomm Linux 1.7 running on a QCS9075 IQ-9075 EVK.
            let bytes = b"qcom,qcs9075-addons-iq-9075-evk\0qcom,qcs9075\0qcom,sa8775p\0";
            assert_eq!(map_compatible(bytes), Some("qualcomm-qcs9075"));
        }

        #[test]
        fn maps_qcs6490_rb3_gen2() {
            let bytes = b"qcom,qcs6490-rb3gen2-vision-kit\0qcom,qcs6490\0";
            assert_eq!(map_compatible(bytes), Some("qualcomm-qcs6490"));
        }

        #[test]
        fn ignores_family_only_compatible() {
            // A Qualcomm board whose SoC isn't in AI Hub — only the
            // family row is present. Must fall through.
            assert_eq!(map_compatible(b"qcom,sa8775p\0"), None);
        }

        #[test]
        fn ignores_non_qualcomm_vendor() {
            assert_eq!(map_compatible(b"brcm,bcm2837\0"), None);
        }

        #[test]
        fn ignores_empty_input() {
            assert_eq!(map_compatible(b""), None);
        }

        #[test]
        fn does_not_match_prefix_collision() {
            // `qcs9075foo` is not `qcs9075`; strip_prefix + table
            // lookup must be exact.
            assert_eq!(map_compatible(b"qcom,qcs9075foo\0"), None);
        }

        #[test]
        fn trailing_nul_missing_is_tolerated() {
            // Final entry without trailing NUL still parses.
            assert_eq!(map_compatible(b"qcom,qcs9075"), Some("qualcomm-qcs9075"));
        }
    }
}

mod android {
    //! Android exposes the SoC identifier through the property system,
    //! not through the boot Device Tree (`/sys/firmware/devicetree/base`
    //! exists but is not readable from app-context SELinux domains on
    //! most shipping devices). We read `ro.soc.model` — the property
    //! backing Java `Build.SOC_MODEL` since Android 12 — and, as a
    //! vendor guard, `ro.soc.manufacturer` (backing `Build.SOC_MANUFACTURER`).
    //!
    //! The SoC model is returned to the caller as-is (e.g. `"SM8750"`).
    //! AI Hub's `resolve_chipset` already treats the lowercased SoC
    //! code as an alias — `sm8750` → `qualcomm-snapdragon-8-elite`,
    //! `sm8650` → `qualcomm-snapdragon-8gen3`, etc. — so we deliberately
    //! do not maintain a second mapping table here.
    //!
    //! Known limitation: `ro.soc.model` does not expose variant
    //! suffixes. A Galaxy S25 reports `SM8750` (resolves to
    //! `qualcomm-snapdragon-8-elite`), not `SM8750-AC` (the alias for
    //! `qualcomm-snapdragon-8-elite-for-galaxy`). Users needing the
    //! variant asset must pass `--chipset` explicitly.
    //!
    //! We accept both `"QTI"` (the actual value on recent Samsung
    //! builds) and `"Qualcomm"` (the value documented in AOSP) as the
    //! manufacturer — anything else falls through to `None`.

    // Pure mapping logic — unit-tested on host. Split from the
    // property probe below so the AI-Hub-facing string contract has
    // coverage without needing an Android target.
    #[cfg_attr(not(target_os = "android"), allow(dead_code))]
    pub(super) fn map_soc<'a>(manuf: &str, model: &'a str) -> Option<&'a str> {
        let ok = manuf.eq_ignore_ascii_case("QTI") || manuf.eq_ignore_ascii_case("Qualcomm");
        if !ok {
            return None;
        }
        let model = model.trim();
        if model.is_empty() {
            return None;
        }
        Some(model)
    }

    #[cfg(target_os = "android")]
    mod probe {
        use super::map_soc;
        use std::ffi::CString;
        use std::os::raw::{c_char, c_int};

        const PROP_VALUE_MAX: usize = 92;

        extern "C" {
            fn __system_property_get(name: *const c_char, value: *mut c_char) -> c_int;
        }

        pub(in crate::source::ai_hub::detect) fn detect_soc_chipset() -> Option<String> {
            let manuf = read_prop("ro.soc.manufacturer")?;
            let model = read_prop("ro.soc.model")?;
            map_soc(&manuf, &model).map(str::to_string)
        }

        fn read_prop(name: &str) -> Option<String> {
            let c_name = CString::new(name).ok()?;
            let mut buf = [0u8; PROP_VALUE_MAX];
            let len = unsafe { __system_property_get(c_name.as_ptr(), buf.as_mut_ptr().cast()) };
            if len <= 0 {
                return None;
            }
            std::str::from_utf8(&buf[..len as usize])
                .ok()
                .map(str::to_string)
        }
    }

    #[cfg(target_os = "android")]
    pub(super) use probe::detect_soc_chipset;

    #[cfg(test)]
    mod tests {
        use super::map_soc;

        #[test]
        fn passes_through_qti_soc_model() {
            // Captured from a Galaxy S25 (SM-S9310) running One UI 7.
            assert_eq!(map_soc("QTI", "SM8750"), Some("SM8750"));
        }

        #[test]
        fn accepts_aosp_qualcomm_spelling() {
            assert_eq!(map_soc("Qualcomm", "SM8650"), Some("SM8650"));
        }

        #[test]
        fn manufacturer_match_is_case_insensitive() {
            assert_eq!(map_soc("qti", "SM8750"), Some("SM8750"));
            assert_eq!(map_soc("qualcomm", "SM8650"), Some("SM8650"));
        }

        #[test]
        fn rejects_non_qualcomm_manufacturer() {
            assert_eq!(map_soc("Samsung", "S5E9945"), None);
            assert_eq!(map_soc("MediaTek", "MT6989"), None);
            assert_eq!(map_soc("Google", "Tensor G4"), None);
        }

        #[test]
        fn rejects_empty_model() {
            assert_eq!(map_soc("QTI", ""), None);
            assert_eq!(map_soc("QTI", "   "), None);
        }

        #[test]
        fn trims_model_whitespace() {
            assert_eq!(map_soc("QTI", "  SM8750  "), Some("SM8750"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_x_elite_brand_string() {
        let brand = "Snapdragon(R) X Elite - X1E80100 - Qualcomm(R) Oryon(TM) CPU".to_string();
        assert_eq!(
            cpu_name_to_chipset_alias(brand).as_deref(),
            Some("qualcomm-snapdragon-x-elite")
        );
    }

    #[test]
    fn parses_x_plus_brand_string() {
        let brand = "Snapdragon X Plus - X1P64100 - Qualcomm Oryon CPU".to_string();
        assert_eq!(
            cpu_name_to_chipset_alias(brand).as_deref(),
            Some("qualcomm-snapdragon-x-plus-8-core")
        );
    }

    #[test]
    fn parses_x2_elite_brand_string() {
        let brand = "Snapdragon X2 Elite - X2E80100 - Qualcomm Oryon CPU".to_string();
        assert_eq!(
            cpu_name_to_chipset_alias(brand).as_deref(),
            Some("qualcomm-snapdragon-x2-elite")
        );
    }

    #[test]
    fn parses_lower_bin_skus() {
        assert_eq!(
            cpu_name_to_chipset_alias("X1E78100".to_string()).as_deref(),
            Some("qualcomm-snapdragon-x-elite")
        );
        assert_eq!(
            cpu_name_to_chipset_alias("X1P42100".to_string()).as_deref(),
            Some("qualcomm-snapdragon-x-plus-8-core")
        );
    }

    #[test]
    fn ignores_non_oryon_cpus() {
        assert!(
            cpu_name_to_chipset_alias("Intel(R) Core(TM) i7-12700H @ 2.30GHz".to_string())
                .is_none()
        );
        assert!(cpu_name_to_chipset_alias("AMD Ryzen 7 7840U".to_string()).is_none());
        assert!(cpu_name_to_chipset_alias(String::new()).is_none());
    }

    #[test]
    fn rejects_almost_sku_shaped_tokens() {
        assert!(!is_oryon_sku("X1E80100A"));
        assert!(!is_oryon_sku("XEE80100"));
        assert!(!is_oryon_sku("X1E10"));
        assert!(!is_oryon_sku(""));
    }
}
