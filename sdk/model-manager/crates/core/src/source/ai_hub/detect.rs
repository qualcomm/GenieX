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
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

/// Map a Qualcomm Oryon SKU number (extracted from the CPU brand
/// string) to the matching AI Hub chipset alias. Covers the SKUs
/// published as of QAIRT 2.45; unknown SKUs return `None` so we
/// fall back to the caller-supplied chipset.
pub(crate) fn cpu_name_to_chipset_alias(brand: String) -> Option<String> {
    let sku = extract_oryon_sku(&brand)?;
    match &sku[..3] {
        "X1E" => Some("qualcomm-snapdragon-x-elite".to_string()),
        "X1P" => Some("qualcomm-snapdragon-x-plus-8-core".to_string()),
        "X2E" => Some("qualcomm-snapdragon-x2-elite".to_string()),
        _ => None,
    }
}

fn extract_oryon_sku(brand: &str) -> Option<String> {
    for tok in brand.split(|c: char| !(c.is_ascii_alphanumeric())) {
        if is_oryon_sku(tok) {
            return Some(tok.to_ascii_uppercase());
        }
    }
    None
}

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
