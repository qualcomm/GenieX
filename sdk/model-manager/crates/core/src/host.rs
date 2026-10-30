// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Best-effort host chipset detection for AI Hub pulls; `None` means the caller must pass one.
pub fn detect_chipset() -> Option<String> {
    #[cfg(target_os = "windows")]
    {
        windows::detect_gpu_chipset()
            .or_else(|| windows::detect_cpu_brand().and_then(cpu_name_to_chipset_alias))
    }
    #[cfg(target_os = "linux")]
    {
        linux::detect_dt_chipset().or_else(linux::detect_socinfo_chipset)
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

#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
pub(crate) fn cpu_name_to_chipset_alias(brand: String) -> Option<String> {
    let sku = extract_oryon_sku(&brand)?;
    match &sku[..3] {
        "X1E" => Some("qualcomm-snapdragon-x-elite".to_string()),
        "X1P" => Some("qualcomm-snapdragon-x-plus-8-core".to_string()),
        "X2E" => Some("qualcomm-snapdragon-x2-elite".to_string()),
        "X2P" => Some("qualcomm-snapdragon-x2-elite".to_string()),
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
    //! Probes via `reg query`: `wmic.exe` is missing on Windows 11 24H2+.
    //! The Adreno GPU name carries the X-series generation (`X<gen>-`), which
    //! maps to one HTP arch, so it is tried before the per-SKU CPU brand table.

    use std::process::Command;

    const KEY: &str = r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0";
    const VALUE: &str = "ProcessorNameString";

    const DISPLAY_CLASS_KEY: &str =
        r"HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}";
    const MATCHING_DEVICE_ID: &str = "MatchingDeviceId";
    const DRIVER_DESC: &str = "DriverDesc";

    // The gen-1 id has no digit (`x-elite`) while gen-2 does (`x2-elite`),
    // so this is an explicit table rather than a string built from the number.
    const GPU_GEN_TO_CHIPSET: &[(u32, &str)] = &[
        (1, "qualcomm-snapdragon-x-elite"),
        (2, "qualcomm-snapdragon-x2-elite"),
    ];

    pub(super) fn detect_gpu_chipset() -> Option<String> {
        let subkey = query_qualcomm_display_subkey()?;
        let desc = query_driver_desc(&subkey)?;
        adreno_name_to_chipset(&desc).map(str::to_string)
    }

    fn query_qualcomm_display_subkey() -> Option<String> {
        let out = Command::new("reg")
            .args(["query", DISPLAY_CLASS_KEY, "/s", "/v", MATCHING_DEVICE_ID])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        find_qualcomm_subkey(&String::from_utf8_lossy(&out.stdout))
    }

    fn query_driver_desc(subkey: &str) -> Option<String> {
        let out = Command::new("reg")
            .args(["query", subkey, "/v", DRIVER_DESC])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        parse_reg_sz(&String::from_utf8_lossy(&out.stdout), DRIVER_DESC)
    }

    fn find_qualcomm_subkey(stdout: &str) -> Option<String> {
        let mut current: Option<&str> = None;
        for line in stdout.lines() {
            if line.starts_with("HKEY_") {
                current = Some(line.trim_end());
                continue;
            }
            let trimmed = line.trim_start();
            if !trimmed.starts_with(MATCHING_DEVICE_ID) {
                continue;
            }
            let Some((_, value)) = trimmed.split_once("REG_SZ") else {
                continue;
            };
            if value.to_ascii_uppercase().contains("VEN_QCOM") {
                return current.map(str::to_string);
            }
        }
        None
    }

    fn parse_reg_sz(stdout: &str, value_name: &str) -> Option<String> {
        for line in stdout.lines() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with(value_name) {
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

    fn adreno_name_to_chipset(desc: &str) -> Option<&'static str> {
        let gen = extract_adreno_generation(desc)?;
        GPU_GEN_TO_CHIPSET
            .iter()
            .find(|(g, _)| *g == gen)
            .map(|(_, c)| *c)
    }

    fn extract_adreno_generation(desc: &str) -> Option<u32> {
        for tok in desc.split(|c: char| !c.is_ascii_alphanumeric()) {
            let Some(rest) = tok.strip_prefix(['X', 'x']) else {
                continue;
            };
            if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
                if let Ok(n) = rest.parse::<u32>() {
                    return Some(n);
                }
            }
        }
        None
    }

    pub(super) fn detect_cpu_brand() -> Option<String> {
        let out = Command::new("reg")
            .args(["query", KEY, "/v", VALUE])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        parse_reg_sz(&String::from_utf8_lossy(&out.stdout), VALUE)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parses_reg_sz_values() {
            let cpu = "\
HKEY_LOCAL_MACHINE\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0\r\n    \
ProcessorNameString    REG_SZ    Snapdragon(R) X 12-core X1E80100 @ 3.40 GHz\r\n\r\n";
            assert_eq!(
                parse_reg_sz(cpu, VALUE).as_deref(),
                Some("Snapdragon(R) X 12-core X1E80100 @ 3.40 GHz")
            );
            assert!(parse_reg_sz("", VALUE).is_none());

            let gpu = "\
HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-...}\\0000\r\n    \
DriverDesc    REG_SZ    Qualcomm(R) Adreno(TM) X2-45 GPU\r\n\r\n";
            assert_eq!(
                parse_reg_sz(gpu, DRIVER_DESC).as_deref(),
                Some("Qualcomm(R) Adreno(TM) X2-45 GPU")
            );
        }

        #[test]
        fn finds_qualcomm_display_subkey_by_vendor_not_slot() {
            let three_adapters = "\r\n\
HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0000\r\n    \
MatchingDeviceId    REG_SZ    ACPI\\VEN_QCOM&DEV_0FF5&REV_0098\r\n\r\n\
HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0001\r\n    \
MatchingDeviceId    REG_SZ    RdpIdd_IndirectDisplay\r\n\r\n\
HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0002\r\n    \
MatchingDeviceId    REG_SZ    Root\\ThirdPartyMirror\r\n\r\n";
            let subkey = find_qualcomm_subkey(three_adapters).expect("subkey");
            assert!(subkey.ends_with("\\0000"), "got {subkey}");

            let virtual_only = "\
HKEY_LOCAL_MACHINE\\...\\0000\r\n    MatchingDeviceId    REG_SZ    RdpIdd_IndirectDisplay\r\n\r\n\
HKEY_LOCAL_MACHINE\\...\\0001\r\n    MatchingDeviceId    REG_SZ    Root\\ThirdPartyMirror\r\n";
            assert_eq!(find_qualcomm_subkey(virtual_only), None);
        }

        #[test]
        fn maps_adreno_names() {
            for (name, want) in [
                (
                    "Qualcomm(R) Adreno(TM) X2-45 GPU",
                    Some("qualcomm-snapdragon-x2-elite"),
                ),
                (
                    "Qualcomm(R) Adreno(TM) X1-85 GPU",
                    Some("qualcomm-snapdragon-x-elite"),
                ),
                (
                    "Qualcomm(R) Adreno(TM) X1-45 GPU",
                    Some("qualcomm-snapdragon-x-elite"),
                ),
                ("Adreno(TM) X9-99 GPU", None),
                ("ThirdPartyMirror Device", None),
            ] {
                assert_eq!(adreno_name_to_chipset(name), want, "{name:?}");
            }
            for (desc, want) in [
                ("Adreno(TM) X2-45 GPU", Some(2)),
                ("Adreno(TM) X1-85 GPU", Some(1)),
                ("Intel(R) UHD Graphics", None),
                ("XG301062", None),
            ] {
                assert_eq!(extract_adreno_generation(desc), want, "{desc:?}");
            }
        }
    }
}

#[cfg(target_os = "linux")]
mod linux {
    //! Devicetree `compatible` first; socinfo is the fallback for boards like
    //! IQ-8275 whose compat list names only `qcs8300`, not the real `QCS8275`.

    use std::fs;

    const DT_COMPATIBLE: &str = "/sys/firmware/devicetree/base/compatible";
    const SOC0_FAMILY: &str = "/sys/devices/soc0/family";
    const SOC0_MACHINE: &str = "/sys/devices/soc0/machine";

    const SOC_TO_CHIPSET: &[(&str, &str)] = &[
        ("qcs6490", "qualcomm-qcs6490"),
        ("qcm6490", "qualcomm-qcs6490"),
        ("qcs9075", "qualcomm-qcs9075"),
    ];

    const MACHINE_TO_CHIPSET: &[(&str, &str)] = &[("QCS8275", "qualcomm-qcs8275")];

    pub(super) fn detect_dt_chipset() -> Option<String> {
        let bytes = fs::read(DT_COMPATIBLE).ok()?;
        map_compatible(&bytes).map(|s| s.to_string())
    }

    pub(super) fn detect_socinfo_chipset() -> Option<String> {
        let family = fs::read_to_string(SOC0_FAMILY).ok()?;
        let machine = fs::read_to_string(SOC0_MACHINE).ok()?;
        map_socinfo(&family, &machine).map(|s| s.to_string())
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

    fn map_socinfo(family: &str, machine: &str) -> Option<&'static str> {
        if !family.trim().eq_ignore_ascii_case("Snapdragon") {
            return None;
        }
        let m = machine.trim().to_ascii_uppercase();
        MACHINE_TO_CHIPSET
            .iter()
            .find(|(k, _)| *k == m)
            .map(|(_, v)| *v)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn map_compatible_cases() {
            let cases: [(&[u8], Option<&str>); 7] = [
                (
                    b"qcom,qcs9075-addons-iq-9075-evk\0qcom,qcs9075\0qcom,sa8775p\0",
                    Some("qualcomm-qcs9075"),
                ),
                (
                    b"qcom,qcs6490-addons-rb3gen2\0qcom,qcm6490\0",
                    Some("qualcomm-qcs6490"),
                ),
                (b"qcom,qcs9075", Some("qualcomm-qcs9075")),
                (b"qcom,sa8775p\0", None),
                (b"brcm,bcm2837\0", None),
                (b"", None),
                (b"qcom,qcs9075foo\0", None),
            ];
            for (bytes, want) in cases {
                assert_eq!(
                    map_compatible(bytes),
                    want,
                    "{:?}",
                    String::from_utf8_lossy(bytes)
                );
            }
        }

        #[test]
        fn map_socinfo_cases() {
            for (family, machine, want) in [
                ("Snapdragon\n", "QCS8275\n", Some("qualcomm-qcs8275")),
                ("snapdragon\n", "qcs8275\n", Some("qualcomm-qcs8275")),
                ("MSM\n", "QCS8275\n", None),
                ("", "QCS8275", None),
                ("Snapdragon\n", "QCS8300\n", None),
            ] {
                assert_eq!(map_socinfo(family, machine), want, "{family:?} {machine:?}");
            }
        }
    }
}

mod android {
    //! Reads `ro.soc.model`: the devicetree is unreadable from app SELinux domains.
    //! The model doubles as an AI Hub alias, except SM8750/SM8850, which the
    //! catalogue lists only as `-ac`/`-ad`. Samsung reports `QTI`, AOSP `Qualcomm`.

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
        if model.eq_ignore_ascii_case("SM8750") {
            return Some("SM8750-AC");
        }
        if model.eq_ignore_ascii_case("SM8850") {
            return Some("SM8850-AD");
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

        pub(in crate::host) fn detect_soc_chipset() -> Option<String> {
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
        fn map_soc_cases() {
            for (manuf, model, want) in [
                ("QTI", "SM8650", Some("SM8650")),
                ("QTI", "SM8750", Some("SM8750-AC")),
                ("QTI", "sm8850", Some("SM8850-AD")),
                ("Qualcomm", "SM8650", Some("SM8650")),
                ("qti", "SM8750", Some("SM8750-AC")),
                ("qualcomm", "SM8650", Some("SM8650")),
                ("QTI", "  SM8650  ", Some("SM8650")),
                ("Samsung", "S5E9945", None),
                ("MediaTek", "MT6989", None),
                ("Google", "Tensor G4", None),
                ("QTI", "", None),
                ("QTI", "   ", None),
            ] {
                assert_eq!(map_soc(manuf, model), want, "{manuf:?} {model:?}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_brand_maps_to_chipset_alias() {
        let x_elite = Some("qualcomm-snapdragon-x-elite");
        let x_plus = Some("qualcomm-snapdragon-x-plus-8-core");
        let x2_elite = Some("qualcomm-snapdragon-x2-elite");
        for (brand, want) in [
            (
                "Snapdragon(R) X Elite - X1E80100 - Qualcomm(R) Oryon(TM) CPU",
                x_elite,
            ),
            ("Snapdragon X Plus - X1P64100 - Qualcomm Oryon CPU", x_plus),
            (
                "Snapdragon X2 Elite - X2E80100 - Qualcomm Oryon CPU",
                x2_elite,
            ),
            (
                "Snapdragon X2 Plus - X2P64100 - Qualcomm Oryon CPU",
                x2_elite,
            ),
            ("X1E78100", x_elite),
            ("X1P42100", x_plus),
            ("Intel(R) Core(TM) i7-12700H @ 2.30GHz", None),
            ("AMD Ryzen 7 7840U", None),
            ("", None),
        ] {
            assert_eq!(
                cpu_name_to_chipset_alias(brand.to_string()).as_deref(),
                want,
                "{brand:?}"
            );
        }
    }

    #[test]
    fn rejects_almost_sku_shaped_tokens() {
        for tok in ["X1E80100A", "XEE80100", "X1E10", ""] {
            assert!(!is_oryon_sku(tok), "{tok:?}");
        }
    }
}
