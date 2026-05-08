//! Windows CPU brand probe via `reg query`.
//!
//! We read the CPU brand string from
//! `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0\ProcessorNameString`,
//! which has been stable since Windows 2000 and doesn't depend on any
//! cmdlet infrastructure — `wmic.exe` was deprecated and is missing
//! from recent Windows 11 builds (24H2+), so shelling out to it is no
//! longer reliable.
//!
//! Any failure (binary missing, parse error, empty value) returns
//! `None` so the caller can fall back to an explicit chipset.

use std::process::Command;

const KEY: &str = r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0";
const VALUE: &str = "ProcessorNameString";

/// Return the CPU brand string reported to firmware, or `None` on error.
pub(super) fn detect() -> Option<String> {
    let out = Command::new("reg")
        .args(["query", KEY, "/v", VALUE])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_reg_query(&String::from_utf8_lossy(&out.stdout))
}

/// Parse a `reg query` stdout block and extract the `REG_SZ` value.
///
/// Expected shape (one blank line before / after the line we want):
/// ```text
/// HKEY_LOCAL_MACHINE\HARDWARE\...\CentralProcessor\0
///     ProcessorNameString    REG_SZ    Snapdragon(R) X 12-core X1E80100 @ 3.40 GHz
/// ```
fn parse_reg_query(stdout: &str) -> Option<String> {
    for line in stdout.lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with(VALUE) {
            continue;
        }
        // Split on "REG_SZ" to isolate the value. The value can contain
        // spaces and punctuation, so we can't naively split_whitespace.
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
        // Verbatim capture from `reg query HKLM\HARDWARE\...` on a real
        // Snapdragon X Elite host (Windows 11 26200).
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
