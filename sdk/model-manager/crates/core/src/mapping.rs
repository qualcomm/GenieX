/// Resolve a short model alias to its canonical "org/repo" name.
/// Matches against the current OS and architecture (same logic as cli/internal/config/model_mapping.go).
///
/// Each entry: (os, arch, full_name).
/// Empty string for os/arch means "any". Later entries override earlier ones when both match.
pub fn resolve_alias(alias: &str) -> Option<String> {
    let entries: &[(&str, &[(&str, &str, &str)])] = &[
        ("qwen3", &[("", "", "NexaAI/Qwen3-4B-GGUF")]),
        ("qwen2vl", &[("", "", "ggml-org/Qwen2-VL-2B-Instruct-GGUF")]),
        (
            "qwen2.5vl",
            &[
                ("", "", "unsloth/Qwen2.5-VL-3B-Instruct-GGUF"),
                ("macos", "aarch64", "Qwen/Qwen2.5-VL-3B-Instruct"),
            ],
        ),
        (
            "qwen3vl",
            &[
                ("windows", "x86_64", "NexaAI/Qwen3-VL-4B-GGUF"),
                ("windows", "aarch64", "NexaAI/Qwen3-VL-4B-NPU"),
            ],
        ),
        ("gemma3", &[("", "", "ggml-org/gemma-3-4b-it-GGUF")]),
        (
            "smolvlm",
            &[("", "", "ggml-org/SmolVLM-500M-Instruct-GGUF")],
        ),
        ("gpt-oss", &[("", "", "NexaAI/gpt-oss-20b-GGUF")]),
        (
            "omni-neural",
            &[("windows", "aarch64", "NexaAI/OmniNeural-4B")],
        ),
    ];

    let cur_os = std::env::consts::OS; // "linux", "macos", "windows"
    let cur_arch = std::env::consts::ARCH; // "x86_64", "aarch64", etc.

    let mut result: Option<&str> = None;
    for &(key, variants) in entries {
        if key != alias {
            continue;
        }
        for &(os, arch, full_name) in variants {
            let os_match = os.is_empty() || os == cur_os;
            let arch_match = arch.is_empty() || arch == cur_arch;
            if os_match && arch_match {
                result = Some(full_name);
            }
        }
    }
    result.map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_alias_resolves() {
        assert!(resolve_alias("qwen3").is_some());
    }

    #[test]
    fn unknown_alias_returns_none() {
        assert!(resolve_alias("nonexistent_model_xyz").is_none());
    }
}
