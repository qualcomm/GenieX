// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#if defined(_WIN32)
#include <windows.h>
#endif

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "external/json.hpp"
#include "types.h"

namespace geniex::qairt::runtime {

// NOTE: no `collect_bin_files()` helper here on purpose. Context-binary shards must
// come from genie_config.json's `ctx-bins` via the core's `modelConfigFromDirectory()`,
// never from a `*.bin` glob: bundles also ship CPU-side payloads as `.bin`.

inline std::optional<std::string> find_optional_file(const std::filesystem::path& dir, const char* filename) {
    const auto file_path = dir / filename;
    if (std::filesystem::exists(file_path)) {
        return file_path.string();
    }
    return std::nullopt;
}

// dsp_arch values the pinned QAIRT release recognizes; keep in step when the
// QAIRT dependency is bumped.
inline constexpr std::array<const char*, 6> kRecognizedDspArchs = {"v68", "v69", "v73", "v75", "v79", "v81"};

// Returns a loggable diagnostic when `htp_backend_ext_config.json` contains a
// dsp_arch outside kRecognizedDspArchs; nullopt otherwise (including a missing
// or unparseable file — QNN owns those failure modes).
inline std::optional<std::string> dsp_arch_diagnostic(const std::string& htp_config_path) {
    if (htp_config_path.empty()) {
        return std::nullopt;
    }
    std::ifstream file(htp_config_path);
    if (!file.is_open()) {
        return std::nullopt;
    }
    const nlohmann::json root = nlohmann::json::parse(file, nullptr, /*allow_exceptions=*/false);
    if (root.is_discarded()) {
        return std::nullopt;
    }

    // Exporters nest dsp_arch differently, so walk the whole document.
    std::vector<std::string>           unrecognized;
    std::vector<const nlohmann::json*> stack{&root};
    while (!stack.empty()) {
        const nlohmann::json* node = stack.back();
        stack.pop_back();
        if (node->is_object()) {
            for (auto it = node->begin(); it != node->end(); ++it) {
                if (it.key() == "dsp_arch" && it.value().is_string()) {
                    const auto arch  = it.value().get<std::string>();
                    const bool known = std::find(kRecognizedDspArchs.begin(), kRecognizedDspArchs.end(), arch) !=
                                       kRecognizedDspArchs.end();
                    if (!known && std::find(unrecognized.begin(), unrecognized.end(), arch) == unrecognized.end()) {
                        unrecognized.push_back(arch);
                    }
                } else {
                    stack.push_back(&it.value());
                }
            }
        } else if (node->is_array()) {
            for (const auto& child : *node) {
                stack.push_back(&child);
            }
        }
    }
    if (unrecognized.empty()) {
        return std::nullopt;
    }

    std::string offending;
    for (const auto& arch : unrecognized) {
        if (!offending.empty()) {
            offending += ", ";
        }
        offending += "'" + arch + "'";
    }
    std::string recognized;
    for (const char* arch : kRecognizedDspArchs) {
        if (!recognized.empty()) {
            recognized += ", ";
        }
        recognized += arch;
    }
    return "htp_backend_ext_config.json specifies dsp_arch " + offending +
           ", which this QAIRT release does not recognize (recognized values: " + recognized +
           "). The bundle was likely exported for a newer QAIRT; re-export it for this runtime or update QAIRT.";
}

// Returns a QnnRuntimeConfig for the given model directory and optional user-supplied
// QNN lib folder path.
//
// If qnn_lib_folder_path is non-empty, the three HTP runtime path fields are set
// explicitly to that directory (user override). Otherwise all path fields are left
// as std::nullopt to let geniex_qairt resolve them automatically at runtime (default behavior).
inline QnnRuntimeConfig make_qnn_runtime_config(const std::filesystem::path& model_dir) {
    namespace fs = std::filesystem;

    QnnRuntimeConfig runtime_cfg{};

    fs::path backend_dir;
#if defined(_WIN32)
    // On Windows, use wide string API to properly handle Unicode paths
    size_t required_size = 0;
    _wgetenv_s(&required_size, nullptr, 0, L"GENIEX_PLUGIN_PATH");
    if (required_size > 0) {
        std::vector<wchar_t> env_buffer(required_size);
        _wgetenv_s(&required_size, env_buffer.data(), required_size, L"GENIEX_PLUGIN_PATH");
        if (env_buffer[0] != L'\0') {
            backend_dir = std::filesystem::path(env_buffer.data());
        }
    }
#else   // _WIN32
    auto env_plugin_path = std::getenv("GENIEX_PLUGIN_PATH");
    if (env_plugin_path) {
        backend_dir = fs::path(env_plugin_path);
    }
#endif  // _WIN32

#if not defined(__ANDROID__)  // android has flattened directory
    if (!backend_dir.empty()) {
        backend_dir = backend_dir / "qairt" / "htp-files";
    }
#endif  // not __ANDROID__

    GENIEX_LOG_DEBUG("Setting ADSP_LIBRARY_PATH to {}", backend_dir.string());
#if defined(WIN32)
    _putenv_s("ADSP_LIBRARY_PATH", backend_dir.string().c_str());
#else
    setenv("ADSP_LIBRARY_PATH", backend_dir.string().c_str(), 1);
#endif

#ifdef _WIN32
    runtime_cfg.backend_path    = (backend_dir / "QnnHtp.dll").string();
    runtime_cfg.system_lib_path = (backend_dir / "QnnSystem.dll").string();
    runtime_cfg.extensions_path = (backend_dir / "QnnHtpNetRunExtensions.dll").string();
    SetDllDirectoryA(backend_dir.string().c_str());
#else  // __ANDROID__ and __linux__
    runtime_cfg.backend_path    = (backend_dir / "libQnnHtp.so").string();
    runtime_cfg.system_lib_path = (backend_dir / "libQnnSystem.so").string();
    runtime_cfg.extensions_path = (backend_dir / "libQnnHtpNetRunExtensions.so").string();
#endif

    static_cast<void>(model_dir);  // reserved for future fallback logic
    return runtime_cfg;
}

}  // namespace geniex::qairt::runtime
