// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#if defined(_WIN32)
#include <windows.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "types.h"

namespace geniex::qairt::runtime {

// Reads an environment variable as a filesystem path, returning an empty path when
// it is unset or empty. On Windows the wide-char API is used so Unicode paths round-trip
// correctly (the narrow name is ignored there and vice versa on POSIX).
inline std::filesystem::path read_env_path(const char* name_utf8, const wchar_t* name_wide) {
#if defined(_WIN32)
    static_cast<void>(name_utf8);
    size_t required_size = 0;
    _wgetenv_s(&required_size, nullptr, 0, name_wide);
    if (required_size > 0) {
        std::vector<wchar_t> env_buffer(required_size);
        _wgetenv_s(&required_size, env_buffer.data(), required_size, name_wide);
        if (env_buffer[0] != L'\0') {
            return std::filesystem::path(env_buffer.data());
        }
    }
    return {};
#else   // _WIN32
    static_cast<void>(name_wide);
    if (const char* value = std::getenv(name_utf8)) {
        return std::filesystem::path(value);
    }
    return {};
#endif  // _WIN32
}

inline std::vector<std::string> collect_bin_files(const std::filesystem::path& dir) {
    std::vector<std::string> bins;
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        return bins;
    }

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            bins.push_back(entry.path().string());
        }
    }

    std::sort(bins.begin(), bins.end());
    return bins;
}

inline std::optional<std::string> find_optional_file(const std::filesystem::path& dir, const char* filename) {
    const auto file_path = dir / filename;
    if (std::filesystem::exists(file_path)) {
        return file_path.string();
    }
    return std::nullopt;
}

// Platform-specific base names of the three QNN shared libraries the QAIRT plugin loads.
#ifdef _WIN32
constexpr const char* kQnnBackendLib    = "QnnHtp.dll";
constexpr const char* kQnnSystemLib     = "QnnSystem.dll";
constexpr const char* kQnnExtensionsLib = "QnnHtpNetRunExtensions.dll";
#else  // __ANDROID__ and __linux__
constexpr const char* kQnnBackendLib    = "libQnnHtp.so";
constexpr const char* kQnnSystemLib     = "libQnnSystem.so";
constexpr const char* kQnnExtensionsLib = "libQnnHtpNetRunExtensions.so";
#endif

// Returns a QnnRuntimeConfig for the given model directory.
//
// QNN library resolution, in priority order:
//   1. GENIEX_QNN_LIB env var (set directly or via the CLI `--qnn-lib` flag) — points at a
//      folder that directly contains the QNN shared libraries. Lets a single GenieX release
//      run against an arbitrary QAIRT/QNN build without reinstalling. Validated eagerly:
//      a missing folder or a folder without the backend library throws std::runtime_error
//      so the caller surfaces a clear message instead of a late dlopen failure.
//   2. GENIEX_PLUGIN_PATH env var — the bundled layout, libraries under `<path>/qairt/htp-files`
//      (flattened to `<path>` on Android).
//   3. Neither set — fall back to bare library names so the OS loader resolves them from its
//      default search path (the default shipped behavior, unchanged).
inline QnnRuntimeConfig make_qnn_runtime_config(const std::filesystem::path& model_dir) {
    namespace fs = std::filesystem;

    QnnRuntimeConfig runtime_cfg{};

    // (1) Explicit override: GENIEX_QNN_LIB folder holds the QNN libs directly.
    const fs::path qnn_lib_dir = read_env_path("GENIEX_QNN_LIB", L"GENIEX_QNN_LIB");
    if (!qnn_lib_dir.empty()) {
        std::error_code ec;
        if (!fs::is_directory(qnn_lib_dir, ec)) {
            throw std::runtime_error("GENIEX_QNN_LIB path is not a directory: " + qnn_lib_dir.string());
        }
        const fs::path backend = qnn_lib_dir / kQnnBackendLib;
        if (!fs::exists(backend, ec)) {
            throw std::runtime_error(
                "GENIEX_QNN_LIB does not contain " + std::string(kQnnBackendLib) + ": " + qnn_lib_dir.string());
        }

        GENIEX_LOG_INFO("Using custom QNN libraries from GENIEX_QNN_LIB: {}", qnn_lib_dir.string());
        GENIEX_LOG_DEBUG("Setting ADSP_LIBRARY_PATH to {}", qnn_lib_dir.string());
#if defined(WIN32)
        _putenv_s("ADSP_LIBRARY_PATH", qnn_lib_dir.string().c_str());
        SetDllDirectoryA(qnn_lib_dir.string().c_str());
#else
        setenv("ADSP_LIBRARY_PATH", qnn_lib_dir.string().c_str(), 1);
#endif
        runtime_cfg.backend_path    = backend.string();
        runtime_cfg.system_lib_path = (qnn_lib_dir / kQnnSystemLib).string();
        runtime_cfg.extensions_path = (qnn_lib_dir / kQnnExtensionsLib).string();

        static_cast<void>(model_dir);  // reserved for future fallback logic
        return runtime_cfg;
    }

    // (2) Bundled layout via GENIEX_PLUGIN_PATH, else (3) auto-detect (empty backend_dir).
    fs::path backend_dir = read_env_path("GENIEX_PLUGIN_PATH", L"GENIEX_PLUGIN_PATH");

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

    runtime_cfg.backend_path    = (backend_dir / kQnnBackendLib).string();
    runtime_cfg.system_lib_path = (backend_dir / kQnnSystemLib).string();
    runtime_cfg.extensions_path = (backend_dir / kQnnExtensionsLib).string();
#ifdef _WIN32
    SetDllDirectoryA(backend_dir.string().c_str());
#endif

    static_cast<void>(model_dir);  // reserved for future fallback logic
    return runtime_cfg;
}

}  // namespace geniex::qairt::runtime
