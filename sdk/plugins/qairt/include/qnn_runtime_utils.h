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

// Platform-specific base names of the three QNN shared libraries the QAIRT plugin loads,
// the host-lib subfolder they live in inside a QAIRT SDK install, and the OS PATH separator.
#if defined(_WIN32)
constexpr const char* kQnnBackendLib    = "QnnHtp.dll";
constexpr const char* kQnnSystemLib     = "QnnSystem.dll";
constexpr const char* kQnnExtensionsLib = "QnnHtpNetRunExtensions.dll";
constexpr const char* kHostLibTriple    = "aarch64-windows-msvc";
constexpr char        kPathSep          = ';';
#elif defined(__ANDROID__)
constexpr const char* kQnnBackendLib    = "libQnnHtp.so";
constexpr const char* kQnnSystemLib     = "libQnnSystem.so";
constexpr const char* kQnnExtensionsLib = "libQnnHtpNetRunExtensions.so";
constexpr const char* kHostLibTriple    = "aarch64-android";
constexpr char        kPathSep          = ':';
#else  // __linux__
constexpr const char* kQnnBackendLib    = "libQnnHtp.so";
constexpr const char* kQnnSystemLib     = "libQnnSystem.so";
constexpr const char* kQnnExtensionsLib = "libQnnHtpNetRunExtensions.so";
constexpr const char* kHostLibTriple    = "aarch64-oe-linux-gcc11.2";
constexpr char        kPathSep          = ':';
#endif

// Given a user-supplied QNN lib location, returns the directory that actually holds the
// host QNN libraries (kQnnBackendLib). Handles both a flat folder (libs sit directly in
// `root`, our bundled htp-files layout) and a full QAIRT SDK install, where the host libs
// live under `root/lib/<triple>`. Returns an empty path when none is found.
inline std::filesystem::path locate_qnn_host_lib_dir(const std::filesystem::path& root) {
    namespace fs = std::filesystem;
    std::error_code ec;

    // Flat folder: libs directly inside root.
    if (fs::exists(root / kQnnBackendLib, ec)) return root;

    // QAIRT SDK: canonical per-platform triple.
    const fs::path triple_dir = root / "lib" / kHostLibTriple;
    if (fs::exists(triple_dir / kQnnBackendLib, ec)) return triple_dir;

    // Fallback: scan lib/* for any subfolder carrying the backend lib. Covers QAIRT
    // triples that vary across SDK versions (e.g. differing Linux gcc suffixes). Hexagon
    // skel folders never contain the host backend lib, so they are naturally skipped.
    const fs::path lib_dir = root / "lib";
    if (fs::is_directory(lib_dir, ec)) {
        for (const auto& entry : fs::directory_iterator(lib_dir, ec)) {
            if (entry.is_directory(ec) && fs::exists(entry.path() / kQnnBackendLib, ec)) {
                return entry.path();
            }
        }
    }
    return {};
}

// Collects the Hexagon DSP skel folders that ADSP_LIBRARY_PATH must point at inside a QAIRT
// SDK: every `root/lib/hexagon-v*/unsigned` (or the arch folder itself when there is no
// `unsigned` subdir), joined with the platform PATH separator. All arch variants are listed
// so QNN's loader selects the one matching the on-device HTP arch. Empty when none exist
// (flat-folder override, where the skels sit alongside the host libs).
inline std::string collect_adsp_library_path(const std::filesystem::path& root) {
    namespace fs = std::filesystem;
    std::error_code          ec;
    std::vector<std::string> dirs;

    const fs::path lib_dir = root / "lib";
    if (fs::is_directory(lib_dir, ec)) {
        for (const auto& entry : fs::directory_iterator(lib_dir, ec)) {
            if (!entry.is_directory(ec)) continue;
            if (entry.path().filename().string().rfind("hexagon-", 0) != 0) continue;
            const fs::path unsigned_dir = entry.path() / "unsigned";
            dirs.push_back(fs::is_directory(unsigned_dir, ec) ? unsigned_dir.string() : entry.path().string());
        }
    }

    std::sort(dirs.begin(), dirs.end());
    std::string joined;
    for (const auto& d : dirs) {
        if (!joined.empty()) joined.push_back(kPathSep);
        joined += d;
    }
    return joined;
}

// Returns a QnnRuntimeConfig for the given model directory.
//
// QNN library resolution, in priority order:
//   1. GENIEX_QNN_LIB env var (set directly or via the CLI `--qnn-lib` flag). Lets a single
//      GenieX release run against an arbitrary QAIRT/QNN build without reinstalling. The value
//      may be either a full QAIRT SDK root (host libs under `lib/<triple>`, Hexagon DSP skels
//      under `lib/hexagon-v*/unsigned`) or a flat folder holding the libs directly. Host-lib
//      and DSP-skel locations are resolved separately because a real QAIRT SDK does not colocate
//      them. Validated eagerly: if no host backend library is found the load fails fast with a
//      std::runtime_error so the caller surfaces a clear message instead of a late dlopen error.
//   2. GENIEX_PLUGIN_PATH env var — the bundled layout, libraries under `<path>/qairt/htp-files`
//      (flattened to `<path>` on Android).
//   3. Neither set — fall back to bare library names so the OS loader resolves them from its
//      default search path (the default shipped behavior, unchanged).
inline QnnRuntimeConfig make_qnn_runtime_config(const std::filesystem::path& model_dir) {
    namespace fs = std::filesystem;

    QnnRuntimeConfig runtime_cfg{};

    // (1) Explicit override: GENIEX_QNN_LIB points at a QAIRT SDK root or a flat lib folder.
    const fs::path qnn_lib_root = read_env_path("GENIEX_QNN_LIB", L"GENIEX_QNN_LIB");
    if (!qnn_lib_root.empty()) {
        std::error_code ec;
        if (!fs::is_directory(qnn_lib_root, ec)) {
            throw std::runtime_error("GENIEX_QNN_LIB path is not a directory: " + qnn_lib_root.string());
        }
        const fs::path host_dir = locate_qnn_host_lib_dir(qnn_lib_root);
        if (host_dir.empty()) {
            throw std::runtime_error("GENIEX_QNN_LIB does not contain " + std::string(kQnnBackendLib) +
                                     " (looked in the folder itself and lib/" + kHostLibTriple +
                                     "): " + qnn_lib_root.string());
        }

        // ADSP_LIBRARY_PATH points at the Hexagon DSP skel folders. In a QAIRT SDK these live
        // apart from the host libs; if none are found (flat folder) fall back to the host dir,
        // which is also where our bundled skels sit.
        std::string adsp_path = collect_adsp_library_path(qnn_lib_root);
        if (adsp_path.empty()) adsp_path = host_dir.string();

        GENIEX_LOG_INFO("Using custom QNN libraries from GENIEX_QNN_LIB: {} (host libs: {})",
            qnn_lib_root.string(),
            host_dir.string());
        GENIEX_LOG_DEBUG("Setting ADSP_LIBRARY_PATH to {}", adsp_path);
#if defined(WIN32)
        _putenv_s("ADSP_LIBRARY_PATH", adsp_path.c_str());
        SetDllDirectoryA(host_dir.string().c_str());
#else
        setenv("ADSP_LIBRARY_PATH", adsp_path.c_str(), 1);
#endif
        runtime_cfg.backend_path    = (host_dir / kQnnBackendLib).string();
        runtime_cfg.system_lib_path = (host_dir / kQnnSystemLib).string();
        runtime_cfg.extensions_path = (host_dir / kQnnExtensionsLib).string();

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
