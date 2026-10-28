// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace geniex::qairt::paths {
#if defined(_WIN32)
inline constexpr const char* kQnnBackendLib    = "QnnHtp.dll";
inline constexpr const char* kQnnSystemLib     = "QnnSystem.dll";
inline constexpr const char* kQnnExtensionsLib = "QnnHtpNetRunExtensions.dll";
inline constexpr const char* kHostLibTriple    = "aarch64-windows-msvc";
#elif defined(__ANDROID__)
inline constexpr const char* kQnnBackendLib    = "libQnnHtp.so";
inline constexpr const char* kQnnSystemLib     = "libQnnSystem.so";
inline constexpr const char* kQnnExtensionsLib = "libQnnHtpNetRunExtensions.so";
inline constexpr const char* kHostLibTriple    = "aarch64-android";
#else
inline constexpr const char* kQnnBackendLib    = "libQnnHtp.so";
inline constexpr const char* kQnnSystemLib     = "libQnnSystem.so";
inline constexpr const char* kQnnExtensionsLib = "libQnnHtpNetRunExtensions.so";
inline constexpr const char* kHostLibTriple    = "aarch64-oe-linux-gcc11.2";
#endif
// FastRPC uses semicolons on Linux/Android too.
inline constexpr char        kPathSep = ';';
inline std::filesystem::path locate_qnn_host_lib_dir(const std::filesystem::path& root) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::exists(root / kQnnBackendLib, ec)) return root;
    const fs::path triple_dir = root / "lib" / kHostLibTriple;
    if (fs::exists(triple_dir / kQnnBackendLib, ec)) return triple_dir;
    const fs::path lib_dir = root / "lib";
    if (fs::is_directory(lib_dir, ec)) {
        for (const auto& entry : fs::directory_iterator(lib_dir, ec)) {
            if (entry.is_directory(ec) && fs::exists(entry.path() / kQnnBackendLib, ec)) return entry.path();
        }
    }
    return {};
}
inline std::string collect_adsp_library_path(const std::filesystem::path& root) {
    namespace fs = std::filesystem;
    std::error_code          ec;
    std::vector<std::string> dirs;
    fs::path                 lib_dir = root / "lib";
    // A caller may select the host directory inside a full SDK. Its skels are
    // sibling hexagon-* directories, not beside the host DLLs.
    if (!fs::is_directory(lib_dir, ec) && root.parent_path().filename() == "lib") lib_dir = root.parent_path();
    if (fs::is_directory(lib_dir, ec)) {
        for (const auto& entry : fs::directory_iterator(lib_dir, ec)) {
            if (!entry.is_directory(ec) || entry.path().filename().string().rfind("hexagon-", 0) != 0) continue;
            const auto unsigned_dir = entry.path() / "unsigned";
            dirs.push_back(fs::is_directory(unsigned_dir, ec) ? unsigned_dir.string() : entry.path().string());
        }
    }
    std::sort(dirs.begin(), dirs.end());
    std::string joined;
    for (const auto& dir : dirs) {
        if (!joined.empty()) joined.push_back(kPathSep);
        joined += dir;
    }
    return joined;
}
}  // namespace geniex::qairt::paths
