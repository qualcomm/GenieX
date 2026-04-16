#pragma once

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "types.h"

namespace geniex::qairt::runtime {

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

inline std::string find_optional_file(const std::filesystem::path& dir, const char* filename) {
    const auto file_path = dir / filename;
    return std::filesystem::exists(file_path) ? file_path.string() : std::string{};
}

// Returns a QnnRuntimeConfig for the given model directory and optional user-supplied
// QNN lib folder path.
//
// If qnn_lib_folder_path is non-empty, the three HTP runtime path fields are set
// explicitly to that directory (user override). Otherwise all path fields are left
// as std::nullopt to let geniex_qairt resolve them automatically at runtime (default behavior).
inline QnnRuntimeConfig make_qnn_runtime_config(
    const std::filesystem::path& model_dir,
    const char*                  qnn_lib_folder_path) {
    namespace fs = std::filesystem;

    QnnRuntimeConfig runtime_cfg{};

    if (qnn_lib_folder_path && qnn_lib_folder_path[0] != '\0') {
        // Explicit user override: resolve the path and set all three fields.
        auto lib_dir = fs::path(qnn_lib_folder_path);
        if (lib_dir.is_relative()) {
            lib_dir = fs::absolute(lib_dir);
        }
        runtime_cfg.backend_path    = (lib_dir / "QnnHtp.dll").string();
        runtime_cfg.system_lib_path = (lib_dir / "QnnSystem.dll").string();
        runtime_cfg.extensions_path = (lib_dir / "QnnHtpNetRunExtensions.dll").string();
    }
    // Otherwise leave all path fields as std::nullopt — geniex_core will
    // call resolveHtpPaths() in Model::initialize() and fill them in automatically.

    static_cast<void>(model_dir);  // reserved for future fallback logic
    return runtime_cfg;
}

}  // namespace geniex::qairt::runtime