#pragma once

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

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

inline std::filesystem::path resolve_qnn_lib_dir(
    const std::filesystem::path& model_dir,
    const char* qnn_lib_folder_path) {
    namespace fs = std::filesystem;

    if (qnn_lib_folder_path && qnn_lib_folder_path[0] != '\0') {
        auto qnn_lib_dir = fs::path(qnn_lib_folder_path);
        return qnn_lib_dir.is_relative() ? fs::absolute(qnn_lib_dir) : qnn_lib_dir;
    }

    auto has_qnn = [](const fs::path& dir) {
        return fs::exists(dir / "QnnHtp.dll");
    };

    fs::path qnn_lib_dir;
    if (has_qnn(model_dir)) {
        qnn_lib_dir = model_dir;
    } else if (has_qnn(model_dir / "htp-files")) {
        qnn_lib_dir = model_dir / "htp-files";
    }

#if defined(_WIN32)
    if (qnn_lib_dir.empty()) {
        if (const auto* env = std::getenv("GENIEX_PLUGIN_PATH")) {
            const auto candidate = fs::path(env) / "qairt" / "htp-files";
            if (has_qnn(candidate)) {
                qnn_lib_dir = candidate;
            }
        }
    }

    if (qnn_lib_dir.empty()) {
        HMODULE module = GetModuleHandleA("geniex_plugin.dll");
        if (module) {
            char path_buf[MAX_PATH];
            if (GetModuleFileNameA(module, path_buf, MAX_PATH)) {
                const auto plugin_dir = fs::path(path_buf).parent_path();
                if (has_qnn(plugin_dir)) {
                    qnn_lib_dir = plugin_dir;
                } else if (has_qnn(plugin_dir / "htp-files")) {
                    qnn_lib_dir = plugin_dir / "htp-files";
                }
            }
        }
    }
#endif

    if (qnn_lib_dir.empty()) {
        qnn_lib_dir = model_dir;
    }

    return qnn_lib_dir.is_relative() ? fs::absolute(qnn_lib_dir) : qnn_lib_dir;
}

inline QnnRuntimeConfig make_qnn_runtime_config(const std::filesystem::path& qnn_lib_dir) {
    QnnRuntimeConfig runtime_cfg{};
    runtime_cfg.backend_path = (qnn_lib_dir / "QnnHtp.dll").string();
    runtime_cfg.system_lib_path = (qnn_lib_dir / "QnnSystem.dll").string();
    runtime_cfg.extensions_path = (qnn_lib_dir / "QnnHtpNetRunExtensions.dll").string();
    return runtime_cfg;
}

inline void configure_qnn_dll_search_path(const std::filesystem::path& qnn_lib_dir) {
#if defined(_WIN32)
    SetDllDirectoryA(qnn_lib_dir.string().c_str());
#else
    static_cast<void>(qnn_lib_dir);
#endif
}

}  // namespace geniex::qairt::runtime