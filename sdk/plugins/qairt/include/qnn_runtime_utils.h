#pragma once

#if defined(_WIN32)
#include <windows.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <optional>
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

inline std::optional<std::string> find_optional_file(const std::filesystem::path& dir, const char* filename) {
    const auto file_path = dir / filename;
    if (std::filesystem::exists(file_path)) {
        return file_path.string();
    }
    return std::nullopt;
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
