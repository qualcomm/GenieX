// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause



#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include "logging.h"
#include "plugin/Plugin.h"
#include "registry.h"
#include "utils.h"

const char* get_geniex_plugin_name() {
#if defined(_WIN32)
    return "geniex_plugin.dll";
#else
    return "libgeniex_plugin.so";
#endif
}

namespace {

// The qairt plugin ships one variant per QAIRT/QNN ABI version, installed side by
// side as lib/qairt-<ver>/ (e.g. qairt-2.45, qairt-2.47, qairt-2.48). No QNN
// runtime libraries are bundled; the user supplies them via GENIEX_QNN_LIB. Each
// variant is compiled against a specific QNN header set and is ABI-compatible
// only with matching libraries, so exactly one variant must be loaded — loading a
// mismatched one crashes at HTP init. selectQairtVariant() reads the QNN version
// from GENIEX_QNN_LIB and returns the qairt-<ver> directory name that should load;
// all other qairt-* directories are skipped.

// Reads GENIEX_QNN_LIB (a QAIRT SDK root or lib folder) and returns its
// QNN_API_VERSION_MINOR, or -1 if it cannot be determined.
int detectQnnApiMinor(const std::filesystem::path& qnn_lib_root) {
    namespace fs = std::filesystem;
    std::error_code ec;
    // QnnCommon.h lives at <sdk-root>/include/QNN/QnnCommon.h in a full QAIRT SDK.
    // Also probe a couple of shallower spots in case a trimmed folder was passed.
    const fs::path candidates[] = {
        qnn_lib_root / "include" / "QNN" / "QnnCommon.h",
        qnn_lib_root / "include" / "QnnCommon.h",
        qnn_lib_root / "QnnCommon.h",
    };
    for (const auto& hdr : candidates) {
        if (!fs::exists(hdr, ec)) continue;
        std::ifstream f(hdr);
        std::string   line;
        const std::regex re(R"(#define\s+QNN_API_VERSION_MINOR\s+(\d+))");
        while (std::getline(f, line)) {
            std::smatch m;
            if (std::regex_search(line, m, re)) {
                return std::stoi(m[1].str());
            }
        }
    }
    return -1;
}

// Maps a QNN API minor version onto the qairt plugin variant that targets it.
// Compatibility windows were established empirically on Snapdragon (Hexagon v73):
// a plugin compiled against minor N runs that minor and, for 2.45, the next one.
//   minor <= 35 (QAIRT <= 2.46) -> qairt-2.45
//   minor == 36 (QAIRT 2.47)    -> qairt-2.47
//   minor >= 37 (QAIRT >= 2.48) -> qairt-2.48
std::string qairtVariantForMinor(int minor) {
    if (minor < 0) return "";
    if (minor <= 35) return "qairt-2.45";
    if (minor == 36) return "qairt-2.47";
    return "qairt-2.48";
}

// Returns the single qairt-<ver> directory name that should be loaded, based on
// GENIEX_QNN_LIB. Empty string means "no QAIRT version selected" (either the env
// var is unset or the version could not be read) — in that case no qairt-* variant
// is loaded and the caller surfaces a clear error if a qairt model is requested.
std::string selectQairtVariant() {
    std::filesystem::path qnn_lib_root;
#if defined(_WIN32)
    size_t required_size = 0;
    _wgetenv_s(&required_size, nullptr, 0, L"GENIEX_QNN_LIB");
    if (required_size > 0) {
        std::vector<wchar_t> buf(required_size);
        _wgetenv_s(&required_size, buf.data(), required_size, L"GENIEX_QNN_LIB");
        if (buf[0] != L'\0') qnn_lib_root = std::filesystem::path(buf.data());
    }
#else
    if (const char* v = std::getenv("GENIEX_QNN_LIB")) qnn_lib_root = std::filesystem::path(v);
#endif
    if (qnn_lib_root.empty()) return "";

    const int minor = detectQnnApiMinor(qnn_lib_root);
    const std::string variant = qairtVariantForMinor(minor);
    if (variant.empty()) {
        GENIEX_LOG_WARN("GENIEX_QNN_LIB set but QNN_API_VERSION_MINOR not found under {}; "
                        "no qairt variant selected", qnn_lib_root.u8string());
    } else {
        GENIEX_LOG_INFO("GENIEX_QNN_LIB QNN API minor {} -> loading plugin variant {}", minor, variant);
    }
    return variant;
}

}  // namespace

namespace geniex {
PluginFactory::PluginFactory(const std::filesystem::path& path) {
    GENIEX_LOG_TRACE("loading plugin from {}", path.u8string());
    load_library(path);
    void* sym = nullptr;

    load_symbol("create_plugin", sym);
    create_func = reinterpret_cast<Plugin* (*)()>(sym);

    load_symbol("plugin_id", sym);
    auto plugin_id_func = reinterpret_cast<const char* (*)()>(sym);
    if (plugin_id_func) {
        const char* id = plugin_id_func();
        if (id) {
            plugin_id = std::string(id);
            GENIEX_LOG_TRACE("plugin id: {}", plugin_id);
        } else {
            throw std::runtime_error("plugin_id function returned null");
        }
    } else {
        throw std::runtime_error("Failed to load plugin_id symbol");
    }
}

PluginFactory::PluginFactory(void* plugin_id_func, void* create_func) {
    GENIEX_LOG_TRACE("creating plugin from ptr: {}, {}", plugin_id_func, create_func);

    this->create_func = reinterpret_cast<Plugin* (*)()>(create_func);
    auto plugin_id_fn = reinterpret_cast<const char* (*)()>(plugin_id_func);
    if (plugin_id_fn) {
        const char* id = plugin_id_fn();
        if (id) {
            this->plugin_id = std::string(id);
            GENIEX_LOG_TRACE("plugin id: {}", this->plugin_id);
        } else {
            throw std::runtime_error("plugin_id function returned null");
        }
    } else {
        throw std::runtime_error("Failed to load plugin_id symbol");
    }
}

PluginFactory::~PluginFactory() {
    GENIEX_LOG_TRACE("destroying plugin {}", this->plugin_id);
    cached_plugin.reset();
    close_library();
}

Plugin* PluginFactory::get_instance() {
    if (!create_func) {
        throw std::runtime_error("Plugin factory not initialized");
    }

    // if not cached, create a new one
    if (!cached_plugin) {
        Plugin* raw_plugin = create_func();
        if (!raw_plugin) {
            throw std::runtime_error("Plugin factory create_func returned null");
        }
        cached_plugin.reset(raw_plugin);
    }

    return cached_plugin.get();
}

// PluginFactory private

void PluginFactory::load_library(const std::filesystem::path& path) {
#if defined(_WIN32)
    std::filesystem::path abs_path = std::filesystem::absolute(path);

    // Use modern DLL search flags that respect AddDllDirectory
    DWORD flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |     // Plugin's own directory
                  LOAD_LIBRARY_SEARCH_APPLICATION_DIR |  // Application directory
                  LOAD_LIBRARY_SEARCH_USER_DIRS |        // Directories added via AddDllDirectory
                  LOAD_LIBRARY_SEARCH_SYSTEM32;          // System directory

    handle = LoadLibraryExW(abs_path.wstring().c_str(), NULL, flags);
    if (!handle) {
        throw std::runtime_error("LoadLibraryExW failed: " + std::to_string(GetLastError()));
    }
#else
    handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
    }
#endif
}

void PluginFactory::load_symbol(const std::string& symbol_name, void*& symbol_ptr) {
#if defined(_WIN32)
    symbol_ptr = (void*)GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol_name.c_str());
#else
    symbol_ptr = dlsym(handle, symbol_name.c_str());
#endif
}

void PluginFactory::close_library() {
    if (cached_plugin) {
        cached_plugin.reset();
    }

    if (handle) {
#if defined(_WIN32)
        FreeLibrary(reinterpret_cast<HMODULE>(handle));
#else
        dlclose(handle);
#endif
    }
}

void Registry::scan_plugins() {
    std::filesystem::path plugin_path;

#if defined(_WIN32)
    // On Windows, use wide string API to properly handle Unicode paths
    size_t required_size = 0;
    _wgetenv_s(&required_size, nullptr, 0, L"GENIEX_PLUGIN_PATH");
    if (required_size > 0) {
        std::vector<wchar_t> env_buffer(required_size);
        _wgetenv_s(&required_size, env_buffer.data(), required_size, L"GENIEX_PLUGIN_PATH");
        if (env_buffer[0] != L'\0') {
            plugin_path = std::filesystem::path(env_buffer.data());
            GENIEX_LOG_TRACE("Using GENIEX_PLUGIN_PATH environment variable: {}", plugin_path.u8string());
        }
    }
#else
    auto env_plugin_path = std::getenv("GENIEX_PLUGIN_PATH");
    if (env_plugin_path) {
        plugin_path = std::filesystem::path(env_plugin_path);
        GENIEX_LOG_TRACE("Using GENIEX_PLUGIN_PATH environment variable: {}", plugin_path.u8string());
    }
#endif

    if (plugin_path.empty()) {
        plugin_path = get_shared_lib_dir();
        GENIEX_LOG_TRACE("Using shared lib directory for plugin path: {}", plugin_path.u8string());
#if defined(_WIN32)
        _wputenv_s(L"GENIEX_PLUGIN_PATH", plugin_path.wstring().c_str());
#else
        setenv("GENIEX_PLUGIN_PATH", plugin_path.c_str(), 1);
#endif
    }

    GENIEX_LOG_TRACE("Scanning plugins in: {}", plugin_path.u8string());

    // The qairt plugin ships as several ABI variants (qairt-2.45, qairt-2.47, …),
    // one per QAIRT/QNN version, that all register the same plugin_id. Exactly one
    // may load — pick the one matching the QNN libs named by GENIEX_QNN_LIB and
    // skip the rest. Empty => no qairt libs specified, so skip all qairt variants.
    const std::string selected_qairt = selectQairtVariant();

    // Search child plugin directories for the brand-specific plugin shared library.
    for (const auto& dir_entry : std::filesystem::directory_iterator(plugin_path)) {
        if (!dir_entry.is_directory()) continue;

        const std::string dir_name = dir_entry.path().filename().u8string();
        if (dir_name.rfind("qairt-", 0) == 0 && dir_name != selected_qairt) {
            GENIEX_LOG_TRACE("Skipping qairt variant {} (selected: '{}')", dir_name, selected_qairt);
            continue;
        }

        GENIEX_LOG_TRACE("Scanning directory: {}", dir_entry.path().u8string());

        for (const auto& file_entry : std::filesystem::directory_iterator(dir_entry.path())) {
            if (file_entry.is_regular_file() && file_entry.path().filename().u8string() == get_geniex_plugin_name()) {
                try {
                    auto        plugin    = std::make_unique<PluginFactory>(file_entry.path());
                    std::string plugin_id = plugin->get_plugin_id();
                    plugins.emplace(plugin_id, std::move(plugin));
                    GENIEX_LOG_TRACE("Registered plugin: {}", plugin_id);
                } catch (const std::exception& ex) {
                    // record failed plugin load
                    try {
                        failed_plugins.push_back(file_entry.path().parent_path().filename().u8string());
                    } catch (...) {
                        GENIEX_LOG_WARN("Failed to record failed plugin for {}", file_entry.path().u8string());
                    }
                    GENIEX_LOG_ERROR("Failed to load plugin {}: {}", file_entry.path().u8string(), ex.what());
                }
            }
        }
    }
}

void Registry::register_plugin(void* plugin_id_func, void* create_func) {
    GENIEX_LOG_TRACE("Registering plugin: {}, {}", plugin_id_func, create_func);
    std::lock_guard<std::mutex> lock(mutex);
    auto                        plugin = std::make_unique<PluginFactory>(plugin_id_func, create_func);
    std::string                 id     = plugin->get_plugin_id();
    plugins.emplace(id, std::move(plugin));
    GENIEX_LOG_TRACE("Registered plugin: {}", id);
}

Registry& Registry::instance() {
    static Registry instance;
    return instance;
}

void Registry::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    GENIEX_LOG_TRACE("Clearing registry - destroying {} plugins", plugins.size());
    plugins.clear();
    GENIEX_LOG_TRACE("Registry cleared successfully");
}

}  // namespace geniex
