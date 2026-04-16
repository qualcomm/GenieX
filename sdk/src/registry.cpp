

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
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

    // Search child plugin directories for the brand-specific plugin shared library.
    for (const auto& dir_entry : std::filesystem::directory_iterator(plugin_path)) {
        if (!dir_entry.is_directory()) continue;
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
