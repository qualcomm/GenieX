#include "ml.h"

#include <stdlib.h>

// keep geniex_plugin link openssl
#ifdef GENIEX_VALIDATION
#include "openssl/crypto.h"
#include "openssl/ssl.h"
void* _ssl_dummy    = (void*)SSL_CTX_get_options;
void* _crypto_dummy = (void*)OpenSSL_version;
#endif

#include <cstdlib>
#ifdef GENIEX_DEBUG
#include <iostream>
#endif

#include "build_config.h"
#include "logging.h"
#include "registry.h"
#include "utils.h"

#ifdef _WIN32
#include <windows.h>
#endif

using namespace geniex;

#ifdef _WIN32
static void setup_windows_dll_search_path() {
    auto core_dir = get_shared_lib_dir();

    auto lib_dir = core_dir / build_config::kCommonLibRelativePath;
    if (!std::filesystem::exists(lib_dir)) {
        GENIEX_LOG_WARN(
            "{} subdirectory not found, skip adding DLL search directory; some runtime dependencies may not be found",
            build_config::kCommonLibRelativePath);
        return;
    }

    auto cookie = AddDllDirectory(lib_dir.wstring().c_str());
    if (!cookie) {
        GENIEX_LOG_ERROR("Failed to add DLL directory (error {})", GetLastError());
    }

    GENIEX_LOG_DEBUG("Added DLL search directory: {}", lib_dir.u8string());
}
#endif

// Default log handler - colorized for debug builds, no-op for release builds
static void default_log_handler(ml_LogLevel level, const char* msg) {
#ifdef GENIEX_DEBUG
    switch (level) {
        case ML_LOG_LEVEL_TRACE:
            std::cerr << "\033[90m[TRACE] " << msg << "\033[0m" << std::endl;
            break;
        case ML_LOG_LEVEL_DEBUG:
            std::cerr << "\033[34m[DEBUG] " << msg << "\033[0m" << std::endl;
            break;
        case ML_LOG_LEVEL_INFO:
            std::cerr << "\033[32m[ INFO] " << msg << "\033[0m" << std::endl;
            break;
        case ML_LOG_LEVEL_WARN:
            std::cerr << "\033[33m[ WARN] " << msg << "\033[0m" << std::endl;
            break;
        case ML_LOG_LEVEL_ERROR:
            std::cerr << "\033[31m[ERROR] " << msg << "\033[0m" << std::endl;
            break;
    }
#else
    // No-op for release builds
    (void)level;
    (void)msg;
#endif
}

int32_t ml_init(void) {
#ifdef _WIN32
    // set console output to UTF-8 code page for Windows
    SetConsoleOutputCP(CP_UTF8);
#endif

    GENIEX_LOG_INFO("initializing ml");

    try {
#ifdef _WIN32
        setup_windows_dll_search_path();
#endif
#ifdef GENIEX_DL
        Registry::instance().scan_plugins();
#endif
        return ML_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to initialize ml: {}", e.what());
        return ML_ERROR_COMMON_UNKNOWN;
    }
}

int32_t ml_register_plugin(ml_plugin_id_func plugin_id_func, ml_create_plugin_func create_func) {
    GENIEX_LOG_INFO("register plugin");

    try {
        void* plugin_id     = (void*)plugin_id_func;
        void* create_plugin = (void*)create_func;
        Registry::instance().register_plugin(plugin_id, create_plugin);
        return ML_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to register plugin: {}", e.what());
        return ML_ERROR_COMMON_UNKNOWN;
    }
}

int32_t ml_deinit(void) {
    GENIEX_LOG_INFO("deinitializing ml");

    try {
        // Clean up the registry to ensure proper plugin destruction
        geniex::Registry::instance().clear();
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("ml_deinit() - Error during registry cleanup: {}", e.what());
    }

    return ML_SUCCESS;
}

// Logging

ml_log_callback ml_log = default_log_handler;

int32_t ml_set_log(ml_log_callback callback) {
    ml_log = callback;
    return ML_SUCCESS;
}

void ml_free(void* ptr) {
    if (ptr) free(ptr);
}

// Version

constexpr const char* version = build_config::kBridgeVersion;

const char* ml_version() { return version; }

// Get Plugin List

int32_t ml_get_plugin_list(ml_GetPluginListOutput* output) {
    GENIEX_LOG_TRACE("getting plugin list: {}", output);
    if (!output) {
        GENIEX_LOG_ERROR("output is nullptr");
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    try {
        auto plugin_list = Registry::instance().get_plugin_list();
        if (plugin_list.empty()) {
            output->plugin_ids   = nullptr;
            output->plugin_count = 0;
            return ML_SUCCESS;
        }

        output->plugin_ids = static_cast<ml_PluginId*>(malloc(plugin_list.size() * sizeof(ml_PluginId)));
        if (!output->plugin_ids) {
            GENIEX_LOG_ERROR("failed to allocate memory for plugin IDs");
            return ML_ERROR_COMMON_MEMORY_ALLOCATION;
        }
        output->plugin_count = static_cast<int32_t>(plugin_list.size());

        for (int32_t i = 0; i < output->plugin_count; i++) {
            output->plugin_ids[i] = strdup(plugin_list[i].c_str());
            if (!output->plugin_ids[i]) {
                GENIEX_LOG_ERROR("failed to duplicate plugin ID at index {}", i);
                for (int32_t j = 0; j < i; j++) {
                    std::free(const_cast<char*>(output->plugin_ids[j]));
                }
                std::free(output->plugin_ids);
                output->plugin_ids   = nullptr;
                output->plugin_count = 0;
                return ML_ERROR_COMMON_MEMORY_ALLOCATION;
            }
        }
        return ML_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to get plugin list: {}", e.what());
        return ML_ERROR_COMMON_UNKNOWN;
    }
}

// Get Device List

int32_t ml_get_device_list(const ml_GetDeviceListInput* input, ml_GetDeviceListOutput* output) {
    GENIEX_LOG_TRACE("getting device list: {}", input);
    if (!input || !input->plugin_id || !output) {
        GENIEX_LOG_ERROR("input or input->plugin_id or output is nullptr");
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    try {
        auto plugin = Registry::instance().get<Plugin>(input->plugin_id);
        if (plugin) {
            return plugin->get_device_list(input, output);
        } else {
            GENIEX_LOG_ERROR("failed to get device list for plugin: {}", input->plugin_id);
            return ML_ERROR_COMMON_UNKNOWN;
        }
        return ML_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to get device list: {}", e.what());
        return ML_ERROR_COMMON_UNKNOWN;
    }
}
