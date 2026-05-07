#include <stdlib.h>

#include "geniex.h"

#if defined(_WIN32)
#define portable_strdup _strdup
#else
#define portable_strdup strdup
#endif

// keep geniex_plugin link openssl
#ifdef GENIEX_VALIDATION
#include "openssl/crypto.h"
#include "openssl/ssl.h"
void* _ssl_dummy    = (void*)SSL_CTX_get_options;
void* _crypto_dummy = (void*)OpenSSL_version;
#endif

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "build_config.h"
#include "logging.h"
#include "registry.h"
#include "utils.h"

#ifdef _WIN32
#include <windows.h>
#endif

using namespace geniex;

namespace {

// Sentinel value representing "no logging" (higher than any real level).
// Not exposed in the public enum to keep ABI stable.
constexpr geniex_LogLevel kLogLevelNone = static_cast<geniex_LogLevel>(GENIEX_LOG_LEVEL_ERROR + 1);

bool parse_log_level(const char* s, geniex_LogLevel& out) {
    if (s == nullptr) return false;
    if (std::strcmp(s, "trace") == 0) {
        out = GENIEX_LOG_LEVEL_TRACE;
        return true;
    }
    if (std::strcmp(s, "debug") == 0) {
        out = GENIEX_LOG_LEVEL_DEBUG;
        return true;
    }
    if (std::strcmp(s, "info") == 0) {
        out = GENIEX_LOG_LEVEL_INFO;
        return true;
    }
    if (std::strcmp(s, "warn") == 0) {
        out = GENIEX_LOG_LEVEL_WARN;
        return true;
    }
    if (std::strcmp(s, "error") == 0) {
        out = GENIEX_LOG_LEVEL_ERROR;
        return true;
    }
    if (std::strcmp(s, "none") == 0) {
        out = kLogLevelNone;
        return true;
    }
    return false;
}

bool use_color() {
#ifdef _WIN32
    return false;
#else
    const char* no_color = std::getenv("NO_COLOR");
    return !(no_color != nullptr && no_color[0] != '\0');
#endif
}

}  // namespace

// Default log handler — always compiled; honors the runtime level threshold.
// Emits to stderr with optional ANSI coloring (disabled when NO_COLOR is set).
static void default_log_handler(geniex_LogLevel level, const char* msg) {
    if (level < geniex_log_level) return;
    const bool  color = use_color();
    const char* prefix;
    const char* colorCode;
    switch (level) {
        case GENIEX_LOG_LEVEL_TRACE:
            prefix    = "[TRACE] ";
            colorCode = "\033[90m";
            break;
        case GENIEX_LOG_LEVEL_DEBUG:
            prefix    = "[DEBUG] ";
            colorCode = "\033[34m";
            break;
        case GENIEX_LOG_LEVEL_INFO:
            prefix    = "[ INFO] ";
            colorCode = "\033[32m";
            break;
        case GENIEX_LOG_LEVEL_WARN:
            prefix    = "[ WARN] ";
            colorCode = "\033[33m";
            break;
        case GENIEX_LOG_LEVEL_ERROR:
            prefix    = "[ERROR] ";
            colorCode = "\033[31m";
            break;
        default:
            return;
    }
    if (color) {
        std::cerr << colorCode << prefix << msg << "\033[0m" << std::endl;
    } else {
        std::cerr << prefix << msg << std::endl;
    }
}

// Tracks whether geniex_set_log_level() was called explicitly, so geniex_init() only
// falls back to GENIEX_LOG env var when the embedder hasn't picked a level already.
static bool s_log_level_user_set = false;

int32_t geniex_init(void) {
#ifdef _WIN32
    // set console output to UTF-8 code page for Windows
    SetConsoleOutputCP(CP_UTF8);
#endif

    if (!s_log_level_user_set) {
        geniex_LogLevel parsed;
        if (parse_log_level(std::getenv("GENIEX_LOG"), parsed)) {
            geniex_log_level = parsed;
        }
    }

    GENIEX_LOG_INFO("initializing ml");

    try {
        Registry::instance().scan_plugins();
        return GENIEX_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to initialize ml: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_register_plugin(geniex_plugin_id_func plugin_id_func, geniex_create_plugin_func create_func) {
    GENIEX_LOG_INFO("register plugin");

    try {
        void* plugin_id     = (void*)plugin_id_func;
        void* create_plugin = (void*)create_func;
        Registry::instance().register_plugin(plugin_id, create_plugin);
        return GENIEX_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to register plugin: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_deinit(void) {
    GENIEX_LOG_INFO("deinitializing ml");

    try {
        // Clean up the registry to ensure proper plugin destruction
        geniex::Registry::instance().clear();
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("geniex_deinit() - Error during registry cleanup: {}", e.what());
    }

    return GENIEX_SUCCESS;
}

// Logging

geniex_log_callback geniex_log = default_log_handler;

// Default threshold: DEBUG in GENIEX_DEBUG builds, INFO otherwise.
// Runtime override via geniex_set_log_level() or GENIEX_LOG env var (read in geniex_init).
#ifdef GENIEX_DEBUG
geniex_LogLevel geniex_log_level = GENIEX_LOG_LEVEL_DEBUG;
#else
geniex_LogLevel geniex_log_level = GENIEX_LOG_LEVEL_INFO;
#endif

int32_t geniex_set_log(geniex_log_callback callback) {
    geniex_log = callback;
    return GENIEX_SUCCESS;
}

int32_t geniex_set_log_level(geniex_LogLevel level) {
    geniex_log_level     = level;
    s_log_level_user_set = true;
    return GENIEX_SUCCESS;
}

geniex_LogLevel geniex_get_log_level(void) { return geniex_log_level; }

void geniex_free(void* ptr) {
    if (ptr) free(ptr);
}

// Version

const char* version = build_config::kBridgeVersion;

const char* geniex_version() { return version; }

// Get Plugin List

int32_t geniex_get_plugin_list(geniex_GetPluginListOutput* output) {
    GENIEX_LOG_TRACE("getting plugin list: {}", output);
    if (!output) {
        GENIEX_LOG_ERROR("output is nullptr");
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    try {
        auto plugin_list = Registry::instance().get_plugin_list();
        if (plugin_list.empty()) {
            output->plugin_ids   = nullptr;
            output->plugin_count = 0;
            return GENIEX_SUCCESS;
        }

        output->plugin_ids = static_cast<geniex_PluginId*>(malloc(plugin_list.size() * sizeof(geniex_PluginId)));
        if (!output->plugin_ids) {
            GENIEX_LOG_ERROR("failed to allocate memory for plugin IDs");
            return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
        }
        output->plugin_count = static_cast<int32_t>(plugin_list.size());

        for (int32_t i = 0; i < output->plugin_count; i++) {
            output->plugin_ids[i] = portable_strdup(plugin_list[i].c_str());
            if (!output->plugin_ids[i]) {
                GENIEX_LOG_ERROR("failed to duplicate plugin ID at index {}", i);
                for (int32_t j = 0; j < i; j++) {
                    std::free(const_cast<char*>(output->plugin_ids[j]));
                }
                std::free(output->plugin_ids);
                output->plugin_ids   = nullptr;
                output->plugin_count = 0;
                return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
            }
        }
        return GENIEX_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to get plugin list: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

// Get Device List

int32_t geniex_get_device_list(const geniex_GetDeviceListInput* input, geniex_GetDeviceListOutput* output) {
    GENIEX_LOG_TRACE("getting device list: {}", input);
    if (!input || !input->plugin_id || !output) {
        GENIEX_LOG_ERROR("input or input->plugin_id or output is nullptr");
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    try {
        auto plugin = Registry::instance().get<Plugin>(input->plugin_id);
        if (plugin) {
            return plugin->get_device_list(input, output);
        } else {
            GENIEX_LOG_ERROR("failed to get device list for plugin: {}", input->plugin_id);
            return GENIEX_ERROR_COMMON_UNKNOWN;
        }
        return GENIEX_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to get device list: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}
