#include "plugin/Plugin.h"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

#include "build_config.h"
#include "embedding.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llm.h"
#include "logging.h"
#include "mtmd-helper.h"
#include "plugin/Plugin.h"
#include "rerank.h"
#include "vlm.h"

namespace {

void ggml_to_geniex_log(ggml_log_level l, const char* t, void*) {
    geniex_LogLevel level;
    switch (l) {
        case GGML_LOG_LEVEL_DEBUG:
            level = GENIEX_LOG_LEVEL_TRACE;
            break;
        case GGML_LOG_LEVEL_INFO:
            level = GENIEX_LOG_LEVEL_DEBUG;
            break;
        case GGML_LOG_LEVEL_WARN:
            level = GENIEX_LOG_LEVEL_INFO;
            break;
        case GGML_LOG_LEVEL_ERROR:
            level = GENIEX_LOG_LEVEL_WARN;
            break;
        case GGML_LOG_LEVEL_CONT:  // drop CONT level
        default:
            return;
    }
    std::string_view s = t ? t : "";
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.remove_suffix(1);
    if (s.empty()) return;
    GENIEX_LEVEL_LOG(level, "{}", s);
}

}  // namespace

namespace geniex {

class LlamaPlugin : public Plugin {
   public:
    LlamaPlugin() {
        GENIEX_LOG_TRACE("creating and initializing llama plugin");
        llama_log_set(ggml_to_geniex_log, nullptr);
        mtmd_helper_log_set(ggml_to_geniex_log, nullptr);

        std::filesystem::path backend_dir;
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
            backend_dir = std::filesystem::path(env_plugin_path);
        }
#endif  // _WIN32

#if not defined(__ANDROID__)  // android has flattened directory
        if (!backend_dir.empty()) {
            backend_dir = backend_dir / "llama_cpp";
        }
#endif  // not __ANDROID__

        GENIEX_LOG_DEBUG("Setting ADSP_LIBRARY_PATH to {}", backend_dir.u8string());
#if defined(WIN32)
        _putenv_s("ADSP_LIBRARY_PATH", backend_dir.u8string().c_str());
#else
        setenv("ADSP_LIBRARY_PATH", backend_dir.u8string().c_str(), 1);
#endif

        if (!backend_dir.empty()) {
#if defined(_WIN32)
            // LoadLibrary() does not reliably resolve transitive DLL dependencies
            // from the DLL's own directory in this Bazel runfiles layout.
            if (!SetDllDirectoryW(backend_dir.wstring().c_str())) {
                GENIEX_LOG_WARN("Failed to set DLL search directory to {}", backend_dir.u8string());
            }
#endif  // _WIN32
            auto path = backend_dir.u8string();
            GENIEX_LOG_DEBUG("Loading GGML backend from path: {}", path);
            ggml_backend_load_all_from_path(path.c_str());
        }

        // #ifdef __ANDROID__
        // Harcoding to use 4 hexagon devices for Hexagon to make GPT-OSS work
        //         setenv("GGML_HEXAGON_NDEV", "4", 1);
        //         GENIEX_LOG_DEBUG("Set GGML_HEXAGON_NDEV=4 for Hexagon backend");
        // #endif  // __ANDROID__

        llama_backend_init();
    }

    ~LlamaPlugin() override { GENIEX_LOG_TRACE("destroying llama plugin"); }

    int32_t get_device_list(const geniex_GetDeviceListInput* input, geniex_GetDeviceListOutput* output) override {
        GENIEX_LOG_TRACE("getting device list");
        if (!input || !output) {
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        auto count = ggml_backend_dev_count();
        auto ids   = (const char**)malloc(count * sizeof(const char*));
        auto names = (const char**)malloc(count * sizeof(const char*));
        if (!ids || !names) {
            return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
        }

        for (size_t i = 0; i < count; ++i) {
            auto* dev = ggml_backend_dev_get(i);
            ids[i]    = ggml_backend_dev_name(dev);
            names[i]  = ggml_backend_dev_description(dev);
        }
        GENIEX_LOG_TRACE("device count: {}", count);
        output->device_ids   = ids;
        output->device_names = names;
        output->device_count = static_cast<int32_t>(count);
        return GENIEX_SUCCESS;
    }

    ILlm* create_llm() override { return new geniex::LlamaLlm; }

    IVlm* create_vlm() override { return new geniex::LlamaVlm; }

    IEmbedding* create_embedding() override { return new geniex::LlamaCppEmbedding; }

    IReranker* create_reranker() override { return new geniex::LlamaCppReranker; }
};

}  // namespace geniex

#ifdef GENIEX_STATIC

#else

geniex_PluginId plugin_id() { return geniex::build_config::kPluginIdLlamaCpp; }

geniex::Plugin* create_plugin() {
    try {
        GENIEX_LOG_TRACE("creating llama plugin");
        return new geniex::LlamaPlugin;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to create llama plugin: {}", e.what());
        return nullptr;
    }
}

#endif
