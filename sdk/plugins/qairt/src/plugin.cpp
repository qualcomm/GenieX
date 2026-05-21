#include "plugin/Plugin.h"

#include <cstdlib>
#include <exception>

#include "build_config.h"
#include "llm.h"
#include "logging.h"
#include "version.h"  // GENIEX_QAIRT_VERSION from third-party/geniex-qairt/core/include
#include "vlm.h"

namespace geniex {

class QairtPlugin : public Plugin {
   public:
    QairtPlugin() { GENIEX_LOG_TRACE("creating and initializing qairt plugin"); }

    ~QairtPlugin() override { GENIEX_LOG_TRACE("destroying qairt plugin"); }

    const char* version() override { return GENIEX_QAIRT_VERSION; }

    int32_t get_device_list(const geniex_GetDeviceListInput* input, geniex_GetDeviceListOutput* output) override {
        if (!input || !output) {
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        auto ids   = (const char**)malloc(sizeof(const char*));
        auto names = (const char**)malloc(sizeof(const char*));
        if (!ids || !names) {
            free(ids);
            free(names);
            return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
        }
        ids[0]   = "NPU";
        names[0] = "Qualcomm NPU (QAIRT)";

        output->device_ids   = ids;
        output->device_names = names;
        output->device_count = 1;
        return GENIEX_SUCCESS;
    }

    ILlm* create_llm() override { return new geniex::QairtLlm; }

    IVlm* create_vlm() override { return new geniex::QairtVlm; }

    IEmbedding* create_embedding() override {
        GENIEX_LOG_WARN("QAIRT embedding is not available in this build");
        return nullptr;
    }

    IReranker* create_reranker() override {
        GENIEX_LOG_WARN("QAIRT reranker is not available in this build");
        return nullptr;
    }

    IAsr* create_asr() override {
        GENIEX_LOG_WARN("QAIRT ASR is not available in this build");
        return nullptr;
    }

    ITts* create_tts() override {
        GENIEX_LOG_WARN("QAIRT TTS is not available in this build");
        return nullptr;
    }

    ICv* create_cv() override {
        GENIEX_LOG_WARN("QAIRT CV is not available in this build");
        return nullptr;
    }

    IImageGen* create_image_gen() override {
        GENIEX_LOG_WARN("QAIRT image-gen is not available in this build");
        return nullptr;
    }

    IDiarize* create_diarize() override {
        GENIEX_LOG_WARN("QAIRT diarize is not available in this build");
        return nullptr;
    }
};

}  // namespace geniex

#ifdef GENIEX_STATIC

#else

geniex_PluginId plugin_id() { return geniex::build_config::kPluginIdQairt; }

geniex::Plugin* create_plugin() {
    try {
        GENIEX_LOG_TRACE("creating qairt plugin");
        return new geniex::QairtPlugin;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to create qairt plugin: {}", e.what());
        return nullptr;
    }
}

#endif
