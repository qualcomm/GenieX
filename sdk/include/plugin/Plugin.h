#pragma once

#include "IAsr.h"
#include "ICv.h"
#include "IDiarize.h"
#include "IEmbedding.h"
#include "IImageGen.h"
#include "ILlm.h"
#include "IReranker.h"
#include "ITts.h"
#include "IVlm.h"
#include "geniex.h"
#include "logging.h"

namespace geniex {

class Plugin {
   public:
    virtual ~Plugin() = default;
    virtual const char* version() { return "unknown"; }
    virtual int32_t     get_device_list(const geniex_GetDeviceListInput*, geniex_GetDeviceListOutput* output) {
        if (!output) {
            GENIEX_LOG_ERROR("output is nullptr");
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
        output->device_ids   = nullptr;
        output->device_names = nullptr;
        output->device_count = 0;
        return GENIEX_SUCCESS;
    }
    virtual ILlm*       create_llm() { return nullptr; }
    virtual IVlm*       create_vlm() { return nullptr; }
    virtual IEmbedding* create_embedding() { return nullptr; }
    virtual IReranker*  create_reranker() { return nullptr; }
    virtual IAsr*       create_asr() { return nullptr; }
    virtual ITts*       create_tts() { return nullptr; }
    virtual ICv*        create_cv() { return nullptr; }
    virtual IImageGen*  create_image_gen() { return nullptr; }
    virtual IDiarize*   create_diarize() { return nullptr; }
};

}  // namespace geniex

#if defined(_WIN32)
#define PLUGIN_API __declspec(dllexport)
#else
#define PLUGIN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

PLUGIN_API geniex_PluginId plugin_id();
PLUGIN_API geniex::Plugin* create_plugin();

#ifdef __cplusplus
}
#endif
