#include <profile.h>

#include <cstring>

#include "geniex.h"
#include "logging.h"
#include "plugin/ITts.h"
#include "registry.h"

using namespace geniex;

int32_t geniex_tts_create(const geniex_TtsCreateInput* input, geniex_TTS** out_handle) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = geniex::Registry::instance().get<geniex::ITts>(input->plugin_id);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_SUPPORTED;
        int32_t res = backend->create(input);
        if (res != GENIEX_SUCCESS) {
            delete backend;
        } else {
            *out_handle = reinterpret_cast<geniex_TTS*>(backend);
        }
        return res;
    } catch (const PluginNotFoundException& e) {
        GENIEX_LOG_ERROR("plugin not found");
        return GENIEX_ERROR_COMMON_PLUGIN_INVALID;
    } catch (const PluginLoadException& e) {
        GENIEX_LOG_ERROR("plugin load error");
        return GENIEX_ERROR_COMMON_PLUGIN_LOAD;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("creating tts error: {}", e.what());
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
}

int32_t geniex_tts_destroy(geniex_TTS* h) {
    GENIEX_LOG_TRACE("destroying tts");

    try {
        auto backend = reinterpret_cast<ITts*>(h);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        delete backend;
        return GENIEX_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("destroy tts error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_tts_synthesize(
    geniex_TTS* h, const geniex_TtsSynthesizeInput* input, geniex_TtsSynthesizeOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<ITts*>(h);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        auto result = backend->synthesize(input, output);
        calculate_profile_data(output->profile_data);
        GENIEX_LOG_TRACE("{}: {}", static_cast<geniex_ErrorCode>(result), output);
        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("tts synthesize error: {}", e.what());
        return GENIEX_ERROR_TTS_SYNTHESIS;
    }
}

int32_t geniex_tts_list_available_voices(
    const geniex_TTS* h, const geniex_TtsListAvailableVoicesInput* input, geniex_TtsListAvailableVoicesOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<ITts*>(const_cast<geniex_TTS*>(h));
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        int32_t result = backend->list_available_voices(input, output);
        GENIEX_LOG_TRACE("{}: {}", static_cast<geniex_ErrorCode>(result), output);
        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("tts list available voices error: {}", e.what());
        return GENIEX_ERROR_TTS_VOICE;
    }
}
