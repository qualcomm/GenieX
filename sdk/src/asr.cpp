#include <profile.h>

#include <cstring>

#include "geniex.h"
#include "logging.h"
#include "plugin/IAsr.h"
#include "registry.h"

using namespace geniex;

int32_t geniex_asr_create(const geniex_AsrCreateInput* input, geniex_ASR** out_handle) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = geniex::Registry::instance().get<geniex::IAsr>(input->plugin_id);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_SUPPORTED;
        int32_t res = backend->create(input);
        if (res != GENIEX_SUCCESS) {
            delete backend;
        } else {
            *out_handle = reinterpret_cast<geniex_ASR*>(backend);
        }
        return res;
    } catch (const PluginNotFoundException& e) {
        GENIEX_LOG_ERROR("plugin not found");
        return GENIEX_ERROR_COMMON_PLUGIN_INVALID;
    } catch (const PluginLoadException& e) {
        GENIEX_LOG_ERROR("plugin load error");
        return GENIEX_ERROR_COMMON_PLUGIN_LOAD;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("creating asr error: {}", e.what());
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
}

int32_t geniex_asr_destroy(geniex_ASR* h) {
    GENIEX_LOG_TRACE("asr destroy");

    try {
        auto backend = reinterpret_cast<IAsr*>(h);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        delete backend;
        return GENIEX_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("destroy asr error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_asr_transcribe(
    geniex_ASR* h, const geniex_AsrTranscribeInput* input, geniex_AsrTranscribeOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<IAsr*>(h);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;

        auto result = backend->transcribe(input, output);
        calculate_profile_data(output->profile_data);
        GENIEX_LOG_TRACE("{}: {}", static_cast<geniex_ErrorCode>(result), output);
        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("asr transcribe error: {}", e.what());
        return GENIEX_ERROR_ASR_TRANSCRIPTION;
    }
}

int32_t geniex_asr_list_supported_languages(const geniex_ASR* h, const geniex_AsrListSupportedLanguagesInput* input,
    geniex_AsrListSupportedLanguagesOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<IAsr*>(const_cast<geniex_ASR*>(h));
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        auto result = backend->list_supported_languages(input, output);
        GENIEX_LOG_TRACE("{}: {}", static_cast<geniex_ErrorCode>(result), output);
        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("asr list supported languages error: {}", e.what());
        return GENIEX_ERROR_ASR_LANGUAGE;
    }
}

// =============================================================================
// Streaming ASR C API Implementation
// =============================================================================

int32_t geniex_asr_stream_begin(
    geniex_ASR* handle, const geniex_AsrStreamBeginInput* input, geniex_AsrStreamBeginOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        if (!handle) {
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        auto* asr    = reinterpret_cast<IAsr*>(handle);
        auto  result = asr->stream_begin(input, output);
        GENIEX_LOG_TRACE("{}: streaming began", static_cast<geniex_ErrorCode>(result));
        return result;

    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("asr stream begin error: {}", e.what());
        return GENIEX_ERROR_ASR_STREAM_NOT_STARTED;
    }
}

int32_t geniex_asr_stream_push_audio(geniex_ASR* handle, const geniex_AsrStreamPushAudioInput* input) {
    // Don't trace this - too verbose for audio data
    try {
        if (!handle) {
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        auto* asr = reinterpret_cast<IAsr*>(handle);
        return asr->stream_push_audio(input);

    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("asr stream push audio error: {}", e.what());
        return GENIEX_ERROR_ASR_STREAM_INVALID_AUDIO;
    }
}

int32_t geniex_asr_stream_stop(geniex_ASR* handle, const geniex_AsrStreamStopInput* input) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        if (!handle) {
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        auto* asr    = reinterpret_cast<IAsr*>(handle);
        auto  result = asr->stream_stop(input);
        GENIEX_LOG_TRACE("{}: streaming stopped", static_cast<geniex_ErrorCode>(result));
        return result;

    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("asr stream stop error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}
