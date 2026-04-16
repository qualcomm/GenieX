#include <profile.h>

#include <cstring>

#include "logging.h"
#include "ml.h"
#include "plugin/IAsr.h"
#include "registry.h"

using namespace geniex;

int32_t ml_asr_create(const ml_AsrCreateInput* input, ml_ASR** out_handle) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = geniex::Registry::instance().get<geniex::IAsr>(input->plugin_id);
        if (!backend) return ML_ERROR_COMMON_NOT_SUPPORTED;
        int32_t res = backend->create(input);
        if (res != ML_SUCCESS) {
            delete backend;
        } else {
            *out_handle = reinterpret_cast<ml_ASR*>(backend);
        }
        return res;
    } catch (const PluginNotFoundException& e) {
        GENIEX_LOG_ERROR("plugin not found");
        return ML_ERROR_COMMON_PLUGIN_INVALID;
    } catch (const PluginLoadException& e) {
        GENIEX_LOG_ERROR("plugin load error");
        return ML_ERROR_COMMON_PLUGIN_LOAD;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("creating asr error: {}", e.what());
        return ML_ERROR_COMMON_MODEL_LOAD;
    }
}

int32_t ml_asr_destroy(ml_ASR* h) {
    GENIEX_LOG_TRACE("asr destroy");

    try {
        auto backend = reinterpret_cast<IAsr*>(h);
        if (!backend) return ML_ERROR_COMMON_NOT_INITIALIZED;
        delete backend;
        return ML_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("destroy asr error: {}", e.what());
        return ML_ERROR_COMMON_UNKNOWN;
    }
}

int32_t ml_asr_transcribe(ml_ASR* h, const ml_AsrTranscribeInput* input, ml_AsrTranscribeOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<IAsr*>(h);
        if (!backend) return ML_ERROR_COMMON_NOT_INITIALIZED;

        auto result = backend->transcribe(input, output);
        calculate_profile_data(output->profile_data);
        GENIEX_LOG_TRACE("{}: {}", static_cast<ml_ErrorCode>(result), output);
        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("asr transcribe error: {}", e.what());
        return ML_ERROR_ASR_TRANSCRIPTION;
    }
}

int32_t ml_asr_list_supported_languages(
    const ml_ASR* h, const ml_AsrListSupportedLanguagesInput* input, ml_AsrListSupportedLanguagesOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<IAsr*>(const_cast<ml_ASR*>(h));
        if (!backend) return ML_ERROR_COMMON_NOT_INITIALIZED;
        auto result = backend->list_supported_languages(input, output);
        GENIEX_LOG_TRACE("{}: {}", static_cast<ml_ErrorCode>(result), output);
        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("asr list supported languages error: {}", e.what());
        return ML_ERROR_ASR_LANGUAGE;
    }
}

// =============================================================================
// Streaming ASR C API Implementation
// =============================================================================

int32_t ml_asr_stream_begin(ml_ASR* handle, const ml_AsrStreamBeginInput* input, ml_AsrStreamBeginOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        if (!handle) {
            return ML_ERROR_COMMON_INVALID_INPUT;
        }

        auto* asr    = reinterpret_cast<IAsr*>(handle);
        auto  result = asr->stream_begin(input, output);
        GENIEX_LOG_TRACE("{}: streaming began", static_cast<ml_ErrorCode>(result));
        return result;

    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("asr stream begin error: {}", e.what());
        return ML_ERROR_ASR_STREAM_NOT_STARTED;
    }
}

int32_t ml_asr_stream_push_audio(ml_ASR* handle, const ml_AsrStreamPushAudioInput* input) {
    // Don't trace this - too verbose for audio data
    try {
        if (!handle) {
            return ML_ERROR_COMMON_INVALID_INPUT;
        }

        auto* asr = reinterpret_cast<IAsr*>(handle);
        return asr->stream_push_audio(input);

    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("asr stream push audio error: {}", e.what());
        return ML_ERROR_ASR_STREAM_INVALID_AUDIO;
    }
}

int32_t ml_asr_stream_stop(ml_ASR* handle, const ml_AsrStreamStopInput* input) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        if (!handle) {
            return ML_ERROR_COMMON_INVALID_INPUT;
        }

        auto* asr    = reinterpret_cast<IAsr*>(handle);
        auto  result = asr->stream_stop(input);
        GENIEX_LOG_TRACE("{}: streaming stopped", static_cast<ml_ErrorCode>(result));
        return result;

    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("asr stream stop error: {}", e.what());
        return ML_ERROR_COMMON_UNKNOWN;
    }
}
