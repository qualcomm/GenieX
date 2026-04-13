#include "asr.h"

#include <cstring>
#include <memory>
#include <string>
#include <string_view>

#include "common.h"
#include "logging.h"
#include "ml.h"

namespace geniex {

// =============================================================================
// QnnAsr Factory Implementation
// =============================================================================

QnnAsr::QnnAsr(std::string lib_path) : m_model_impl(), m_lib_path(std::move(lib_path)) {}

QnnAsr::~QnnAsr() {}

/**
 * @brief Create ASR model implementation (parakeet, wav2vec2 supported on Android/Windows/Linux)
 *
 * @param input Model creation input with model name and config
 * @return ML_SUCCESS or error code
 */
int32_t QnnAsr::create_impl(const ml_AsrCreateInput* input) {
    if (!input || !input->model_name) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    std::string_view model_name(input->model_name);

#if defined(__ANDROID__)
    if (model_name == "parakeet") {
        m_model_impl = std::unique_ptr<IAsr>(create_qnn_parakeet());
    } else if (model_name == "wav2vec2") {
        m_model_impl = std::unique_ptr<IAsr>(create_qnn_wav2vec2());
    } else {
        GENIEX_LOG_ERROR("Unsupported model name for npu ASR implementation: {}", input->model_name);
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

#elif defined(_WIN32)
    if (model_name == "parakeet") {
        m_model_impl = std::unique_ptr<IAsr>(create_qnn_parakeet());
    } else if (model_name == "wav2vec2") {
        m_model_impl = std::unique_ptr<IAsr>(create_qnn_wav2vec2());
    } else {
        GENIEX_LOG_ERROR("Unsupported model name for npu ASR implementation: {}", input->model_name);
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

#elif defined(__linux__)
    // Linux: parakeet is now available
    if (model_name == "parakeet") {
        m_model_impl = std::unique_ptr<IAsr>(create_qnn_parakeet());
    } else {
        GENIEX_LOG_ERROR(
            "Unsupported model name for Linux npu ASR implementation: {}. Available models: parakeet", model_name);
        return ML_ERROR_COMMON_INVALID_INPUT;
    }
#endif

    if (!m_model_impl) {
        GENIEX_LOG_ERROR("Failed to create specific model implementation");
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

    // inject qnn path, use class for c_str lifetime management
    QnnFolderPathFiller _(input, m_lib_path);
    return m_model_impl->create(input);
}

/**
 * @brief Transcribe audio file to text
 *
 * @param input Audio file path and transcription config
 * @param output Transcribed text result
 * @return ML_SUCCESS or error code
 */
int32_t QnnAsr::transcribe(const ml_AsrTranscribeInput* input, ml_AsrTranscribeOutput* output) {
    if (!m_model_impl || !output) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    return m_model_impl->transcribe(input, output);
}

/**
 * @brief List languages supported by ASR model
 *
 * @param input Request configuration
 * @param output Array of supported language codes
 * @return ML_SUCCESS or error code
 */
int32_t QnnAsr::list_supported_languages(
    const ml_AsrListSupportedLanguagesInput* input, ml_AsrListSupportedLanguagesOutput* output) {
    if (!m_model_impl || !output) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    return m_model_impl->list_supported_languages(input, output);
}

/**
 * @brief Begin streaming ASR session with default config if not provided
 *
 * @param input Stream configuration (chunk duration, overlap, sample rate, etc.)
 * @param output Stream session handle
 * @return ML_SUCCESS or error code
 */
int32_t QnnAsr::stream_begin(const ml_AsrStreamBeginInput* input, ml_AsrStreamBeginOutput* output) {
    if (!m_model_impl || !input || !output) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    // fill default config
    ml_AsrStreamBeginInput* mutable_input = const_cast<ml_AsrStreamBeginInput*>(input);
    auto                    origin_config = input->stream_config;
    if (input->stream_config == nullptr) {
        ml_ASRStreamConfig default_config{};
        default_config.chunk_duration   = 4.0f;
        default_config.overlap_duration = 3.5f;
        default_config.sample_rate      = 16000;
        default_config.max_queue_size   = 10;
        default_config.buffer_size      = 1024;
        default_config.timestamps       = "segment";
        default_config.beam_size        = 4;

        // inject qnn path
        mutable_input->stream_config = &default_config;
    }

    int32_t res                  = m_model_impl->stream_begin(input, output);
    mutable_input->stream_config = origin_config;  // reset to avoid dangling pointer
    return res;
}

/**
 * @brief Push audio data to streaming ASR session
 *
 * @param input Audio buffer and session handle
 * @return ML_SUCCESS or error code
 */
int32_t QnnAsr::stream_push_audio(const ml_AsrStreamPushAudioInput* input) {
    if (!m_model_impl || !input) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    return m_model_impl->stream_push_audio(input);
}

/**
 * @brief Stop streaming ASR session and finalize transcription
 *
 * @param input Session handle to stop
 * @return ML_SUCCESS or error code
 */
int32_t QnnAsr::stream_stop(const ml_AsrStreamStopInput* input) {
    if (!m_model_impl || !input) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    return m_model_impl->stream_stop(input);
}

}  // namespace geniex
