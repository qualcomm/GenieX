#include "diarize.h"

#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "image_utils.h"
#include "logging.h"
#include "ml.h"

namespace geniex {

QnnDiarize::QnnDiarize(std::string lib_path) : m_model_impl(nullptr), m_lib_path(std::move(lib_path)) {}

QnnDiarize::~QnnDiarize() {}

/**
 * @brief Create diarization model implementation (Windows: pyannote)
 *
 * @param input Model creation input with model name and config
 * @return ML_SUCCESS or error code
 */
int32_t QnnDiarize::create_impl(const ml_DiarizeCreateInput* input) {
    if (!input || !input->model_name) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    GENIEX_LOG_TRACE("Creating specific npu Diarization model for: {}", fmt::to_string(input->model_name));

    auto model_name = std::string_view(input->model_name);

#if defined(__ANDROID__)

#elif defined(_WIN32)
    if (model_name == "pyannote") {
        m_model_impl = std::unique_ptr<IDiarize>(create_qnn_pyannote());
    } else {
        GENIEX_LOG_ERROR("Unsupported model name for npu Diarization implementation: {}", input->model_name);
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
 * @brief Perform speaker diarization with audio resampling to 16kHz mono
 *
 * @param input Audio file path
 * @param output Speaker segments with timestamps
 * @return ML_SUCCESS or error code
 */
int32_t QnnDiarize::infer(const ml_DiarizeInferInput* input, ml_DiarizeInferOutput* output) {
    if (!m_model_impl || !input || !output) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    auto* mutable_input = const_cast<ml_DiarizeInferInput*>(input);

    ml_Path     original_audio_path = nullptr;
    std::string resampled_audio_path;

    GENIEX_LOG_INFO("Processing audio file for diarization");

    // Resample audio if provided
    if (mutable_input->audio_path && mutable_input->audio_path[0] != '\0') {
        try {
            // Resample audio to 16kHz mono (standard format for diarization)
            const char* audio_array[1] = {mutable_input->audio_path};
            resampled_audio_path       = geniex::image_utils::concat_and_resample_audio(audio_array, 1, 16000, 1);

            // Save original and replace with resampled
            original_audio_path       = mutable_input->audio_path;
            mutable_input->audio_path = resampled_audio_path.c_str();

            GENIEX_LOG_INFO("Audio resampled: {} -> {}", original_audio_path, resampled_audio_path);
        } catch (const std::exception& e) {
            GENIEX_LOG_WARN(
                "Audio resampling failed, proceeding with original audio file. Error: {}", fmt::to_string(e.what()));
            // Keep original audio path - no changes needed
        }
    }

    int32_t result = m_model_impl->infer(mutable_input, output);

    // Restore original audio path if we modified it
    if (original_audio_path) {
        mutable_input->audio_path = original_audio_path;
    }

    return result;
}

}  // namespace geniex
