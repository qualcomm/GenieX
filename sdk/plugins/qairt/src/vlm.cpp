#include "vlm.h"

#include <filesystem>
#include <string>
#include <vector>

#include "common.h"
#include "image_utils.h"
#include "logging.h"
#include "ml.h"

/**
 * @brief Resize image for Qwen3VL model with aspect ratio preservation
 *
 * Resizes width to max_length while maintaining aspect ratio. Only processes if width > max_length.
 *
 * @param image_path Path to input image file
 * @param max_length Maximum width in pixels (default: 512)
 * @return Path to resized temporary image file
 * @throws std::runtime_error if ffmpeg operations fail
 */
std::string resize_qwen3vl_image(const char* image_path, int max_length = 512) {
    auto outfile = geniex::image_utils::generate_temp_filename("qwen3vl-resized-", ".jpg");

    try {
        auto dims   = geniex::image_utils::get_image_dimensions(image_path);
        int  width  = dims.first;
        int  height = dims.second;

        std::string cmd;

        // Apply qwen3vl resize logic - max_length only controls width
        if (width <= max_length) {
            // No action needed, just copy
            GENIEX_LOG_INFO("Image width {} <= max_length {}, no resize needed", width, max_length);
            cmd = "ffmpeg -hide_banner -loglevel error -y -i \"" + std::string(image_path) + "\" \"" + outfile + "\"";
        } else {
            // Calculate new height to maintain aspect ratio
            int new_height = (height * max_length) / width;
            GENIEX_LOG_INFO("Resizing image from {}x{} to {}x{} (max_length: {})",
                width,
                height,
                max_length,
                new_height,
                max_length);

            // Resize width to max_length while keeping aspect ratio
            cmd = "ffmpeg -hide_banner -loglevel error -y -i \"" + std::string(image_path) + "\" " +
                  "-vf \"scale=" + std::to_string(max_length) + ":-1\" " + "\"" + outfile + "\"";
        }

        GENIEX_LOG_DEBUG("qwen3vl resize image ffmpeg command: {}", cmd);

        int ret = std::system(cmd.c_str());
        if (ret != 0) {
            throw std::runtime_error("ffmpeg qwen3vl resize failed, command: " + cmd);
        }
        return outfile;
    } catch (const std::exception& e) {
        // Clean up output file if it exists
        std::remove(outfile.c_str());
        throw;
    }
}

namespace geniex {

QnnVlm::QnnVlm(std::string lib_path) : m_model_impl(), m_lib_path(std::move(lib_path)) {}
QnnVlm::~QnnVlm() {}

/**
 * @brief Create platform-specific VLM model implementation
 *
 * Android: omni-neural, qwen3vl, autoneural. Windows: omni-neural, qwen3vl, autoneural. Linux: autoneural.
 *
 * @param input Model creation input with model name and config
 * @return ML_SUCCESS or error code
 */
int32_t QnnVlm::create_impl(const ml_VlmCreateInput* input) {
    if (!input || !input->model_name) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    std::string_view model_name(input->model_name);
    m_model_name = std::string(model_name);  // Store model name for later use

#if defined(__ANDROID__)
    // Android: omni-neural, qwen3vl, and autoneural are supported
    if (model_name == "omni-neural") {
        m_model_impl = std::unique_ptr<IVlm>(create_qnn_omni_neural());
    } else if (model_name == "qwen3vl") {
        m_model_impl = std::unique_ptr<IVlm>(create_qnn_qwen3vl());
    } else if (model_name == "autoneural") {
        m_model_impl = std::unique_ptr<IVlm>(create_qnn_autoneural());
    } else {
        GENIEX_LOG_ERROR("Unsupported VLM model for Android: {}", model_name);
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

#elif defined(_WIN32)
    // Windows: omni-neural, qwen3vl, and autoneural are supported
    if (model_name == "omni-neural") {
        m_model_impl = std::unique_ptr<IVlm>(create_qnn_omni_neural());
    } else if (model_name == "qwen3vl") {
        m_model_impl = std::unique_ptr<IVlm>(create_qnn_qwen3vl());
    } else if (model_name == "auto-neural") {
        m_model_impl = std::unique_ptr<IVlm>(create_qnn_autoneural());
    } else {
        GENIEX_LOG_ERROR("Unsupported VLM model: {}", model_name);
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

#elif defined(__linux__)
    // Linux: autoneural is supported
    if (model_name == "auto-neural") {
        m_model_impl = std::unique_ptr<IVlm>(create_qnn_autoneural());
    } else {
        GENIEX_LOG_ERROR("Unsupported VLM model for Linux: {}", model_name);
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

#endif

    if (!m_model_impl) {
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

    // inject qnn path, use class for c_str lifetime management
    QnnFolderPathFiller _(input, m_lib_path);
    return m_model_impl->create(input);
}

/**
 * @brief Reset VLM model state
 *
 * @return ML_SUCCESS or error code
 */
int32_t QnnVlm::reset() {
    if (!m_model_impl) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }
    return m_model_impl->reset();
}

/**
 * @brief Extract text content from last message (images/audio filtered out)
 *
 * @param input Chat messages with multi-modal content
 * @param output Formatted text output (text only, no images/audio)
 * @return ML_SUCCESS or error code
 */
int32_t QnnVlm::apply_chat_template(const ml_VlmApplyChatTemplateInput* input, ml_VlmApplyChatTemplateOutput* output) {
    if (!m_model_impl) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }
    return m_model_impl->apply_chat_template(input, output);
}

/**
 * @brief Generate VLM output with audio/image preprocessing
 *
 * Preprocessing (model-specific):
 * - Audio: Concatenates and resamples multiple files into single 16kHz mono
 * - Images: qwen3vl uses aspect-ratio preserving resize, omni-neural uses resize+pad
 *
 * Falls back to original files if preprocessing fails (non-blocking errors).
 *
 * @param input Generation input with prompt, images, audio, config
 * @param output Generated text response
 * @return ML_SUCCESS or error code
 */
int32_t QnnVlm::generate(const ml_VlmGenerateInput* input, ml_VlmGenerateOutput* output) {
    if (!m_model_impl || !input || !output || !input->config) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    ml_VlmGenerateInput processed_input = *input;
    ml_GenerationConfig processed_config = *input->config;
    processed_input.config = &processed_config;

    std::vector<std::string> audio_paths_storage;
    std::vector<std::string> image_paths_storage;
    std::vector<ml_Path>     audio_paths_array;
    std::vector<ml_Path>     image_paths_array;

    if (processed_config.audio_paths && processed_config.audio_count > 0) {
        GENIEX_LOG_INFO("Preprocessing {} audio file(s)", processed_config.audio_count);

        try {
            std::string concatenated_audio = geniex::image_utils::concat_and_resample_audio(
                processed_config.audio_paths,
                processed_config.audio_count
            );

            audio_paths_storage.push_back(std::move(concatenated_audio));
            audio_paths_array.push_back(audio_paths_storage.back().c_str());

            processed_config.audio_paths = audio_paths_array.data();
            processed_config.audio_count = 1;

            GENIEX_LOG_INFO("Audio preprocessing complete: {} files → {}",
                input->config->audio_count, audio_paths_storage.back());
        } catch (const std::exception& e) {
            GENIEX_LOG_WARN("Audio preprocessing failed, using original files. Error: {}",
                fmt::to_string(e.what()));
        }
    }

    if (processed_config.image_paths && processed_config.image_count > 0) {
        GENIEX_LOG_INFO("Preprocessing {} image(s) for model: {}",
            processed_config.image_count, m_model_name);

        image_paths_array.reserve(processed_config.image_count);

        for (int i = 0; i < processed_config.image_count; i++) {
            const char* original_path = processed_config.image_paths[i];
            std::string processed_path;

            if (processed_config.image_max_length <= 0) {
                GENIEX_LOG_DEBUG("Skipping image {} resize (image_max_length: {})",
                    i + 1, processed_config.image_max_length);
                processed_path = original_path;
            } else {
                try {
                    if (m_model_name == "qwen3vl") {
                        processed_path = resize_qwen3vl_image(original_path, processed_config.image_max_length);
                    } else {
                        processed_path = geniex::image_utils::resize_and_pad_image(original_path);
                    }

                    GENIEX_LOG_DEBUG("Image {}/{}: {} -> {}",
                        i + 1, processed_config.image_count, original_path, processed_path);
                } catch (const std::exception& e) {
                    GENIEX_LOG_WARN("Image {} preprocessing failed, using original. Error: {}",
                        i + 1, fmt::to_string(e.what()));
                    processed_path = original_path;
                }
            }

            image_paths_storage.push_back(std::move(processed_path));
            image_paths_array.push_back(image_paths_storage.back().c_str());
        }

        processed_config.image_paths = image_paths_array.data();
        processed_config.image_count = static_cast<int>(image_paths_array.size());
    }

    return m_model_impl->generate(&processed_input, output);
}

}  // namespace geniex
