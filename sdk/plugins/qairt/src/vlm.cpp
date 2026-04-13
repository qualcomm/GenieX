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
 * Audio: concatenates and resamples multiple files into single 16kHz mono.
 * Images: model-specific resize (qwen3vl: aspect-ratio, omni-neural: pad).
 *
 * @param input Generation input with prompt, images, audio, config
 * @param output Generated text response
 * @return ML_SUCCESS or error code
 */
int32_t QnnVlm::generate(const ml_VlmGenerateInput* input, ml_VlmGenerateOutput* output) {
    if (!m_model_impl) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }
    if (!input || !output) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    auto* mutable_input  = const_cast<ml_VlmGenerateInput*>(input);
    auto* mutable_config = const_cast<ml_GenerationConfig*>(mutable_input->config);

    ml_Path* original_audio_paths = nullptr;
    int      original_audio_count = 0;
    ml_Path* original_image_paths = nullptr;
    int      original_image_count = 0;

    std::vector<std::vector<char>> audio_buffers;
    std::vector<std::vector<char>> image_buffers;

    GENIEX_LOG_INFO("found {} audio(s) to process", mutable_config->audio_count);

    // Multiple audio files → Single file
    if (mutable_config->audio_paths && mutable_config->audio_count > 0) {
        try {
            std::string new_audio_path =
                geniex::image_utils::concat_and_resample_audio(mutable_config->audio_paths, mutable_config->audio_count);

            // Only save original and allocate new if preprocessing succeeds
            original_audio_paths = mutable_config->audio_paths;
            original_audio_count = mutable_config->audio_count;

            audio_buffers.emplace_back(new_audio_path.begin(), new_audio_path.end());
            audio_buffers.back().push_back('\0');

            mutable_config->audio_paths    = new ml_Path[1];
            mutable_config->audio_paths[0] = audio_buffers.back().data();
            mutable_config->audio_count    = 1;

            GENIEX_LOG_INFO("concat and resample {} audios -> {}", original_audio_count, new_audio_path);
        } catch (const std::exception& e) {
            GENIEX_LOG_WARN("Audio preprocessing failed ({}), proceeding with original audio files. Error: {}",
                mutable_config->audio_count > 1 ? "concat/resample" : "resample",
                fmt::to_string(e.what()));
            // Keep original audio paths - no changes needed, original_audio_paths remains nullptr
        }
    }

    GENIEX_LOG_INFO("found {} images to process", mutable_config->image_count);

    if (mutable_config->image_paths && mutable_config->image_count > 0) {
        original_image_paths = mutable_config->image_paths;
        original_image_count = mutable_config->image_count;

        mutable_config->image_paths = new ml_Path[original_image_count];
        for (int i = 0; i < original_image_count; i++) {
            std::string resized_image_path;

            // Skip resize if image_max_length is 0 or negative
            if (mutable_config->image_max_length <= 0) {
                GENIEX_LOG_INFO("Skipping image resize due to image_max_length: {} (model: {})",
                    mutable_config->image_max_length,
                    m_model_name);
                resized_image_path = std::string(original_image_paths[i]);
            } else {
                auto image_max_length = mutable_config->image_max_length;

                try {
                    if (m_model_name == "qwen3vl") {
                        // For qwen3vl, use the new resize logic
                        resized_image_path = resize_qwen3vl_image(original_image_paths[i], image_max_length);
                    } else {
                        // For other models (omni-neural), use the original resize and pad logic
                        resized_image_path = geniex::image_utils::resize_and_pad_image(original_image_paths[i]);
                    }
                } catch (const std::exception& e) {
                    GENIEX_LOG_WARN("Image {} preprocessing failed, proceeding with original file. Error: {}",
                        i + 1,
                        fmt::to_string(e.what()));
                    // Use original image path
                    resized_image_path = std::string(original_image_paths[i]);
                }
            }

            image_buffers.emplace_back(resized_image_path.begin(), resized_image_path.end());
            image_buffers.back().push_back('\0');

            mutable_config->image_paths[i] = image_buffers.back().data();

            GENIEX_LOG_INFO("Processed image {}/{}: {} -> {} (model: {})",
                i + 1,
                original_image_count,
                original_image_paths[i],
                resized_image_path,
                m_model_name);
        }
        mutable_config->image_count = original_image_count;
    }

    int32_t result = m_model_impl->generate(mutable_input, output);

    if (original_audio_paths) {
        delete[] mutable_config->audio_paths;
        mutable_config->audio_paths = original_audio_paths;
        mutable_config->audio_count = original_audio_count;
    }
    if (original_image_paths) {
        delete[] mutable_config->image_paths;
        mutable_config->image_paths = original_image_paths;
        mutable_config->image_count = original_image_count;
    }

    return result;
}

}  // namespace geniex
