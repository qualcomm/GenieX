#include "vlm.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#define portable_strdup _strdup
#else
#define portable_strdup strdup
#endif

#include "logging.h"
#include "qnn_runtime_utils.h"
#include "vlm_model_registry.h"
#include "vlm/vlm_types.h"
#include "types.h"

namespace fs = std::filesystem;

namespace geniex {

QairtVlm::~QairtVlm() = default;

// ── Helpers ──────────────────────────────────────────────────────────────────

// Convert BatchFeatures pixel data to geniex::PixelData.
static PixelData toPixelData(const mm_process::BatchFeatures& bf) {
    PixelData pd;
    if (bf.image_grid_thw.dimension() == 0 || bf.image_grid_thw.shape()[0] == 0)
        return pd;

    pd.pixel_values.assign(bf.pixel_values.cbegin(), bf.pixel_values.cend());

    const size_t n = bf.image_grid_thw.shape()[0];
    pd.image_grid_thw.resize(n);
    for (size_t i = 0; i < n; ++i) {
        pd.image_grid_thw[i] = {
            static_cast<int32_t>(bf.image_grid_thw(i, 0)),
            static_cast<int32_t>(bf.image_grid_thw(i, 1)),
            static_cast<int32_t>(bf.image_grid_thw(i, 2)),
        };
    }
    return pd;
}

// Convert BatchFeatures audio data to geniex::AudioData.
static AudioData toAudioData(const mm_process::BatchFeatures& bf) {
    AudioData ad;
    if (bf.audio_features.dimension() < 3 || bf.audio_features.shape()[2] == 0)
        return ad;

    const int32_t num_mel  = static_cast<int32_t>(bf.audio_features.shape()[1]);
    const int32_t T_padded = static_cast<int32_t>(bf.audio_features.shape()[2]);

    int32_t valid_frames = 0;
    for (int32_t t = 0; t < T_padded; ++t)
        valid_frames += bf.audio_attention_mask(0, t);

    ad.num_mel_bins = num_mel;
    ad.num_frames   = valid_frames;
    ad.audio_features.resize(static_cast<size_t>(num_mel) * valid_frames);
    for (int32_t m = 0; m < num_mel; ++m)
        for (int32_t t = 0; t < valid_frames; ++t)
            ad.audio_features[m * valid_frames + t] = bf.audio_features(0, m, t);

    ad.audio_attention_mask.assign(valid_frames, 1);
    return ad;
}

// ── create_impl ──────────────────────────────────────────────────────────────

int32_t QairtVlm::create_impl(const ml_VlmCreateInput* input) {
    if (!input || !input->model_name || !input->model_path) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    model_name_ = input->model_name;
    enable_thinking_ = input->config.enable_thinking;

    // Look up model in VLM registry
    auto& registry = vlm_model_registry();
    auto it = registry.find(model_name_);
    if (it == registry.end()) {
        GENIEX_LOG_ERROR("Unknown QAIRT VLM model name: {}", model_name_);
        return ML_ERROR_COMMON_MODEL_INVALID;
    }

    const auto& entry = it->second;

    // Parse model_path to get model directory
    fs::path model_path(input->model_path);
    fs::path model_dir = model_path.parent_path();

    const fs::path qnn_lib_dir = qairt::runtime::resolve_qnn_lib_dir(
        model_dir, input->config.qnn_lib_folder_path);

    GENIEX_LOG_DEBUG("VLM QNN lib dir resolved to: {}", qnn_lib_dir.string());

    QnnRuntimeConfig runtime_cfg = qairt::runtime::make_qnn_runtime_config(qnn_lib_dir);
    qairt::runtime::configure_qnn_dll_search_path(qnn_lib_dir);

    // Create VLM model via registry factory
    model_ = entry.make_model(runtime_cfg, model_dir);
    if (!model_) {
        GENIEX_LOG_ERROR("Failed to create QAIRT VLM model: {}", model_name_);
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

    // Resolve tokenizer path
    if (input->tokenizer_path && input->tokenizer_path[0] != '\0') {
        tokenizer_path_ = input->tokenizer_path;
    } else {
        tokenizer_path_ = qairt::runtime::find_optional_file(model_dir, "tokenizer.json");
    }
    if (tokenizer_path_.empty()) {
        GENIEX_LOG_ERROR("tokenizer.json not found for VLM model: {}", model_name_);
        model_.reset();
        return ML_ERROR_COMMON_FILE_NOT_FOUND;
    }

    // Create tokenizer
    tokenizer_ = tokenizers::Tokenizer::FromJSON(tokenizer_path_);
    if (!tokenizer_) {
        GENIEX_LOG_ERROR("Failed to load tokenizer: {}", tokenizer_path_);
        model_.reset();
        return ML_ERROR_COMMON_FILE_NOT_FOUND;
    }

    // Create mm-process processor based on model type
    switch (entry.processor_type) {
    case VlmProcessorType::kQwen2_5Omni:
        omni_processor_ = mm_process::qwen2_5_omni::create_qwen2_5_omni_processor(tokenizer_path_);
        if (!omni_processor_) {
            GENIEX_LOG_ERROR("Failed to create Qwen2.5-Omni processor");
            model_.reset();
            return ML_ERROR_COMMON_MODEL_LOAD;
        }
        break;
    case VlmProcessorType::kQwen3VL:
        qwen3vl_processor_ = mm_process::qwen3vl::create_qwen3vl_processor(tokenizer_path_);
        if (!qwen3vl_processor_) {
            GENIEX_LOG_ERROR("Failed to create Qwen3-VL processor");
            model_.reset();
            return ML_ERROR_COMMON_MODEL_LOAD;
        }
        break;
    }

    // Set system prompt if provided
    // (system prompt is typically baked into the chat template for VLM models)

    GENIEX_LOG_DEBUG("QAIRT VLM created: model={}, processor={}", model_name_, entry.pipeline_name);
    return ML_SUCCESS;
}

// ── reset ────────────────────────────────────────────────────────────────────

int32_t QairtVlm::reset() {
    if (!model_) return ML_ERROR_COMMON_NOT_INITIALIZED;
    model_->resetKVCache();
    return ML_SUCCESS;
}

// ── apply_chat_template ──────────────────────────────────────────────────────

int32_t QairtVlm::apply_chat_template(const ml_VlmApplyChatTemplateInput* input,
                                       ml_VlmApplyChatTemplateOutput* output) {
    if (!model_) return ML_ERROR_COMMON_NOT_INITIALIZED;
    if (!input || !output) return ML_ERROR_COMMON_INVALID_INPUT;
    if (!input->messages || input->message_count <= 0) return ML_ERROR_COMMON_INVALID_INPUT;

    // Convert ml_VlmChatMessage array to mm_process::ChatMessage vector
    std::vector<mm_process::ChatMessage> messages;
    messages.reserve(input->message_count);

    for (int32_t i = 0; i < input->message_count; ++i) {
        const auto& msg = input->messages[i];
        mm_process::ChatMessage cm(
            msg.role ? msg.role : "user",
            "");

        if (msg.contents && msg.content_count > 0) {
            for (int64_t j = 0; j < msg.content_count; ++j) {
                const auto& c = msg.contents[j];
                if (!c.type) continue;

                if (std::strcmp(c.type, "text") == 0) {
                    cm.content += (c.text ? c.text : "");
                } else if (std::strcmp(c.type, "image") == 0) {
                    mm_process::MMContent mmc{};
                    mmc.type = mm_process::MMContent::Type::IMAGE;
                    mmc.image = c.text ? c.text : "";
                    cm.mm_contents.push_back(std::move(mmc));
                } else if (std::strcmp(c.type, "audio") == 0) {
                    mm_process::MMContent mmc{};
                    mmc.type = mm_process::MMContent::Type::AUDIO;
                    mmc.audio = c.text ? c.text : "";
                    cm.mm_contents.push_back(std::move(mmc));
                }
            }
        }
        messages.push_back(std::move(cm));
    }

    bool thinking = input->enable_thinking || enable_thinking_;

    // Apply chat template based on processor type
    std::string formatted;
    if (omni_processor_) {
        formatted = mm_process::qwen2_5_omni::apply_chat_template(messages, true, thinking);
    } else if (qwen3vl_processor_) {
        formatted = mm_process::qwen3vl::apply_chat_template(messages, true, thinking);
    } else {
        GENIEX_LOG_ERROR("No VLM processor available for chat template");
        return ML_ERROR_COMMON_NOT_INITIALIZED;
    }

    output->formatted_text = portable_strdup(formatted.c_str());
    if (!output->formatted_text) return ML_ERROR_COMMON_MEMORY_ALLOCATION;

    return ML_SUCCESS;
}

// ── generate ─────────────────────────────────────────────────────────────────

int32_t QairtVlm::generate(const ml_VlmGenerateInput* input, ml_VlmGenerateOutput* output) {
    if (!model_) return ML_ERROR_COMMON_NOT_INITIALIZED;
    if (!input || !output) return ML_ERROR_COMMON_INVALID_INPUT;
    if (!input->prompt_utf8) return ML_ERROR_COMMON_INVALID_INPUT;

    // ── Map ml_GenerationConfig ──────────────────────────────────────────
    GenerationConfig gen_cfg{};
    if (input->config) {
        gen_cfg.max_tokens = input->config->max_tokens > 0 ? input->config->max_tokens : 512;
        if (input->config->sampler_config) {
            gen_cfg.temperature = input->config->sampler_config->temperature;
            gen_cfg.top_p = input->config->sampler_config->top_p;
        }
    }
    gen_cfg.thinking_mode = enable_thinking_;

    // ── Collect media paths from config ──────────────────────────────────
    // Media paths are passed via ml_GenerationConfig and need to be loaded
    // by mm-process to produce PixelData/AudioData for the VLM encoders.
    std::vector<std::string> image_paths;
    std::vector<std::string> audio_paths;
    if (input->config) {
        if (input->config->image_paths && input->config->image_count > 0) {
            for (int32_t i = 0; i < input->config->image_count; ++i) {
                if (input->config->image_paths[i])
                    image_paths.emplace_back(input->config->image_paths[i]);
            }
        }
        if (input->config->audio_paths && input->config->audio_count > 0) {
            for (int32_t i = 0; i < input->config->audio_count; ++i) {
                if (input->config->audio_paths[i])
                    audio_paths.emplace_back(input->config->audio_paths[i]);
            }
        }
    }

    // ── Build ChatMessage for mm-process media extraction ────────────────
    // prompt_utf8 is already chat-template-formatted text from
    // apply_chat_template().  We build a synthetic message list so that
    // process_mm_info() can locate and load the referenced media files.
    mm_process::ChatMessage user_msg(mm_process::ChatMessage::ROLE_USER, "");
    for (const auto& img : image_paths) {
        mm_process::MMContent mmc{};
        mmc.type = mm_process::MMContent::Type::IMAGE;
        mmc.image = img;
        if (input->config && input->config->image_max_length > 0) {
            mmc.max_pixels = input->config->image_max_length * input->config->image_max_length;
        }
        user_msg.mm_contents.push_back(std::move(mmc));
    }
    for (const auto& aud : audio_paths) {
        mm_process::MMContent mmc{};
        mmc.type = mm_process::MMContent::Type::AUDIO;
        mmc.audio = aud;
        user_msg.mm_contents.push_back(std::move(mmc));
    }
    std::vector<mm_process::ChatMessage> messages = {std::move(user_msg)};

    // ── Preprocess with mm-process ───────────────────────────────────────
    // The processor tokenises prompt_utf8 (already formatted) and loads
    // image/audio files into tensors for the VLM encoders.
    mm_process::BatchFeatures bf;

    try {
        if (omni_processor_) {
            auto [audios, images, videos] =
                mm_process::qwen2_5_omni::process_mm_info(messages);
            bf = omni_processor_->process(input->prompt_utf8, images, videos, audios);
        } else if (qwen3vl_processor_) {
            auto [images, videos] =
                mm_process::qwen3vl::process_mm_info(messages);
            bf = qwen3vl_processor_->process(input->prompt_utf8, images, videos);
        } else {
            GENIEX_LOG_ERROR("No VLM processor available");
            return ML_ERROR_COMMON_NOT_INITIALIZED;
        }
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("VLM preprocessing failed: {}", e.what());
        return ML_ERROR_VLM_GENERATION_FAILED;
    }

    // ── Convert BatchFeatures to geniex types ────────────────────────────
    AudioVLMInput vlm_input;
    vlm_input.pixel_data = toPixelData(bf);
    vlm_input.audio_data = toAudioData(bf);

    std::vector<int32_t> prompt_tokens(bf.input_ids.cbegin(), bf.input_ids.cend());

    // ── Generate ─────────────────────────────────────────────────────────
    using Clock = std::chrono::high_resolution_clock;
    auto t_start = Clock::now();
    Clock::time_point t_first_token;
    bool got_first = false;
    bool user_stopped = false;

    std::ostringstream full_text;

    // Wrap token callback.  VLMModel::generate() uses void(int32_t),
    // so we always install a callback to capture timing and text.
    auto token_cb = [&](int32_t tok) {
        if (!got_first) {
            t_first_token = Clock::now();
            got_first = true;
        }

        std::string piece = tokenizer_->Decode({tok});
        full_text << piece;

        if (input->on_token && !piece.empty()) {
            if (!input->on_token(piece.c_str(), input->user_data))
                user_stopped = true;
        }
    };

    std::vector<int32_t> output_tokens;
    try {
        output_tokens = model_->generate(prompt_tokens, vlm_input, gen_cfg, token_cb);
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("VLM generation failed: {}", e.what());
        model_->resetKVCache();
        return ML_ERROR_VLM_GENERATION_FAILED;
    }

    auto t_end = Clock::now();

    // ── Map result to output ─────────────────────────────────────────────
    std::string text = full_text.str();
    output->full_text = portable_strdup(text.c_str());
    if (!output->full_text) return ML_ERROR_COMMON_MEMORY_ALLOCATION;

    // Profile data (convert ms -> us)
    if (got_first) {
        double ttft_ms = std::chrono::duration<double, std::milli>(
                             t_first_token - t_start).count();
        double decode_ms = std::chrono::duration<double, std::milli>(
                               t_end - t_first_token).count();

        output->profile_data.ttft = static_cast<int64_t>(ttft_ms * 1000.0);
        output->profile_data.prompt_time = output->profile_data.ttft;
        output->profile_data.decode_time = static_cast<int64_t>(decode_ms * 1000.0);
        output->profile_data.prompt_tokens = static_cast<int64_t>(prompt_tokens.size());
        output->profile_data.generated_tokens = static_cast<int64_t>(output_tokens.size());

        size_t decode_tok = output_tokens.size() > 1 ? output_tokens.size() - 1 : 0;
        output->profile_data.decoding_speed =
            decode_ms > 0.0 ? decode_tok / (decode_ms / 1000.0) : 0.0;
        output->profile_data.prefill_speed =
            prompt_tokens.size() > 0 && ttft_ms > 0.0
                ? prompt_tokens.size() / (ttft_ms / 1000.0)
                : 0.0;
    }

    // Stop reason
    static const char* kStopEos = "eos";
    static const char* kStopLength = "length";
    static const char* kStopUser = "user";
    if (user_stopped)
        output->profile_data.stop_reason = kStopUser;
    else if (static_cast<int64_t>(output_tokens.size()) >= gen_cfg.max_tokens)
        output->profile_data.stop_reason = kStopLength;
    else
        output->profile_data.stop_reason = kStopEos;

    return ML_SUCCESS;
}

}  // namespace geniex
