#pragma once

#define FMT_HEADER_ONLY
#ifndef FMT_USE_CONSTEVAL
#define FMT_USE_CONSTEVAL 0
#endif
#include "external/fmt/core.h"
#include "geniex.h"

GENIEX_API extern geniex_log_callback geniex_log;

template <typename T>
inline auto lp(T arg) {
    if constexpr (std::is_pointer_v<T>) {
        if (arg == nullptr) {
            return fmt::format("nullptr");
        } else if constexpr (std::is_same_v<std::remove_cv_t<std::remove_pointer_t<T>>, char>) {
            return fmt::format("{}", arg);
        } else if constexpr (std::is_same_v<std::remove_cv_t<T>, void*> ||
                             std::is_same_v<std::remove_cv_t<T>, const void*>) {
            return fmt::format("{}", static_cast<const void*>(arg));
        } else {
            return fmt::format("{}", *arg);
        }
    } else {
        return fmt::format("{}", arg);
    }
}

#ifdef GENIEX_DEBUG

#include <cstring>

template <typename... Args>
void geniex_log_internal(geniex_LogLevel level, const char* file, int32_t line, const char* func,
    fmt::format_string<Args...> fmt, Args&&... args) {
    if (geniex_log == nullptr) return;
#ifdef PROJECT_SOURCE_DIR
    auto p        = std::strstr(file, PROJECT_SOURCE_DIR);
    auto filename = p ? p + std::strlen(PROJECT_SOURCE_DIR) + 1 : file;
#else
    auto filename = file;
#endif
    geniex_log(level,
        fmt::format("[{}:{}:{}] {}", filename, line, func, fmt::format(fmt, lp(std::forward<Args>(args))...)).c_str());
}
#define GENIEX_LEVEL_LOG(level, ...) geniex_log_internal(level, __FILE__, __LINE__, __func__, __VA_ARGS__)

#else  // GENIEX_DEBUG

template <typename... Args>
inline void geniex_log_internal(geniex_LogLevel level, fmt::format_string<Args...> fmt, Args&&... args) {
    if (geniex_log == nullptr) return;
    geniex_log(level, fmt::format(fmt, lp(std::forward<Args>(args))...).c_str());
}
#define GENIEX_LEVEL_LOG(level, ...) geniex_log_internal(level, __VA_ARGS__)

#endif  // GENIEX_DEBUG

#ifdef GENIEX_DEBUG
#define GENIEX_LOG_TRACE(...) GENIEX_LEVEL_LOG(GENIEX_LOG_LEVEL_TRACE, __VA_ARGS__)
#define GENIEX_LOG_DEBUG(...) GENIEX_LEVEL_LOG(GENIEX_LOG_LEVEL_DEBUG, __VA_ARGS__)
#else  // GENIEX_DEBUG
#define GENIEX_LOG_TRACE(...) ((void)0)
#define GENIEX_LOG_DEBUG(...) ((void)0)
#endif  // GENIEX_DEBUG

#define GENIEX_LOG_INFO(...) GENIEX_LEVEL_LOG(GENIEX_LOG_LEVEL_INFO, __VA_ARGS__)
#define GENIEX_LOG_WARN(...) GENIEX_LEVEL_LOG(GENIEX_LOG_LEVEL_WARN, __VA_ARGS__)
#define GENIEX_LOG_ERROR(...) GENIEX_LEVEL_LOG(GENIEX_LOG_LEVEL_ERROR, __VA_ARGS__)

template <>
struct fmt::formatter<geniex_LogLevel> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

    auto format(const geniex_LogLevel& p, fmt::format_context& ctx) const {
        const char* level_str = "";
        switch (p) {
            case GENIEX_LOG_LEVEL_TRACE:
                level_str = "TRACE";
                break;
            case GENIEX_LOG_LEVEL_DEBUG:
                level_str = "DEBUG";
                break;
            case GENIEX_LOG_LEVEL_INFO:
                level_str = "INFO";
                break;
            case GENIEX_LOG_LEVEL_WARN:
                level_str = "WARN";
                break;
            case GENIEX_LOG_LEVEL_ERROR:
                level_str = "ERROR";
                break;
            default:
                level_str = "UNKNOWN";
                break;
        }
        return fmt::format_to(ctx.out(), "LogLevel[{}]({})", lp(static_cast<int32_t>(p)), lp(level_str));
    }
};

template <>
struct fmt::formatter<geniex_KvCacheSaveInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_KvCacheSaveInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(), "KvCacheSaveInput(path: {})", lp(p.path));
    }
};

template <>
struct fmt::formatter<geniex_KvCacheSaveOutput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_KvCacheSaveOutput& _, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(), "KvCacheSaveOutput()");
    }
};

template <>
struct fmt::formatter<geniex_KvCacheLoadInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_KvCacheLoadInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(), "KvCacheLoadInput(path: {})", lp(p.path));
    }
};

template <>
struct fmt::formatter<geniex_KvCacheLoadOutput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_KvCacheLoadOutput& _, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(), "KvCacheLoadOutput()");
    }
};
template <>
struct fmt::formatter<geniex_ErrorCode> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_ErrorCode& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(), "ErrorCode[{}]({})", static_cast<int32_t>(p), (geniex_get_error_message(p)));
    }
};

template <>
struct fmt::formatter<geniex_GetPluginListOutput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_GetPluginListOutput& p, fmt::format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "GetPluginListOutput(plugin_ids: {}, plugin_count: {})", lp(p.plugin_ids), lp(p.plugin_count));
    }
};

template <>
struct fmt::formatter<geniex_GetDeviceListInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_GetDeviceListInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(), "GetDeviceListInput(plugin_id: {})", lp(p.plugin_id));
    }
};

template <>
struct fmt::formatter<geniex_GetDeviceListOutput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_GetDeviceListOutput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "GetDeviceListOutput(device_ids: {}, device_names: {}, device_count: {})",
            lp(fmt::ptr(p.device_ids)),
            lp(fmt::ptr(p.device_names)),
            lp(p.device_count));
    }
};

template <>
struct fmt::formatter<geniex_ProfileData> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_ProfileData& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "ProfileData(ttft: {} us, prompt_time: {} us, decode_time: {} us, prompt_tokens: {}, "
                      "generated_tokens: {}, audio_duration: {} us, prefill_speed: {} tokens/s, decoding_speed: {} "
                      "tokens/s, real_time_factor: {}, stop_reason: {})",
            lp(p.ttft),
            lp(p.prompt_time),
            lp(p.decode_time),
            lp(p.prompt_tokens),
            lp(p.generated_tokens),
            lp(p.audio_duration),
            lp(p.prefill_speed),
            lp(p.decoding_speed),
            lp(p.real_time_factor),
            lp(p.stop_reason));
    }
};

template <>
struct fmt::formatter<geniex_SamplerConfig> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_SamplerConfig& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "SamplerConfig(temperature: {}, top_p: {}, top_k: {}, min_p: {}, repetition_penalty: {}, presence_penalty: "
                      "{}, "
                      "frequency_penalty: {}, seed: {}, grammar_path: {}, grammar_string: {}, enable_json: {})",
            lp(p.temperature),
            lp(p.top_p),
            lp(p.top_k),
            lp(p.min_p),
            lp(p.repetition_penalty),
            lp(p.presence_penalty),
            lp(p.frequency_penalty),
            lp(p.seed),
            lp(p.grammar_path),
            lp(p.grammar_string),
            lp(p.enable_json));
    }
};

template <>
struct fmt::formatter<geniex_GenerationConfig> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_GenerationConfig& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "GenerationConfig(max_tokens: {}, stop_count: {}, n_past: {}, sampler_config: {}, image_paths: {}, "
                      "image_count: {}, audio_paths: {}, audio_count: {}, image_max_length: {})",
            lp(p.max_tokens),
            lp(p.stop_count),
            lp(p.n_past),
            lp(p.sampler_config),
            lp(fmt::ptr(p.image_paths)),
            lp(p.image_count),
            lp(fmt::ptr(p.audio_paths)),
            lp(p.audio_count),
            lp(p.image_max_length));
    }
};

template <>
struct fmt::formatter<geniex_ModelConfig> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_ModelConfig& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "ModelConfig(n_ctx: {}, n_threads: {}, n_threads_batch: {}, n_batch: {}, n_ubatch: {}, n_seq_max: {}, "
                      "n_gpu_layers: {}, "
                      "chat_template_path: {}, chat_template_content: {}, system_prompt: {}, enable_sampling: {}, grammar_str: "
                      "{}, max_tokens: {}, "
                      "enable_thinking: {}, verbose: {})",
            lp(p.n_ctx),
            lp(p.n_threads),
            lp(p.n_threads_batch),
            lp(p.n_batch),
            lp(p.n_ubatch),
            lp(p.n_seq_max),
            lp(p.n_gpu_layers),
            lp(p.chat_template_path),
            lp(p.chat_template_content),
            lp(p.system_prompt),
            lp(p.enable_sampling),
            lp(p.grammar_str),
            lp(p.max_tokens),
            lp(p.enable_thinking),
            lp(p.verbose));
    }
};

// // LLM formatters
template <>
struct fmt::formatter<geniex_LlmCreateInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_LlmCreateInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "LlmCreateInput(model_path: {}, tokenizer_path: {}, config: {}, plugin_id: {}, device_id: {})",
            lp(p.model_path),
            lp(p.tokenizer_path),
            lp(p.config),
            lp(p.plugin_id),
            lp(p.device_id));
    }
};

template <>
struct fmt::formatter<geniex_LlmApplyChatTemplateInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_LlmApplyChatTemplateInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "LlmApplyChatTemplateInput(messages: {}, message_count: {}, tools: {}, enable_thinking: {})",
            lp(fmt::ptr(p.messages)),
            lp(p.message_count),
            lp(p.tools),
            lp(p.enable_thinking));
    }
};

template <>
struct fmt::formatter<geniex_LlmChatMessage> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_LlmChatMessage& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(), "LlmChatMessage(role: {}, content: {})", lp(p.role), lp(p.content));
    }
};

template <>
struct fmt::formatter<geniex_LlmApplyChatTemplateOutput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_LlmApplyChatTemplateOutput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(), "LlmApplyChatTemplateOutput(formatted_text: {})", lp(p.formatted_text));
    }
};

template <>
struct fmt::formatter<geniex_LlmGenerateInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_LlmGenerateInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "LlmGenerateInput(prompt_utf8: {}, input_ids: {}, input_ids_count: {}, config: {}, on_token: {}, "
                      "user_data: {})",
            lp(p.prompt_utf8),
            lp(fmt::ptr(p.input_ids)),
            lp(p.input_ids_count),
            lp(p.config),
            lp(fmt::ptr(p.on_token)),
            lp(fmt::ptr(p.user_data)));
    }
};

template <>
struct fmt::formatter<geniex_LlmGenerateOutput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_LlmGenerateOutput& p, fmt::format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "LlmGenerateOutput(full_text: {}, profile_data: {})", lp(p.full_text), p.profile_data);
    }
};

// // VLM formatters
template <>
struct fmt::formatter<geniex_VlmContent> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_VlmContent& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(), "VlmContent(type: {}, text: {})", lp(p.type), lp(p.text));
    }
};

template <>
struct fmt::formatter<geniex_VlmChatMessage> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_VlmChatMessage& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "VlmChatMessage(role: {}, content_count: {}, contents: {})",
            lp(p.role),
            lp(p.content_count),
            lp(fmt::ptr(p.contents)));
    }
};

template <>
struct fmt::formatter<geniex_VlmCreateInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_VlmCreateInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "VlmCreateInput(model_name: {}, model_path: {}, mmproj_path: {}, config: {}, plugin_id: {}, device_id: {}, "
                      "tokenizer_path: {}, license_id: {}, license_key: {})",
            lp(p.model_name),
            lp(p.model_path),
            lp(p.mmproj_path),
            lp(p.config),
            lp(p.plugin_id),
            lp(p.device_id),
            lp(p.tokenizer_path),
            lp(p.license_id),
            lp(p.license_key));
    }
};

template <>
struct fmt::formatter<geniex_VlmApplyChatTemplateInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_VlmApplyChatTemplateInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "VlmApplyChatTemplateInput(messages: {}, message_count: {}, tools: {}, enable_thinking: {})",
            lp(fmt::ptr(p.messages)),
            lp(p.message_count),
            lp(p.tools),
            lp(p.enable_thinking));
    }
};

template <>
struct fmt::formatter<geniex_VlmApplyChatTemplateOutput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_VlmApplyChatTemplateOutput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(), "VlmApplyChatTemplateOutput(formatted_text: {})", lp(p.formatted_text));
    }
};

template <>
struct fmt::formatter<geniex_VlmGenerateInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_VlmGenerateInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "VlmGenerateInput(prompt_utf8: {}, config: {}, on_token: {}, user_data: {})",
            lp(p.prompt_utf8),
            lp(p.config),
            lp(fmt::ptr(p.on_token)),
            lp(fmt::ptr(p.user_data)));
    }
};

template <>
struct fmt::formatter<geniex_VlmGenerateOutput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_VlmGenerateOutput& p, fmt::format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "VlmGenerateOutput(full_text: {}, profile_data: {})", lp(p.full_text), p.profile_data);
    }
};

// // Embedding formatters
template <>
struct fmt::formatter<geniex_EmbeddingConfig> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_EmbeddingConfig& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "EmbeddingConfig(batch_size: {}, normalize: {}, normalize_method: {})",
            lp(p.batch_size),
            lp(p.normalize),
            lp(p.normalize_method));
    }
};

template <>
struct fmt::formatter<geniex_EmbedderCreateInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_EmbedderCreateInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "EmbedderCreateInput(model_name: {}, model_path: {}, tokenizer_path: {}, config: {}, plugin_id: {}, "
                      "device_id: {})",
            lp(p.model_name),
            lp(p.model_path),
            lp(p.tokenizer_path),
            lp(p.config),
            lp(p.plugin_id),
            lp(p.device_id));
    }
};

template <>
struct fmt::formatter<geniex_EmbedderEmbedInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_EmbedderEmbedInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "EmbedderEmbedInput(texts: {}, text_count: {}, config: {}, input_ids_2d: {}, input_ids_row_lengths: {}, "
                      "input_ids_row_count: {}, task_type: {})",
            lp(fmt::ptr(p.texts)),
            lp(p.text_count),
            lp(p.config),
            lp(fmt::ptr(p.input_ids_2d)),
            lp(fmt::ptr(p.input_ids_row_lengths)),
            lp(p.input_ids_row_count),
            lp(p.task_type));
    }
};

template <>
struct fmt::formatter<geniex_EmbedderEmbedOutput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_EmbedderEmbedOutput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "EmbedderEmbedOutput(embeddings: {}, embedding_count: {}, profile_data: {})",
            lp(fmt::ptr(p.embeddings)),
            lp(p.embedding_count),
            p.profile_data);
    }
};

template <>
struct fmt::formatter<geniex_EmbedderDimOutput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_EmbedderDimOutput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(), "EmbedderDimOutput(dimension: {})", lp(p.dimension));
    }
};

// // Reranking formatters
template <>
struct fmt::formatter<geniex_RerankConfig> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_RerankConfig& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "RerankConfig(batch_size: {}, normalize: {}, normalize_method: {})",
            lp(p.batch_size),
            lp(p.normalize),
            lp(p.normalize_method));
    }
};

template <>
struct fmt::formatter<geniex_RerankerCreateInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_RerankerCreateInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "RerankerCreateInput(model_name: {}, model_path: {}, tokenizer_path: {}, config: {}, plugin_id: {}, "
                      "device_id: {})",
            lp(p.model_name),
            lp(p.model_path),
            lp(p.tokenizer_path),
            lp(p.config),
            lp(p.plugin_id),
            lp(p.device_id));
    }
};

template <>
struct fmt::formatter<geniex_RerankerRerankInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_RerankerRerankInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "RerankerRerankInput(query: {}, documents: {}, documents_count: {}, config: {})",
            lp(p.query),
            lp(fmt::ptr(p.documents)),
            lp(p.documents_count),
            lp(fmt::ptr(p.config)));
    }
};

template <>
struct fmt::formatter<geniex_RerankerRerankOutput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_RerankerRerankOutput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "RerankerRerankOutput(scores: {}, score_count: {}, profile_data: {})",
            lp(fmt::ptr(p.scores)),
            lp(p.score_count),
            p.profile_data);
    }
};

// // Image Generation formatters
template <>
struct fmt::formatter<geniex_ImageSamplerConfig> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_ImageSamplerConfig& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "ImageSamplerConfig(method: {}, steps: {}, guidance_scale: {}, eta: {}, seed: {})",
            lp(p.method),
            lp(p.steps),
            lp(p.guidance_scale),
            lp(p.eta),
            lp(p.seed));
    }
};

template <>
struct fmt::formatter<geniex_SchedulerConfig> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_SchedulerConfig& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "SchedulerConfig(type: {}, num_train_timesteps: {}, steps_offset: {}, beta_start: {}, beta_end: {}, "
                      "beta_schedule: {}, prediction_type: {}, timestep_type: {}, timestep_spacing: {}, "
                      "interpolation_type: {}, config_path: {})",
            lp(p.type),
            lp(p.num_train_timesteps),
            lp(p.steps_offset),
            lp(p.beta_start),
            lp(p.beta_end),
            lp(p.beta_schedule),
            lp(p.prediction_type),
            lp(p.timestep_type),
            lp(p.timestep_spacing),
            lp(p.interpolation_type),
            lp(p.config_path));
    }
};

template <>
struct fmt::formatter<geniex_ImageGenerationConfig> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_ImageGenerationConfig& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "ImageGenerationConfig(prompts: {}, prompt_count: {}, negative_prompts: {}, negative_prompt_count: {}, "
                      "height: {}, width: {}, sampler_config: {}, scheduler_config: {}, strength: {})",
            lp(fmt::ptr(p.prompts)),
            lp(p.prompt_count),
            lp(fmt::ptr(p.negative_prompts)),
            lp(p.negative_prompt_count),
            lp(p.height),
            lp(p.width),
            lp(p.sampler_config),
            lp(p.scheduler_config),
            lp(p.strength));
    }
};

template <>
struct fmt::formatter<geniex_ImageGenCreateInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_ImageGenCreateInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "ImageGenCreateInput(model_name: {}, model_path: {}, config: {}, scheduler_config_path: {}, plugin_id: {}, "
                      "device_id: {})",
            lp(p.model_name),
            lp(p.model_path),
            lp(p.config),
            lp(p.scheduler_config_path),
            lp(p.plugin_id),
            lp(p.device_id));
    }
};

template <>
struct fmt::formatter<geniex_ImageGenTxt2ImgInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_ImageGenTxt2ImgInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "ImageGenTxt2ImgInput(prompt_utf8: {}, config: {}, output_path: {})",
            lp(p.prompt_utf8),
            lp(p.config),
            lp(p.output_path));
    }
};

template <>
struct fmt::formatter<geniex_ImageGenOutput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_ImageGenOutput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(), "ImageGenOutput(output_image_path: {})", lp(p.output_image_path));
    }
};

template <>
struct fmt::formatter<geniex_ImageGenImg2ImgInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_ImageGenImg2ImgInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "ImageGenImg2ImgInput(init_image_path: {}, prompt_utf8: {}, config: {}, output_path: {})",
            lp(p.init_image_path),
            lp(p.prompt_utf8),
            lp(p.config),
            lp(p.output_path));
    }
};

// // Computer Vision formatters
template <>
struct fmt::formatter<geniex_BoundingBox> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_BoundingBox& p, fmt::format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "BoundingBox(x: {}, y: {}, width: {}, height: {})", lp(p.x), lp(p.y), lp(p.width), lp(p.height));
    }
};

template <>
struct fmt::formatter<geniex_CVResult> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_CVResult& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "CVResult(image_paths: {}, image_count: {}, class_id: {}, confidence: {}, bbox: {}, text: {}, "
                      "embedding: {}, embedding_dim: {})",
            lp(fmt::ptr(p.image_paths)),
            lp(p.image_count),
            lp(p.class_id),
            lp(p.confidence),
            lp(p.bbox),
            lp(p.text),
            lp(fmt::ptr(p.embedding)),
            lp(p.embedding_dim));
    }
};

template <>
struct fmt::formatter<geniex_CVCapabilities> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_CVCapabilities& p, fmt::format_context& ctx) const {
        const char* capability_str = "";
        switch (p) {
            case GENIEX_CV_OCR:
                capability_str = "OCR";
                break;
            case GENIEX_CV_CLASSIFICATION:
                capability_str = "CLASSIFICATION";
                break;
            case GENIEX_CV_SEGMENTATION:
                capability_str = "SEGMENTATION";
                break;
            case GENIEX_CV_CUSTOM:
                capability_str = "CUSTOM";
                break;
            default:
                capability_str = "UNKNOWN";
                break;
        }
        return fmt::format_to(ctx.out(), "CVCapabilities[{}]({})", static_cast<int32_t>(p), capability_str);
    }
};

template <>
struct fmt::formatter<geniex_CVModelConfig> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_CVModelConfig& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "geniex_CVModelConfig(capabilities: {}, det_model_path: {}, rec_model_path: {}, char_dict_path: {})",
            lp(p.capabilities),
            lp(p.det_model_path),
            lp(p.rec_model_path),
            lp(p.char_dict_path));
    }
};

template <>
struct fmt::formatter<geniex_CVCreateInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_CVCreateInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "CVCreateInput(model_name: {}, config: {}, plugin_id: {}, device_id: {})",
            lp(p.model_name),
            lp(p.config),
            lp(p.plugin_id),
            lp(p.device_id));
    }
};

template <>
struct fmt::formatter<geniex_CVInferInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_CVInferInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(), "CVInferInput(input_image_path: {})", lp(p.input_image_path));
    }
};

template <>
struct fmt::formatter<geniex_CVInferOutput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_CVInferOutput& p, fmt::format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "CVInferOutput(results: {}, result_count: {})", lp(fmt::ptr(p.results)), lp(p.result_count));
    }
};

// // ASR formatters
template <>
struct fmt::formatter<geniex_ASRConfig> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_ASRConfig& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "ASRConfig(timestamps: {}, beam_size: {}, stream: {})",
            lp(p.timestamps),
            lp(p.beam_size),
            lp(p.stream));
    }
};

template <>
struct fmt::formatter<geniex_ASRResult> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_ASRResult& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "ASRResult(transcript: {}, confidence_scores: {}, confidence_count: {}, timestamps: {}, timestamp_count: "
                      "{})",
            lp(p.transcript),
            lp(fmt::ptr(p.confidence_scores)),
            lp(p.confidence_count),
            lp(fmt::ptr(p.timestamps)),
            lp(p.timestamp_count));
    }
};

template <>
struct fmt::formatter<geniex_AsrCreateInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_AsrCreateInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "AsrCreateInput(model_name: {}, model_path: {}, tokenizer_path: {}, config: {}, language: {}, plugin_id: "
                      "{}, device_id: {}, license_id: {}, license_key: {})",
            lp(p.model_name),
            lp(p.model_path),
            lp(p.tokenizer_path),
            lp(p.config),
            lp(p.language),
            lp(p.plugin_id),
            lp(p.device_id),
            lp(p.license_id),
            lp(p.license_key));
    }
};

template <>
struct fmt::formatter<geniex_AsrListSupportedLanguagesInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_AsrListSupportedLanguagesInput& _, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(), "AsrListSupportedLanguagesInput()");
    }
};

template <>
struct fmt::formatter<geniex_AsrTranscribeInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_AsrTranscribeInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "AsrTranscribeInput(audio_path: {}, config: {}, language: {})",
            lp(p.audio_path),
            lp(p.config),
            lp(p.language));
    }
};

template <>
struct fmt::formatter<geniex_AsrTranscribeOutput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_AsrTranscribeOutput& p, fmt::format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "AsrTranscribeOutput(result: {}, profile_data: {})", lp(p.result), p.profile_data);
    }
};

template <>
struct fmt::formatter<geniex_AsrListSupportedLanguagesOutput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_AsrListSupportedLanguagesOutput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "AsrListSupportedLanguagesOutput(language_codes: {}, language_count: {})",
            lp(p.language_codes),
            lp(p.language_count));
    }
};

// ASR Streaming formatters
template <>
struct fmt::formatter<geniex_ASRStreamConfig> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_ASRStreamConfig& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "ASRStreamConfig(chunk_duration: {}, overlap_duration: {}, sample_rate: {}, max_queue_size: {}, "
                      "buffer_size: {}, timestamps: {}, beam_size: {})",
            lp(p.chunk_duration),
            lp(p.overlap_duration),
            lp(p.sample_rate),
            lp(p.max_queue_size),
            lp(p.buffer_size),
            lp(p.timestamps),
            lp(p.beam_size));
    }
};

template <>
struct fmt::formatter<geniex_AsrStreamBeginInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_AsrStreamBeginInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "AsrStreamBeginInput(stream_config: {}, language: {}, on_transcription: {}, user_data: {})",
            lp(p.stream_config),
            lp(p.language),
            lp(fmt::ptr(p.on_transcription)),
            lp(fmt::ptr(p.user_data)));
    }
};

template <>
struct fmt::formatter<geniex_AsrStreamStopInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_AsrStreamStopInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(), "AsrStreamStopInput(graceful: {})", lp(p.graceful));
    }
};

// // TTS formatters
template <>
struct fmt::formatter<geniex_TTSConfig> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_TTSConfig& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "TTSConfig(voice: {}, speed: {}, seed: {}, sample_rate: {})",
            lp(p.voice),
            lp(p.speed),
            lp(p.seed),
            lp(p.sample_rate));
    }
};

template <>
struct fmt::formatter<geniex_TTSSamplerConfig> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_TTSSamplerConfig& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "TTSSamplerConfig(temperature: {}, noise_scale: {}, length_scale: {})",
            lp(p.temperature),
            lp(p.noise_scale),
            lp(p.length_scale));
    }
};

template <>
struct fmt::formatter<geniex_TtsListAvailableVoicesInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_TtsListAvailableVoicesInput& _, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(), "TtsListAvailableVoicesInput()");
    }
};

template <>
struct fmt::formatter<geniex_TtsListAvailableVoicesOutput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_TtsListAvailableVoicesOutput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "TtsListAvailableVoicesOutput(voice_ids: {}, voice_count: {})",
            lp(fmt::ptr(p.voice_ids)),
            lp(p.voice_count));
    }
};

template <>
struct fmt::formatter<geniex_TtsCreateInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_TtsCreateInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "TtsCreateInput(model_path: {}, vocoder_path: {}, plugin_id: {}, device_id: {})",
            lp(p.model_path),
            lp(p.vocoder_path),
            lp(p.plugin_id),
            lp(p.device_id));
    }
};

template <>
struct fmt::formatter<geniex_TTSResult> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_TTSResult& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "TTSResult(audio_path: {}, duration_seconds: {}, sample_rate: {}, channels: {}, num_samples: {})",
            lp(fmt::ptr(p.audio_path)),
            lp(p.duration_seconds),
            lp(p.sample_rate),
            lp(p.channels),
            lp(p.num_samples));
    }
};

template <>
struct fmt::formatter<geniex_TtsSynthesizeInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_TtsSynthesizeInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "TtsSynthesizeInput(text_utf8: {}, config: {}, output_path: {})",
            lp(p.text_utf8),
            lp(p.config),
            lp(p.output_path));
    }
};

template <>
struct fmt::formatter<geniex_TtsSynthesizeOutput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_TtsSynthesizeOutput& p, fmt::format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "TtsSynthesizeOutput(result: {}, profile_data: {})", lp(p.result), lp(p.profile_data));
    }
};

// // Diarization formatters
template <>
struct fmt::formatter<geniex_DiarizeConfig> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_DiarizeConfig& p, fmt::format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "DiarizeConfig(min_speakers: {}, max_speakers: {})", lp(p.min_speakers), lp(p.max_speakers));
    }
};

template <>
struct fmt::formatter<geniex_DiarizeSpeechSegment> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_DiarizeSpeechSegment& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "DiarizeSpeechSegment(start_time: {}, end_time: {}, speaker_label: {})",
            lp(p.start_time),
            lp(p.end_time),
            lp(p.speaker_label));
    }
};

template <>
struct fmt::formatter<geniex_DiarizeCreateInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_DiarizeCreateInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "DiarizeCreateInput(model_name: {}, model_path: {}, config: {}, plugin_id: {}, device_id: {}, license_id: "
                      "{}, license_key: {})",
            lp(p.model_name),
            lp(p.model_path),
            lp(p.config),
            lp(p.plugin_id),
            lp(p.device_id),
            lp(p.license_id),
            lp(p.license_key));
    }
};

template <>
struct fmt::formatter<geniex_DiarizeInferInput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_DiarizeInferInput& p, fmt::format_context& ctx) const {
        return fmt::format_to(
            ctx.out(), "DiarizeInferInput(audio_path: {}, config: {})", lp(p.audio_path), lp(p.config));
    }
};

template <>
struct fmt::formatter<geniex_DiarizeInferOutput> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }
    auto           format(const geniex_DiarizeInferOutput& p, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(),
            "DiarizeInferOutput(segments: {}, segment_count: {}, num_speakers: {}, duration: {}, profile_data: {})",
            lp(fmt::ptr(p.segments)),
            lp(p.segment_count),
            lp(p.num_speakers),
            lp(p.duration),
            p.profile_data);
    }
};
