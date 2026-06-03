#pragma once

#define FMT_HEADER_ONLY
#ifndef FMT_USE_CONSTEVAL
#define FMT_USE_CONSTEVAL 0
#endif

#include <cstring>

#include "external/fmt/core.h"
#include "geniex.h"

GENIEX_API extern geniex_log_callback geniex_log;
GENIEX_API extern geniex_LogLevel     geniex_log_level;

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

template <typename... Args>
void geniex_log_internal(geniex_LogLevel level, const char* file, int32_t line, const char* func,
    fmt::format_string<Args...> fmt, Args&&... args) {
    if (geniex_log == nullptr) return;
    auto p        = std::strstr(file, PROJECT_SOURCE_DIR);
    auto filename = p ? p + std::strlen(PROJECT_SOURCE_DIR) + 1 : file;
    geniex_log(level,
        fmt::format("[{}:{}:{}] {}", filename, line, func, fmt::format(fmt, lp(std::forward<Args>(args))...)).c_str());
}
#define GENIEX_LEVEL_LOG(level, ...)                                                 \
    do {                                                                             \
        if ((level) >= geniex_log_level && geniex_log != nullptr) {                  \
            geniex_log_internal((level), __FILE__, __LINE__, __func__, __VA_ARGS__); \
        }                                                                            \
    } while (0)

// All log macros forward to the registered callback; the embedder is
// responsible for any further filtering.
#define GENIEX_LOG_TRACE(...) GENIEX_LEVEL_LOG(GENIEX_LOG_LEVEL_TRACE, __VA_ARGS__)
#define GENIEX_LOG_DEBUG(...) GENIEX_LEVEL_LOG(GENIEX_LOG_LEVEL_DEBUG, __VA_ARGS__)
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
