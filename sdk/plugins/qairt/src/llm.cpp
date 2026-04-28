#include "llm.h"

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#if defined(_WIN32)
#define portable_strdup _strdup
#else
#define portable_strdup strdup
#endif

#include "llm_model_registry.h"  // provided by geniex-qairt/models/
#include "logging.h"
#include "pipeline/llm_pipeline.h"
#include "qnn_runtime_utils.h"
#include "types.h"

namespace fs = std::filesystem;

namespace geniex {

QairtLlm::~QairtLlm() = default;

int32_t QairtLlm::create_impl(const geniex_LlmCreateInput* input) {
    if (!input || !input->model_name || !input->model_path) {
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    model_name_      = input->model_name;
    enable_thinking_ = input->config.enable_thinking;

    // Look up model in registry
    auto& registry = llm_model_registry();
    auto  it       = registry.find(model_name_);
    if (it == registry.end()) {
        GENIEX_LOG_ERROR("Unknown QAIRT model name: {}", model_name_);
        return GENIEX_ERROR_COMMON_MODEL_INVALID;
    }

    const auto& entry = it->second;

    // Parse model_path to get model directory
    fs::path model_path(input->model_path);
    fs::path model_dir = model_path.parent_path();

    QnnRuntimeConfig runtime_cfg = qairt::runtime::make_qnn_runtime_config(model_dir);

    // Discover .bin model shards
    auto bin_shards = qairt::runtime::collect_bin_files(model_dir);
    if (bin_shards.empty()) {
        GENIEX_LOG_ERROR("No .bin model shards found in: {}", model_dir.string());
        return GENIEX_ERROR_COMMON_FILE_NOT_FOUND;
    }

    GENIEX_LOG_DEBUG("Found {} model shards in {}", bin_shards.size(), model_dir.string());

    // Build ModelConfig
    ModelConfig model_cfg{};
    model_cfg.model_paths = std::move(bin_shards);

    // Tokenizer path
    if (input->tokenizer_path && input->tokenizer_path[0] != '\0') {
        model_cfg.tokenizer_path = input->tokenizer_path;
    } else {
        model_cfg.tokenizer_path = qairt::runtime::find_optional_file(model_dir, "tokenizer.json");
    }
    if (model_cfg.tokenizer_path.empty()) {
        GENIEX_LOG_ERROR("tokenizer.json not found in: {}", model_dir.string());
        return GENIEX_ERROR_COMMON_FILE_NOT_FOUND;
    }

    // Embedding table (optional - AI Hub models do embedding on-device)
    model_cfg.embedding_path = qairt::runtime::find_optional_file(model_dir, "embed_tokens.npy");

    // HTP backend config
    model_cfg.htp_config_path = qairt::runtime::find_optional_file(model_dir, "htp_backend_ext_config.json");

    // Create LLMPipeline via the per-model factory (handles makeModel + chat template internally).
    // Returns std::nullopt on QNN init failure, missing tokenizer, etc.
    auto pipe = entry.make_pipeline(runtime_cfg, model_cfg);
    if (!pipe) {
        GENIEX_LOG_ERROR("Failed to create QAIRT LLM pipeline for model: {}", model_name_);
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
    pipeline_ = std::make_unique<LLMPipeline>(std::move(*pipe));

    // Set system prompt if provided
    if (input->config.system_prompt && input->config.system_prompt[0] != '\0') {
        pipeline_->setSystemPrompt(input->config.system_prompt);
    }

    GENIEX_LOG_DEBUG("QAIRT LLM created successfully: model={}", model_name_);
    return GENIEX_SUCCESS;
}

int32_t QairtLlm::reset() {
    if (!pipeline_) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
    pipeline_->reset();
    return GENIEX_SUCCESS;
}

int32_t QairtLlm::save_kv_cache(const geniex_KvCacheSaveInput* input, geniex_KvCacheSaveOutput*) {
    if (!pipeline_) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
    if (!input || !input->path) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    pipeline_->saveKVCache(input->path);
    return GENIEX_SUCCESS;
}

int32_t QairtLlm::load_kv_cache(const geniex_KvCacheLoadInput* input, geniex_KvCacheLoadOutput*) {
    if (!pipeline_) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
    if (!input || !input->path) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    pipeline_->loadKVCache(input->path);
    return GENIEX_SUCCESS;
}

int32_t QairtLlm::apply_chat_template(
    const geniex_LlmApplyChatTemplateInput* input, geniex_LlmApplyChatTemplateOutput* output) {
    if (!pipeline_) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
    if (!input || !output) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    if (!input->messages || input->message_count <= 0) return GENIEX_ERROR_COMMON_INVALID_INPUT;

    // Extract the last user message
    const char* user_message = nullptr;
    for (int32_t i = input->message_count - 1; i >= 0; --i) {
        if (input->messages[i].role && std::strcmp(input->messages[i].role, "user") == 0) {
            user_message = input->messages[i].content;
            break;
        }
    }

    if (!user_message) {
        GENIEX_LOG_ERROR("No user message found in chat messages");
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    bool        thinking  = input->enable_thinking || enable_thinking_;
    std::string formatted = pipeline_->applyChatTemplate(user_message, thinking);

    output->formatted_text = portable_strdup(formatted.c_str());
    if (!output->formatted_text) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;

    return GENIEX_SUCCESS;
}

int32_t QairtLlm::generate(const geniex_LlmGenerateInput* input, geniex_LlmGenerateOutput* output) {
    if (!pipeline_) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
    if (!input || !output) return GENIEX_ERROR_COMMON_INVALID_INPUT;

    // Check for input_ids path (not supported in Phase 1)
    if (input->input_ids && input->input_ids_count > 0) {
        GENIEX_LOG_ERROR("QAIRT plugin does not support input_ids yet, use prompt_utf8");
        return GENIEX_ERROR_COMMON_NOT_SUPPORTED;
    }

    if (!input->prompt_utf8) return GENIEX_ERROR_COMMON_INVALID_INPUT;

    // Map geniex_GenerationConfig -> geniex::GenerationConfig
    GenerationConfig gen_cfg{};
    if (input->config) {
        gen_cfg.max_tokens = input->config->max_tokens > 0 ? input->config->max_tokens : 512;
        if (input->config->sampler_config) {
            gen_cfg.temperature = input->config->sampler_config->temperature;
            gen_cfg.top_p       = input->config->sampler_config->top_p;
        }
    }
    gen_cfg.thinking_mode = enable_thinking_;

    // Wrap token callback
    std::function<bool(const char*)> on_token_fn;
    if (input->on_token) {
        auto cb     = input->on_token;
        auto ud     = input->user_data;
        on_token_fn = [cb, ud](const char* token) -> bool { return cb(token, ud); };
    }

    // Generate
    GenerateResult result = pipeline_->generate(input->prompt_utf8, gen_cfg, on_token_fn);

    // Map result to output
    output->full_text = portable_strdup(result.full_text.c_str());
    if (!output->full_text) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;

    // Profile data (convert ms -> us)
    output->profile_data.ttft             = static_cast<int64_t>(result.ttft_ms * 1000.0);
    output->profile_data.prompt_time      = output->profile_data.ttft;  // approximate
    output->profile_data.decode_time      = static_cast<int64_t>(result.decode_ms * 1000.0);
    output->profile_data.prompt_tokens    = result.prompt_tokens;
    output->profile_data.generated_tokens = result.generated_tokens;
    output->profile_data.decoding_speed   = result.tokens_per_second;
    output->profile_data.prefill_speed =
        result.prompt_tokens > 0 && result.ttft_ms > 0.0 ? result.prompt_tokens / (result.ttft_ms / 1000.0) : 0.0;

    // Stop reason (string must be static/persistent)
    static const char* kStopEos    = "eos";
    static const char* kStopLength = "length";
    static const char* kStopUser   = "user";
    if (result.stop_reason == "eos")
        output->profile_data.stop_reason = kStopEos;
    else if (result.stop_reason == "length")
        output->profile_data.stop_reason = kStopLength;
    else if (result.stop_reason == "user")
        output->profile_data.stop_reason = kStopUser;
    else
        output->profile_data.stop_reason = kStopEos;

    return GENIEX_SUCCESS;
}

}  // namespace geniex
