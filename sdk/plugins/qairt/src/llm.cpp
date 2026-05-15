#include "llm.h"

#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define portable_strdup _strdup
#else
#define portable_strdup strdup
#endif

#include "external/json.hpp"
#include "llm_model_registry.h"  // provided by geniex-qairt/models/
#include "logging.h"
#include "pipeline/chat_template.h"
#include "pipeline/llm_pipeline.h"
#include "qnn_runtime_utils.h"
#include "sampler_config_utils.h"
#include "types.h"

namespace fs = std::filesystem;

namespace geniex {

namespace {
// Default system prompt used on the first turn when the caller does not supply one
// via a `system` role chat message.
constexpr const char* kDefaultSystemPrompt = "You are a helpful AI assistant.";
}  // namespace

QairtLlm::~QairtLlm() = default;

int32_t QairtLlm::create_impl(const geniex_LlmCreateInput* input) {
    if (!input || !input->model_name || !input->model_path) {
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    model_name_      = input->model_name;
    enable_thinking_ = input->config.enable_thinking;

    // Reject llama.cpp-only parameters that have no meaning in the QAIRT plugin
    if (input->config.n_gpu_layers != 0) {
        GENIEX_LOG_ERROR("--ngl (n_gpu_layers) is not supported by the qairt plugin");
        return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }
    if (input->config.n_ctx != 0) {
        GENIEX_LOG_ERROR("--nctx (n_ctx) is not supported by the qairt plugin");
        return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }

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
    model_cfg.embedding_path = qairt::runtime::find_optional_file(model_dir, "embedding_weights.raw");
    if (model_cfg.embedding_path.empty()) {
        model_cfg.embedding_path = qairt::runtime::find_optional_file(model_dir, "embed_tokens.npy");
    }

    // HTP backend config
    model_cfg.htp_config_path = qairt::runtime::find_optional_file(model_dir, "htp_backend_ext_config.json");

    // Create LLMPipeline via the per-model factory (handles makeModel + chat template internally).
    // Returns std::nullopt on QNN init failure, missing tokenizer, etc.
    auto pipe = entry.make_pipeline(runtime_cfg, model_cfg);
    if (!pipe) {
        GENIEX_LOG_ERROR("Failed to create QAIRT LLM pipeline for model: {}", model_name_);
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
    pipeline_      = std::make_unique<LLMPipeline>(std::move(*pipe));
    is_first_turn_ = true;

    GENIEX_LOG_DEBUG("QAIRT LLM created successfully: model={}", model_name_);
    return GENIEX_SUCCESS;
}

int32_t QairtLlm::reset() {
    if (!pipeline_) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
    pipeline_->reset();
    is_first_turn_ = true;
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

    // Extract the last user message (and the system message, if this is the first turn).
    const char* user_message  = nullptr;
    const char* system_prompt = nullptr;
    for (int32_t i = input->message_count - 1; i >= 0; --i) {
        const auto& m = input->messages[i];
        if (!m.role) continue;
        if (!user_message && std::strcmp(m.role, "user") == 0) {
            user_message = m.content;
        } else if (is_first_turn_ && !system_prompt && std::strcmp(m.role, "system") == 0) {
            system_prompt = m.content;
        }
        if (user_message && (!is_first_turn_ || system_prompt)) break;
    }

    if (!user_message) {
        GENIEX_LOG_ERROR("No user message found in chat messages");
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    if (is_first_turn_) {
        const char* sp = (system_prompt && system_prompt[0] != '\0') ? system_prompt : kDefaultSystemPrompt;
        pipeline_->setSystemPrompt(sp);
    }

    // Tools may change between turns (dynamic tool registration), so stage
    // unconditionally — not gated by is_first_turn_. Parse OAI-compat tools
    // JSON here at the plugin boundary; the qairt core sees only typed
    // ChatTool entries and never raw JSON.
    if (input->tools && input->tools[0] != '\0') {
        try {
            auto j = nlohmann::ordered_json::parse(input->tools);
            if (!j.is_array()) {
                GENIEX_LOG_ERROR("tools JSON must be an array");
                return GENIEX_ERROR_COMMON_INVALID_INPUT;
            }
            ChatTools tools;
            tools.reserve(j.size());
            for (const auto& el : j) {
                // OAI compat wraps each entry as {"type":"function","function":{...}};
                // tolerate the un-wrapped form too.
                const auto& fn = (el.is_object() && el.contains("function")) ? el["function"] : el;
                if (!fn.is_object()) continue;
                ChatTool t;
                t.name        = fn.value("name", std::string{});
                t.description = fn.value("description", std::string{});
                if (fn.contains("parameters")) {
                    t.parameters_json = fn["parameters"].dump();
                }
                if (!t.name.empty()) tools.push_back(std::move(t));
            }
            pipeline_->setTools(std::move(tools));
        } catch (const std::exception& e) {
            GENIEX_LOG_ERROR("Failed to parse tools JSON: {}", e.what());
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
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

    // Reject llama.cpp-only parameters that have no meaning in the QAIRT plugin
    if (input->input_ids && input->input_ids_count > 0) {
        GENIEX_LOG_ERROR("--token-file (input_ids) is not supported by the qairt plugin");
        return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }
    if (input->config && input->config->stop && input->config->stop_count > 0) {
        GENIEX_LOG_ERROR("--stop / --stop-file (stop sequences) is not supported by the qairt plugin");
        return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }

    if (!input->prompt_utf8) return GENIEX_ERROR_COMMON_INVALID_INPUT;

    // Map geniex_GenerationConfig -> geniex::GenerationConfig
    GenerationConfig gen_cfg{};
    if (input->config) {
        gen_cfg.max_tokens = input->config->max_tokens > 0 ? input->config->max_tokens : 512;
        qairt::apply_sampler_config(input->config->sampler_config, gen_cfg);
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
    GenerateResult result;
    try {
        result = pipeline_->generate(input->prompt_utf8, gen_cfg, on_token_fn);
    } catch (const ContextLengthExceededError& e) {
        GENIEX_LOG_ERROR("QAIRT generate: context length exceeded: {}", e.what());
        return GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
    }
    is_first_turn_ = false;

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
