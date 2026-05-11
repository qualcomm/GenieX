// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "vlm.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#if defined(_WIN32)
#define portable_strdup _strdup
#else
#define portable_strdup strdup
#endif

#include "geniex-proc/types.h"  // ChatMessage, MMContent, Role::, Modality::
#include "logging.h"
#include "pipeline/vlm_pipeline.h"
#include "qnn_runtime_utils.h"
#include "types.h"
#include "vlm_model_registry.h"  // vlm_model_registry()

namespace fs = std::filesystem;

namespace geniex {

namespace {
// Default system prompt prepended on the first turn when the caller does not include
// a `system` role message in the chat history.
constexpr const char* kDefaultSystemPrompt = "You are a helpful AI assistant.";
}  // namespace

QairtVlm::~QairtVlm() = default;

int32_t QairtVlm::create_impl(const geniex_VlmCreateInput* input) {
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

    // Look up model in VLM registry
    auto& registry = vlm_model_registry();
    auto  it       = registry.find(model_name_);
    if (it == registry.end()) {
        GENIEX_LOG_ERROR("Unknown QAIRT VLM model name: {}", model_name_);
        return GENIEX_ERROR_COMMON_MODEL_INVALID;
    }

    // Derive model directory from the model_path
    fs::path model_path(input->model_path);
    fs::path model_dir = model_path.parent_path();

    QnnRuntimeConfig runtime_cfg = qairt::runtime::make_qnn_runtime_config(model_dir);

    // Resolve vision encoder path first so we can exclude it from LLM shards.
    std::string resolved_vision_bin;
    if (input->mmproj_path && input->mmproj_path[0] != '\0') {
        resolved_vision_bin = input->mmproj_path;
    } else {
        resolved_vision_bin = qairt::runtime::find_optional_file(model_dir, "vision_encoder.bin");
    }

    // ── LLM config ────────────────────────────────────────────────────────────
    ModelConfig llm_cfg{};

    auto bin_shards = qairt::runtime::collect_bin_files(model_dir);
    if (!resolved_vision_bin.empty()) {
        const fs::path vision_path(resolved_vision_bin);
        bin_shards.erase(std::remove_if(bin_shards.begin(),
                             bin_shards.end(),
                             [&](const std::string& p) {
                                 std::error_code ec;
                                 if (fs::equivalent(fs::path(p), vision_path, ec)) return true;
                                 return fs::path(p).filename() == vision_path.filename();
                             }),
            bin_shards.end());
    }
    if (bin_shards.empty()) {
        GENIEX_LOG_ERROR("No .bin LLM shards found in: {}", model_dir.string());
        return GENIEX_ERROR_COMMON_FILE_NOT_FOUND;
    }
    GENIEX_LOG_DEBUG("Found {} LLM shards in {}", bin_shards.size(), model_dir.string());
    llm_cfg.model_paths = std::move(bin_shards);

    // Tokenizer
    if (input->tokenizer_path && input->tokenizer_path[0] != '\0') {
        llm_cfg.tokenizer_path = input->tokenizer_path;
    } else {
        llm_cfg.tokenizer_path = qairt::runtime::find_optional_file(model_dir, "tokenizer.json");
    }
    if (llm_cfg.tokenizer_path.empty()) {
        GENIEX_LOG_ERROR("tokenizer.json not found in: {}", model_dir.string());
        return GENIEX_ERROR_COMMON_FILE_NOT_FOUND;
    }

    // Embedding table (optional - AI Hub models do embedding on-device)
    llm_cfg.embedding_path = qairt::runtime::find_optional_file(model_dir, "embedding_weights.raw");
    if (llm_cfg.embedding_path.empty()) {
        llm_cfg.embedding_path = qairt::runtime::find_optional_file(model_dir, "embed_tokens.npy");
    }
    llm_cfg.htp_config_path = qairt::runtime::find_optional_file(model_dir, "htp_backend_ext_config.json");

    // ── Vision encoder config ─────────────────────────────────────────────────
    ModelConfig vision_cfg{};

    if (!resolved_vision_bin.empty()) {
        vision_cfg.model_paths = {resolved_vision_bin};
    }
    if (vision_cfg.model_paths.empty()) {
        GENIEX_LOG_WARN("No vision encoder found for VLM model '{}' — text-only mode", model_name_);
    }
    vision_cfg.htp_config_path = llm_cfg.htp_config_path;

    // ── Build VLMConfig and create pipeline ───────────────────────────────────
    VLMConfig vlm_cfg{};
    vlm_cfg.llm_config    = std::move(llm_cfg);
    vlm_cfg.vision_config = std::move(vision_cfg);

    auto pipe = it->second.make_pipeline(runtime_cfg, vlm_cfg);
    if (!pipe) {
        GENIEX_LOG_ERROR("Failed to create QAIRT VLM pipeline for model: {}", model_name_);
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
    pipeline_ = std::make_unique<VLMPipeline>(std::move(*pipe));

    GENIEX_LOG_DEBUG("QAIRT VLM created successfully: model={}", model_name_);
    return GENIEX_SUCCESS;
}

int32_t QairtVlm::reset() {
    if (!pipeline_) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
    pipeline_->reset();
    history_size_         = 0;
    pending_history_size_ = 0;
    return GENIEX_SUCCESS;
}

int32_t QairtVlm::apply_chat_template(
    const geniex_VlmApplyChatTemplateInput* input, geniex_VlmApplyChatTemplateOutput* output) {
    if (!pipeline_) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
    if (!input || !output) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    if (!input->messages || input->message_count <= 0) return GENIEX_ERROR_COMMON_INVALID_INPUT;

    // Convert C API messages → geniex::ChatMessage
    std::vector<ChatMessage> messages;
    messages.reserve(static_cast<size_t>(input->message_count));

    for (int32_t i = 0; i < input->message_count; ++i) {
        const geniex_VlmChatMessage& src = input->messages[i];

        ChatMessage msg{};

        // Map role string → MessageRole
        if (!src.role || std::strcmp(src.role, "user") == 0) {
            msg.role = Role::User;
        } else if (std::strcmp(src.role, "assistant") == 0) {
            msg.role = Role::Assistant;
        } else if (std::strcmp(src.role, "system") == 0) {
            msg.role = Role::System;
        } else {
            GENIEX_LOG_WARN("Unknown VLM message role '{}', treating as user", src.role);
            msg.role = Role::User;
        }

        // Map content items
        for (int64_t j = 0; j < src.content_count; ++j) {
            const geniex_VlmContent& c = src.contents[j];
            if (!c.type) continue;

            if (std::strcmp(c.type, "text") == 0) {
                if (c.text) msg.content += c.text;
            } else if (std::strcmp(c.type, "image") == 0) {
                if (c.text) msg.mm_content.push_back({Modality::Image, std::string(c.text)});
            } else if (std::strcmp(c.type, "audio") == 0) {
                if (c.text) msg.mm_content.push_back({Modality::Audio, std::string(c.text)});
            } else {
                GENIEX_LOG_WARN("Unknown VLM content type '{}', skipping", c.type);
            }
        }

        messages.push_back(std::move(msg));
    }

    // If the caller passed fewer messages than we've already committed, they implicitly
    // reset the conversation without calling reset() — treat it as a hard reset.
    if (messages.size() <= history_size_) {
        GENIEX_LOG_WARN(
            "VLM history shrank ({} → {}) without reset() — resetting KV cache", history_size_, messages.size());
        pipeline_->reset();
        history_size_         = 0;
        pending_history_size_ = 0;
    }

    // Slice out only the new messages since the last committed generate().
    std::vector<ChatMessage> new_messages(messages.begin() + static_cast<ptrdiff_t>(history_size_), messages.end());

    // On the first turn, ensure there is a system prompt at the front. If the caller did
    // not supply one, inject the default. Subsequent turns reuse the already-cached system.
    if (history_size_ == 0) {
        const bool has_system = !new_messages.empty() && new_messages.front().role == Role::System;
        if (!has_system) {
            ChatMessage sys{};
            sys.role    = Role::System;
            sys.content = kDefaultSystemPrompt;
            new_messages.insert(new_messages.begin(), std::move(sys));
        }
    }

    // Record pending size — committed to history_size_ only after a successful generate().
    pending_history_size_ = messages.size();

    std::string formatted = pipeline_->applyChatTemplate(new_messages, /*add_generation_prompt=*/true);

    output->formatted_text = portable_strdup(formatted.c_str());
    if (!output->formatted_text) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;

    return GENIEX_SUCCESS;
}

int32_t QairtVlm::generate(const geniex_VlmGenerateInput* input, geniex_VlmGenerateOutput* output) {
    if (!pipeline_) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
    if (!input || !output) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    if (!input->prompt_utf8) return GENIEX_ERROR_COMMON_INVALID_INPUT;

    // Collect image paths from GenerationConfig (same convention as llama_cpp plugin)
    std::vector<std::string> image_paths;
    if (input->config && input->config->image_paths && input->config->image_count > 0) {
        image_paths.reserve(static_cast<size_t>(input->config->image_count));
        for (int32_t i = 0; i < input->config->image_count; ++i) {
            if (input->config->image_paths[i]) {
                image_paths.emplace_back(input->config->image_paths[i]);
            }
        }
    }

    // Map geniex_GenerationConfig → geniex::GenerationConfig
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

    // Commit pending history size before running — this turn is now in the KV cache.
    history_size_ = pending_history_size_ + 1;

    // Run VLM pipeline (incremental — only new messages since last generate() are in the prompt)
    GenerateResult result;
    try {
        result = pipeline_->generate(input->prompt_utf8, image_paths, gen_cfg, on_token_fn);
    } catch (const ContextLengthExceededError& e) {
        GENIEX_LOG_ERROR("QAIRT VLM generate: context length exceeded: {}", e.what());
        return GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
    }

    // Map result to output
    output->full_text = portable_strdup(result.full_text.c_str());
    if (!output->full_text) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;

    // Profile data (convert ms → µs to match geniex_ProfileData convention)
    output->profile_data.ttft             = static_cast<int64_t>(result.ttft_ms * 1000.0);
    output->profile_data.prompt_time      = output->profile_data.ttft;
    output->profile_data.decode_time      = static_cast<int64_t>(result.decode_ms * 1000.0);
    output->profile_data.prompt_tokens    = result.prompt_tokens;
    output->profile_data.generated_tokens = result.generated_tokens;
    output->profile_data.decoding_speed   = result.tokens_per_second;
    output->profile_data.prefill_speed    = (result.prompt_tokens > 0 && result.ttft_ms > 0.0)
                                                ? static_cast<double>(result.prompt_tokens) / (result.ttft_ms / 1000.0)
                                                : 0.0;

    // Stop reason
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