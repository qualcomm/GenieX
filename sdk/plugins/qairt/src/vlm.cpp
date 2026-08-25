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

#include "dispatch.h"               // provided by geniex-qairt/models/
#include "geniex-proc/tokenizer.h"  // ApplyChatTemplateOptions
#include "geniex-proc/types.h"      // ChatMessage, MMContent, Role::, Modality::
#include "llm/llm_spec_loader.h"    // parseGenieSamplerConfig
#include "logging.h"
#include "path_utils.h"
#include "pipeline/vlm_pipeline.h"
#include "qnn_runtime_utils.h"
#include "sampler_config_utils.h"
#include "types.h"

namespace fs = std::filesystem;

namespace geniex {

namespace {
// Default system prompt prepended on the first turn when the caller does not include
// a `system` role message in the chat history.
constexpr const char* kDefaultSystemPrompt = "You are a helpful AI assistant.";
}  // namespace

QairtVlm::~QairtVlm() = default;

int32_t QairtVlm::create(const geniex_VlmCreateInput* input) {
    if (!input || !input->model_path) {
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    // Reject llama.cpp-only parameters that have no meaning in the QAIRT plugin
    if (input->config.n_gpu_layers != 0) {
        GENIEX_LOG_ERROR("--ngl (n_gpu_layers) is not supported by the qairt plugin");
        return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }
    if (input->config.n_ctx != 0) {
        GENIEX_LOG_ERROR("--nctx (n_ctx) is not supported by the qairt plugin");
        return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }

    // Derive model directory from the model_path
    fs::path model_path(input->model_path);
    fs::path model_dir = model_path.parent_path();

    bundle_sampler_ = parseGenieSamplerConfig(model_dir);

    QnnRuntimeConfig runtime_cfg = qairt::runtime::make_qnn_runtime_config(model_dir);

    // Resolve vision encoder path first so we can exclude it from LLM shards.
    std::string resolved_vision_bin;
    if (input->mmproj_path && input->mmproj_path[0] != '\0') {
        resolved_vision_bin = input->mmproj_path;
    } else {
        resolved_vision_bin = qairt::runtime::find_optional_file(model_dir, "vision_encoder.bin").value_or("");
    }

    // ── LLM config ────────────────────────────────────────────────────────────
    // Bundle layout comes from the core's `ctx-bins` handling, not a `*.bin` glob.
    // See the note in llm.cpp.
    ModelConfig llm_cfg{};
    try {
        llm_cfg = modelConfigFromDirectory(model_dir);
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("Failed to resolve QAIRT bundle layout in {}: {}", model_dir.string(), e.what());
        return GENIEX_ERROR_COMMON_FILE_NOT_FOUND;
    }

    // The vision encoder is driven as its own graph, so keep it out of the LLM
    // shard list even if ctx-bins lists it.
    if (!resolved_vision_bin.empty()) {
        auto&          shards = llm_cfg.model_paths;
        const fs::path vision_path(resolved_vision_bin);
        shards.erase(std::remove_if(shards.begin(),
                         shards.end(),
                         [&](const std::string& p) {
                             std::error_code ec;
                             if (fs::equivalent(fs::path(p), vision_path, ec)) return true;
                             return fs::path(p).filename() == vision_path.filename();
                         }),
            shards.end());
    }
    if (llm_cfg.model_paths.empty()) {
        GENIEX_LOG_ERROR("No .bin LLM shards found in: {}", model_dir.string());
        return GENIEX_ERROR_COMMON_FILE_NOT_FOUND;
    }
    GENIEX_LOG_DEBUG("Found {} LLM shards in {}", llm_cfg.model_paths.size(), model_dir.string());

    // Tokenizer: an explicit caller override wins over the bundle's own.
    if (input->tokenizer_path && input->tokenizer_path[0] != '\0') {
        llm_cfg.tokenizer_path = input->tokenizer_path;
    }
    if (llm_cfg.tokenizer_path.empty()) {
        GENIEX_LOG_ERROR("tokenizer.json not found in: {}", model_dir.string());
        return GENIEX_ERROR_COMMON_FILE_NOT_FOUND;
    }

    // Embedding table (nullopt — AI Hub models do embedding on-device).
    llm_cfg.embedding_path = qairt::runtime::find_optional_file(model_dir, "embedding_weights.raw");
    if (!llm_cfg.embedding_path) {
        llm_cfg.embedding_path = qairt::runtime::find_optional_file(model_dir, "embed_tokens.npy");
    }
    // htp_config_path is already resolved by modelConfigFromDirectory.

    // ── Vision encoder config ─────────────────────────────────────────────────
    ModelConfig vision_cfg{};

    if (!resolved_vision_bin.empty()) {
        vision_cfg.model_paths = {resolved_vision_bin};
    }
    if (vision_cfg.model_paths.empty()) {
        GENIEX_LOG_WARN("No vision encoder found in {} — text-only mode", model_dir.string());
    }
    has_vision_encoder_        = !vision_cfg.model_paths.empty();
    vision_cfg.htp_config_path = llm_cfg.htp_config_path;

    // ── Build VLMConfig and create pipeline ───────────────────────────────────
    VLMConfig vlm_cfg{};
    vlm_cfg.llm_config    = std::move(llm_cfg);
    vlm_cfg.vision_config = std::move(vision_cfg);

    // Dispatcher reads metadata.json `model_id` from the bundle and routes to
    // the matching VLM family factory (currently qwen2_5_vl_*).
    auto pipe = makeVLMPipeline(runtime_cfg, vlm_cfg);
    if (!pipe) {
        GENIEX_LOG_ERROR("Failed to create QAIRT VLM pipeline from bundle: {}", model_dir.string());
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
    pipeline_ = std::make_unique<VLMPipeline>(std::move(*pipe));

    GENIEX_LOG_DEBUG("QAIRT VLM created successfully from bundle: {}", model_dir.string());
    return GENIEX_SUCCESS;
}

int32_t QairtVlm::get_capabilities(geniex_VlmCapabilities* output) {
    if (!output) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    output->supports_vision = has_vision_encoder_;
    output->supports_audio  = false;
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

    // Tools are injected on the first turn only, matching the LLM path.
    ApplyChatTemplateOptions opts;
    opts.add_generation_prompt = true;
    if (history_size_ == 0 && input->tools && input->tools[0] != '\0') {
        opts.tools_json = input->tools;
    }
    opts.enable_thinking = input->enable_thinking;

    std::string formatted;
    try {
        formatted = pipeline_->applyChatTemplate(new_messages, opts);
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("applyChatTemplate failed: {}", e.what());
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

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
                image_paths.emplace_back(qairt::to_loadable_path(input->config->image_paths[i]));
            }
        }
    }

    // Map geniex_GenerationConfig → geniex::GenerationConfig
    GenerationConfig gen_cfg{};
    if (input->config) {
        gen_cfg.max_tokens = input->config->max_tokens > 0 ? input->config->max_tokens : 512;
        qairt::apply_sampler_config(input->config->sampler_config, gen_cfg, bundle_sampler_);
    }

    // Wrap token callback
    std::function<bool(const char*)> on_token_fn;
    if (input->on_token) {
        auto cb     = input->on_token;
        auto ud     = input->user_data;
        on_token_fn = [cb, ud](const char* token) -> bool { return cb(token, ud); };
    }

    // Commit pending history size before running — this turn is now in the KV cache.
    history_size_ = pending_history_size_ + 1;

    // Run VLM pipeline (incremental — only new messages since last generate() are in the prompt).
    GenerateResult result = pipeline_->generate(input->prompt_utf8, image_paths, gen_cfg, on_token_fn);

    // Map result to output
    output->full_text = portable_strdup(result.full_text.c_str());
    if (!output->full_text) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;

    // Profile data (ms → µs). media_time is the encoder only; the media-token
    // prefill stays in prompt_time, derived as ttft − encoder since QAIRT exposes
    // no separate prefill number. When the encoder dominates ttft the subtraction
    // clamps to 0, so prefill_speed reports 0 (unknown) rather than a bogus rate.
    const int64_t media_us  = static_cast<int64_t>(result.media_ms * 1000.0);
    const int64_t prompt_us = std::max<int64_t>(static_cast<int64_t>(result.ttft_ms * 1000.0) - media_us, 0);

    output->profile_data.ttft             = static_cast<int64_t>(result.ttft_ms * 1000.0);
    output->profile_data.media_time       = media_us;
    output->profile_data.prompt_time      = prompt_us;
    output->profile_data.decode_time      = static_cast<int64_t>(result.decode_ms * 1000.0);
    output->profile_data.prompt_tokens    = result.prompt_tokens;
    output->profile_data.generated_tokens = result.generated_tokens;
    output->profile_data.prefill_speed    = prompt_us > 0 ? result.prompt_tokens / (prompt_us / 1e6) : 0.0;
    output->profile_data.decoding_speed   = result.tokens_per_second;

    // Stop Reason
    if (result.stop_reason == "user") {
        output->profile_data.stop_reason = "user";
    } else if (result.stop_reason == "length") {
        output->profile_data.stop_reason = "length";
    } else if (result.stop_reason == "context_length") {
        output->profile_data.stop_reason = "length";
        GENIEX_LOG_WARN("QAIRT VLM generate: context length exceeded (partial result populated)");
        return GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
    } else if (result.stop_reason == "prompt_too_long") {
        output->profile_data.stop_reason = "length";
        GENIEX_LOG_WARN("QAIRT VLM generate: prompt exceeds max context length");
        return GENIEX_ERROR_LLM_GENERATION_PROMPT_TOO_LONG;
    } else if (result.stop_reason == "error") {
        output->profile_data.stop_reason = "eos";
        GENIEX_LOG_ERROR("QAIRT VLM generate failed during prompt processing (empty result)");
        return GENIEX_ERROR_VLM_GENERATION_FAILED;
    } else {
        output->profile_data.stop_reason = "eos";
    }
    return GENIEX_SUCCESS;
}

}  // namespace geniex
