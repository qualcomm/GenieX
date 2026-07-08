// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "llm.h"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <vector>

#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "htp_session.h"
#include "logging.h"
#include "params.h"
#include "profiler.h"

namespace geniex {

LlamaLlm::~LlamaLlm() {
    if (sampler) common_sampler_free(sampler);
    if (ctx) llama_free(ctx);
    if (model) llama_model_free(model);
    // pools_ frees its threadpools in its own destructor, after ctx is freed.
}

int32_t LlamaLlm::create_impl(const geniex_LlmCreateInput* input) {
    if (!input || !input->model_path) {
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    const Device              device = classify_device(input->device_id, input->config.n_gpu_layers);
    const geniex_ModelConfig& config = input->config;
    llama_model_params        mpar   = build_model_params(config, device);

    // MoE override + null terminator; must outlive the load_from_file call below.
    llama_model_tensor_buft_override tensor_overrides[2];

    // FIX: HTP backend patch
    { htp::reacquire_before_load(); }

    // FIX: gpt oss offload patch
    {
        std::string model_path_lower(input->model_path);
        std::transform(model_path_lower.begin(), model_path_lower.end(), model_path_lower.begin(), ::tolower);
        bool is_gpt_oss_model =
            (model_path_lower.find("gpt") != std::string::npos) && (model_path_lower.find("oss") != std::string::npos);

        this->allow_special_tokens = is_gpt_oss_model;
        if (is_gpt_oss_model) {
            tensor_overrides[0]        = {"\\.ffn_(up|down|gate)_exps\\.(weight|bias)", ggml_backend_cpu_buffer_type()};
            tensor_overrides[1]        = {nullptr, nullptr};  // Null terminator
            mpar.tensor_buft_overrides = tensor_overrides;
            GENIEX_LOG_INFO(
                "GPT OSS model detected - MoE expert tensors "
                "(ffn_*_exps.weight/bias) will be forced to CPU");
        } else {
            mpar.tensor_buft_overrides = nullptr;
        }
    }

    auto selection = resolve_devices(input->device_id);
    if (!selection) {
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }
    if (!selection->empty()) {
        mpar.devices = selection->data();
    }

    // FIX: HTP backend patch
    {
        if (htp::htp_backend_present()) {
            htp_guard_.mark_htp();
        }
    }

    this->model = llama_model_load_from_file(input->model_path, mpar);
    if (!this->model) {
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }

    llama_context_params cpar = build_context_params(config, /*n_ctx_default=*/4096, device);
    this->ctx                 = llama_init_from_model(this->model, cpar);
    if (!this->ctx) {
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }

    ggml_threadpool_params tpp_main  = build_threadpool_params(cpar.n_threads, device);
    ggml_threadpool_params tpp_batch = build_threadpool_params(cpar.n_threads_batch, device);
    int32_t                tp_ret    = this->pools_.attach(this->ctx, tpp_main, tpp_batch);
    if (tp_ret != GENIEX_SUCCESS) {
        return tp_ret;
    }

    // Load chat template if path is provided
    if (config.chat_template_content) {
        try {
            std::string content(config.chat_template_content);
            this->chat_template_str.emplace(content);
        } catch (const std::exception& e) {
        }
    } else if (config.chat_template_path) {
        std::ifstream file(config.chat_template_path);
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            this->chat_template_str = buffer.str();
            file.close();
        }
    }

    this->reset();
    this->set_sampler(nullptr);

    return GENIEX_SUCCESS;
}

int32_t LlamaLlm::reset() {
    this->n_past_global = 0;
    this->n_past        = 0;
    this->past_prompt_tokens.clear();

    llama_memory_clear(llama_get_memory(this->ctx), /*clear data=*/true);

    return GENIEX_SUCCESS;
}

int32_t LlamaLlm::save_kv_cache(const geniex_KvCacheSaveInput* input, geniex_KvCacheSaveOutput* _) {
    return llama_state_save_file(this->ctx, input->path, nullptr, 0) ? GENIEX_SUCCESS : GENIEX_ERROR_COMMON_UNKNOWN;
}

int32_t LlamaLlm::load_kv_cache(const geniex_KvCacheLoadInput* input, geniex_KvCacheLoadOutput* _) {
    size_t  out;
    int32_t ret = llama_state_load_file(this->ctx, input->path, nullptr, 0, &out);

    // get KV cache size from llama memory
    llama_memory_t mem     = llama_get_memory(this->ctx);
    llama_pos      pos_min = llama_memory_seq_pos_min(mem, 0);
    llama_pos      pos_max = llama_memory_seq_pos_max(mem, 0);

    int32_t n_past = 0;
    if (pos_min >= 0 && pos_max >= 0) {
        n_past = pos_max - pos_min + 1;
    }

    this->n_past        = n_past;
    this->n_past_global = 0;
    this->past_prompt_tokens.clear();

    return ret ? GENIEX_SUCCESS : GENIEX_ERROR_COMMON_UNKNOWN;
}

int32_t LlamaLlm::apply_chat_template(
    const geniex_LlmApplyChatTemplateInput* input, geniex_LlmApplyChatTemplateOutput* output) {
    if (!input || !input->messages || !output || input->message_count <= 0) {
        return GENIEX_ERROR_COMMON_INVALID_INPUT;  // error: invalid input
    }

    // Convert geniex_ChatMessage to common_chat_msg
    std::vector<common_chat_msg> common_messages;
    common_messages.reserve(input->message_count);

    for (int32_t i = 0; i < input->message_count; ++i) {
        common_chat_msg msg;
        msg.role    = input->messages[i].role;
        msg.content = input->messages[i].content;
        common_messages.push_back(msg);
    }

    // Initialize chat templates
    // Always pass the model, let chat_template_override handle template selection
    std::string               template_override = this->chat_template_str ? *this->chat_template_str : "";
    common_chat_templates_ptr tmpls             = common_chat_templates_init(this->model, template_override, "", "");

    // Set up inputs
    common_chat_templates_inputs inputs;
    inputs.use_jinja             = true;
    inputs.messages              = common_messages;
    inputs.add_generation_prompt = input->add_generation_prompt;

    if (input->tools && strlen(input->tools) > 0) {
        inputs.tools = common_chat_tools_parse_oaicompat(nlohmann::ordered_json::parse(std::string(input->tools)));
    }

    inputs.enable_thinking = input->enable_thinking;

    // Apply chat template
    auto result = common_chat_templates_apply(tmpls.get(), inputs);

    output->formatted_text = strdup(result.prompt.c_str());
    if (!output->formatted_text) {
        return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;  // error: memory allocation failed
    }
    return GENIEX_SUCCESS;
}

int32_t LlamaLlm::generate(const geniex_LlmGenerateInput* input, geniex_LlmGenerateOutput* output) {
    // Validate input
    if (!input) return GENIEX_ERROR_COMMON_INVALID_INPUT;

    bool has_input_ids   = input->input_ids != nullptr && input->input_ids_count > 0;
    bool has_prompt_utf8 = input->prompt_utf8 != nullptr;

    if (!has_input_ids && !has_prompt_utf8)
        return GENIEX_ERROR_COMMON_INVALID_INPUT;  // error: neither input_ids nor prompt_utf8 provided

    geniex_GenerationConfig cfg = input->config ? *input->config : geniex_GenerationConfig{};
    cfg.max_tokens              = cfg.max_tokens > 0 ? cfg.max_tokens : 128;

    // Initialzie resources
    this->set_sampler(cfg.sampler_config);
    auto*              mem       = llama_get_memory(this->ctx);
    const llama_vocab* vocab     = llama_model_get_vocab(this->model);
    const int          n_ctx     = llama_n_ctx(this->ctx);
    const int          n_batch   = llama_n_batch(this->ctx);
    const bool         can_shift = llama_memory_can_shift(mem) && !llama_model_is_recurrent(this->model);

    // Encode the full prompt (either from input_ids or prompt_utf8)
    std::vector<llama_token> prompt_ids;
    if (has_input_ids) {
        const int32_t vocab_size = llama_vocab_n_tokens(vocab);
        // Validate token IDs are within vocabulary range
        for (int32_t i = 0; i < input->input_ids_count; i++) {
            if (input->input_ids[i] < 0 || input->input_ids[i] >= vocab_size) {
                GENIEX_LOG_ERROR("token ID out of range: {}", input->input_ids[i]);
                return GENIEX_ERROR_COMMON_INVALID_INPUT;  // error: token ID out of vocabulary range
            }
        }

        prompt_ids.assign(input->input_ids, input->input_ids + input->input_ids_count);
    } else {
        // Use text tokenization path
        try {
            prompt_ids = common_tokenize(vocab, std::string(input->prompt_utf8), true, true);
        } catch (const std::exception& e) {
            return GENIEX_ERROR_LLM_TOKENIZATION_FAILED;  // error: prompt encoding failed
        }
    }

    // Prefix Match

    int32_t prompt_len = static_cast<int32_t>(prompt_ids.size());
    int     match_len  = 0;
    while (match_len < std::min((int)past_prompt_tokens.size(), prompt_len) &&
           past_prompt_tokens[match_len] == prompt_ids[match_len]) {
        match_len++;
    }
    GENIEX_LOG_DEBUG(
        "prefix match: past_prompt_tokens size: {}, prompt_len: {}, "
        "match_len: {}",
        past_prompt_tokens.size(),
        prompt_len,
        match_len);

    if (match_len < (int)this->past_prompt_tokens.size()) {
        if (match_len < this->n_past_global - this->n_past) {
            // match out of kvcache, need reset
            llama_memory_seq_rm(mem, 0, 0, this->n_past);
            this->n_past        = 0;
            this->n_past_global = prompt_len > n_ctx - 4 ? n_ctx - 4 : 0;
            GENIEX_LOG_INFO("prefix match: n_past_global rollback to: {}", this->n_past_global);
        } else {
            // match in kvcache, need rollback
            llama_memory_seq_rm(mem, 0, match_len, this->n_past - match_len);
            this->n_past        = match_len;
            this->n_past_global = match_len;
            GENIEX_LOG_INFO("prefix match: n_past_global rollback to: {}", this->n_past_global);
        }
    }

    std::vector<llama_token> embd_inp(prompt_ids.begin() + this->n_past_global, prompt_ids.end());

    // Main loop

    int32_t          res = GENIEX_SUCCESS;
    common::Profiler profiler;
    profiler.prompt_start();

    // Discard tokens past the first n_keep to fit n_fit more; returns the count discarded, 0 once down to n_keep.
    auto slide_window = [&](int n_fit) -> int {
        const int n_keep        = 4;
        const int n_past_before = this->n_past;
        const int needed        = this->n_past + n_fit - n_ctx + 1;
        int       n_discard     = std::max(this->n_past / 2 - n_keep, needed);
        n_discard               = std::min(n_discard, this->n_past - n_keep);
        if (n_discard <= 0) {
            return 0;
        }

        llama_memory_seq_rm(mem, 0, n_keep, n_keep + n_discard);
        llama_memory_seq_add(mem, 0, n_keep + n_discard, this->n_past, -n_discard);
        this->n_past -= n_discard;

        GENIEX_LOG_INFO(
            "Context shifting - discarding {} tokens, n_keep: {}, "
            "this->n_past before: {}, this->n_past after: {}",
            n_discard,
            n_keep,
            n_past_before,
            this->n_past);
        return n_discard;
    };

    // Decode one batch (caller chunks long inputs) and advance n_past.
    auto process = [&](const llama_token* tokens, int n_tokens) -> int32_t {
        llama_batch batch = llama_batch_get_one(const_cast<llama_token*>(tokens), n_tokens);
        // decode returns 1 when the batch does not fit; slide on demand and retry rather than gating on
        // n_past >= n_ctx, which never trips for SWA models (physical KV fills before n_past reaches n_ctx).
        int rc = llama_decode(this->ctx, batch);
        while (rc == 1 && can_shift && slide_window(n_tokens) > 0) {
            rc = llama_decode(this->ctx, batch);
        }
        switch (rc) {
            case 0:
                break;
            case 1:
                return GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
            default:
                return GENIEX_ERROR_LLM_GENERATION_FAILED;
        }
        n_past += n_tokens;
        return GENIEX_SUCCESS;
    };

    // Process input (prefilling)

    for (llama_token id : prompt_ids) {
        common_sampler_accept(this->sampler, id, /* accept_grammar= */ false);
    }

    for (int i = 0; i < (int)embd_inp.size() && res == GENIEX_SUCCESS; i += n_batch) {
        int n_eval = std::min(n_batch, (int)embd_inp.size() - i);
        res        = process(embd_inp.data() + i, n_eval);
    }

    profiler.prompt_end();
    profiler.update_prompt_tokens(prompt_len - this->n_past_global);
    profiler.decode_start();

    // Process output

    bool                     first_token_generated = false;
    std::vector<llama_token> generated_tokens;
    std::stringstream        full_text;

    while (res == GENIEX_SUCCESS && (int)generated_tokens.size() < cfg.max_tokens) {
        llama_token id = common_sampler_sample(this->sampler, this->ctx, -1);
        common_sampler_accept(this->sampler, id, /* accept_grammar= */ true);

        // Record TTFT on first token generation
        if (!first_token_generated) {
            profiler.record_ttft();
            first_token_generated = true;
            GENIEX_LOG_DEBUG("First token generated, TTFT recorded");
        }

        // Check EOS token (skipped when ignore_eos forces a full-length run)
        if (!cfg.ignore_eos && llama_vocab_is_eog(vocab, id)) {
            GENIEX_LOG_DEBUG("EOS token generated, stopping generation");
            profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_EOS);
            break;
        }

        // Convert token to string
        char token_buf[64];
        int  n = llama_token_to_piece(vocab, id, token_buf, sizeof(token_buf) - 1, 0, this->allow_special_tokens);
        if (n < 0) {
            res = GENIEX_ERROR_LLM_GENERATION_FAILED;
            break;
        }
        token_buf[n] = '\0';

        // Check stop sequences
        const bool stop_matched = std::any_of(
            cfg.stop, cfg.stop + cfg.stop_count, [&](const char* s) { return s && strcmp(token_buf, s) == 0; });
        if (stop_matched) {
            GENIEX_LOG_DEBUG("Stop sequence matched");
            profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_STOP_SEQUENCE);
            break;
        }

        generated_tokens.push_back(id);

        // Call the callback directly (UTF-8 validation is now handled at bridge
        // level)
        if (input->on_token && !input->on_token(token_buf, input->user_data)) {
            GENIEX_LOG_WARN("User callback requested stop during token generation");
            profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_USER);
            break;
        }
        full_text << token_buf;

        res = process(&id, 1);
    }

    // update output and profiler data
    if (res == GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH) {
        GENIEX_LOG_WARN("LLM generate: context window ({}) exhausted; truncating", n_ctx);
        profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_LENGTH);
    } else if ((int)generated_tokens.size() >= cfg.max_tokens) {
        profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_LENGTH);
    }
    profiler.decode_end();
    profiler.update_generated_tokens(generated_tokens.size());
    profiler.to_profile_data(output->profile_data);
    output->full_text = strdup(full_text.str().c_str());

    // update past record
    this->n_past_global = prompt_len + generated_tokens.size();
    this->past_prompt_tokens.insert(this->past_prompt_tokens.end(), embd_inp.begin(), embd_inp.end());
    this->past_prompt_tokens.insert(this->past_prompt_tokens.end(), generated_tokens.begin(), generated_tokens.end());

    return res;
}

int32_t LlamaLlm::get_model_info(geniex_LlmModelInfo* output) {
    if (!this->model) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
    const llama_vocab* vocab = llama_model_get_vocab(this->model);
    output->vocab_size       = llama_vocab_n_tokens(vocab);
    const llama_token bos    = llama_vocab_bos(vocab);
    output->bos_token        = (bos == LLAMA_TOKEN_NULL) ? -1 : static_cast<int32_t>(bos);
    output->add_bos          = llama_vocab_get_add_bos(vocab) ? 1 : 0;
    return GENIEX_SUCCESS;
}
}  // namespace geniex

// Private
namespace geniex {

void LlamaLlm::set_sampler(const geniex_SamplerConfig* cfg) {
    if (this->sampler) {
        common_sampler_free(this->sampler);
        this->sampler = nullptr;
    }
    common_params_sampling s = build_sampling_params(cfg);
    this->sampler            = common_sampler_init(this->model, s);
}

}  // namespace geniex
