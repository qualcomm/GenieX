// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "llm.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
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
    if (spec) common_speculative_free(spec);
    if (sampler) common_sampler_free(sampler);
    if (draft_ctx) llama_free(draft_ctx);
    if (draft_model) llama_model_free(draft_model);
    if (ctx) llama_free(ctx);
    if (model) llama_model_free(model);
    // pools_ frees its threadpools in its own destructor, after ctx is freed.
}

int32_t LlamaLlm::create(const geniex_LlmCreateInput* input) {
    if (!input || !input->model_path) {
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    const Device              device = classify_device(input->device_id, input->config.n_gpu_layers);
    const geniex_ModelConfig& config = input->config;
    llama_model_params        mpar   = build_model_params(config, device);

    // MoE override + null terminator; must outlive the load_from_file call below.
    llama_model_tensor_buft_override tensor_overrides[2];

    // FIX: HTP backend patch — reacquire whenever the HTP backend is present
    // in the ggml registry, regardless of the current target device. Any load
    // walks the registry's device list, and a stale session pointer left from
    // a prior release_sessions crashes the load even on cpu / gpu targets.
    {
        if (htp::htp_backend_present()) {
            htp::reacquire_before_load();
        }
    }

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

    // Resolve the compute-unit alias to a devices[] list for llama.cpp; must
    // outlive mpar (mpar.devices points into selection's buffer).
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

    // Parse the speculative config before the target context: a drafting target
    // needs its rollback snapshots and per-draft logits rows sized up front.
    std::optional<common_params_speculative> spar = build_speculative_params(config);

    llama_context_params cpar = build_context_params(config, /*n_ctx_default=*/4096, device, spar ? &*spar : nullptr);
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

    // Speculative decoding: optional, keyed on a non-empty spec_type ("none" also
    // disables). Failure is non-fatal — we log and fall back to plain decoding.
    if (spar) {
        int32_t spec_ret = setup_speculative(config, device, input->device_id, *spar);
        if (spec_ret != GENIEX_SUCCESS) {
            GENIEX_LOG_WARN("speculative decoding setup failed; falling back to plain decoding");
            teardown_speculative();
        }
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
            llama_memory_seq_rm(mem, 0, match_len, -1);
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

    const bool spec_prefill = this->spec != nullptr && (this->draft_ctx != nullptr);

    // Decode one batch (caller chunks long inputs) and advance n_past. overflow_err
    // is returned when the context is exhausted even after shifting, letting the
    // caller distinguish a too-long prompt (prefill) from a full window (decode).
    auto process = [&](const llama_token* tokens, int n_tokens, int32_t overflow_err) -> int32_t {
        int rc;
        if (spec_prefill) {
            llama_batch batch = llama_batch_init(n_tokens, /*embd=*/0, /*n_seq_max=*/1);
            for (int i = 0; i < n_tokens; ++i) {
                common_batch_add(batch, tokens[i], this->n_past + i, {0}, /*logits=*/i == n_tokens - 1);
            }
            rc = llama_decode(this->ctx, batch);
            while (rc == 1 && can_shift && slide_window(n_tokens) > 0) {
                rc = llama_decode(this->ctx, batch);
            }
            if (rc == 0 && !common_speculative_process(this->spec, batch)) {
                rc = -1;
            }
            llama_batch_free(batch);
        } else {
            llama_batch batch = llama_batch_get_one(const_cast<llama_token*>(tokens), n_tokens);
            rc                = llama_decode(this->ctx, batch);
            while (rc == 1 && can_shift && slide_window(n_tokens) > 0) {
                rc = llama_decode(this->ctx, batch);
            }
        }
        switch (rc) {
            case 0:
                break;
            case 1:
                return overflow_err;
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

    // A context overflow during prefill means the prompt itself doesn't fit,
    // even after any context shift (or the model can't shift at all); during
    // decode it means the window filled up mid-generation. Distinct causes, so
    // process() reports the one matching the phase.
    for (int i = 0; i < (int)embd_inp.size() && res == GENIEX_SUCCESS; i += n_batch) {
        int n_eval = std::min(n_batch, (int)embd_inp.size() - i);
        res        = process(embd_inp.data() + i, n_eval, GENIEX_ERROR_LLM_GENERATION_PROMPT_TOO_LONG);
    }

    profiler.prompt_end();
    profiler.update_prompt_tokens(prompt_len - this->n_past_global);
    profiler.decode_start();

    // Process output

    bool                     first_token_generated = false;
    std::vector<llama_token> generated_tokens;
    std::stringstream        full_text;

    // Emit one sampled token: records TTFT, applies EOS / stop-sequence / user
    // callback checks, and appends to the output. Returns false when generation
    // should stop (the stop reason is set on the profiler). Shared by the plain
    // and speculative decode loops.
    auto emit = [&](llama_token id) -> bool {
        if (!first_token_generated) {
            profiler.record_ttft();
            first_token_generated = true;
        }

        if (llama_vocab_is_eog(vocab, id)) {
            profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_EOS);
            return false;
        }

        char token_buf[64];
        int  n = llama_token_to_piece(vocab, id, token_buf, sizeof(token_buf) - 1, 0, this->allow_special_tokens);
        if (n < 0) {
            res = GENIEX_ERROR_LLM_GENERATION_FAILED;
            return false;
        }
        token_buf[n] = '\0';

        const bool stop_matched = std::any_of(
            cfg.stop, cfg.stop + cfg.stop_count, [&](const char* s) { return s && strcmp(token_buf, s) == 0; });
        if (stop_matched) {
            profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_STOP_SEQUENCE);
            return false;
        }

        generated_tokens.push_back(id);

        if (input->on_token && !input->on_token(token_buf, input->user_data)) {
            GENIEX_LOG_WARN("User callback requested stop during token generation");
            profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_USER);
            return false;
        }
        full_text << token_buf;
        return true;
    };

    if (this->spec) {
        auto n_generated = [&]() { return (int)generated_tokens.size(); };
        res              = decode_speculative(cfg, prompt_ids, emit, n_generated, profiler);
    } else {
        while (res == GENIEX_SUCCESS && (int)generated_tokens.size() < cfg.max_tokens) {
            llama_token id = common_sampler_sample(this->sampler, this->ctx, -1);
            common_sampler_accept(this->sampler, id, /* accept_grammar= */ true);

            if (!emit(id)) {
                break;
            }

            res = process(&id, 1, GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH);
        }
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

int32_t LlamaLlm::forward_logits(const geniex_LlmForwardLogitsInput* input, geniex_LlmForwardLogitsOutput* output) {
    if (!this->ctx || !this->model) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
    if (!input || !output) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    if (!input->input_ids || input->input_ids_count <= 0) return GENIEX_ERROR_COMMON_INVALID_INPUT;

    const llama_vocab* vocab   = llama_model_get_vocab(this->model);
    const int          n_vocab = llama_vocab_n_tokens(vocab);
    const int          n_ctx   = llama_n_ctx(this->ctx);
    const int          n_batch = llama_n_batch(this->ctx);
    const int          n_tok   = input->input_ids_count;

    for (int i = 0; i < n_tok; i++) {
        if (input->input_ids[i] < 0 || input->input_ids[i] >= n_vocab) {
            GENIEX_LOG_ERROR("forward_logits: token ID out of range: {}", input->input_ids[i]);
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
    }
    if (n_tok > n_ctx) {
        GENIEX_LOG_WARN("forward_logits: input ({}) exceeds context length ({})", n_tok, n_ctx);
        return GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
    }

    // Fresh KV in, clean KV out: result depends only on input_ids and
    // generate()'s prefix-cache/KV state is left undisturbed.
    this->reset();

    const bool         all_positions = input->all_positions;
    std::vector<float> all_logits;
    all_logits.reserve(static_cast<size_t>(all_positions ? n_tok : 1) * n_vocab);

    int32_t     rc    = GENIEX_SUCCESS;
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (int start = 0; start < n_tok; start += n_batch) {
        const int n    = std::min(n_batch, n_tok - start);
        batch.n_tokens = n;
        for (int j = 0; j < n; j++) {
            const int abs_pos  = start + j;
            batch.token[j]     = input->input_ids[abs_pos];
            batch.pos[j]       = abs_pos;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            // Always emit the last token's row; emit every position when requested.
            batch.logits[j] = (all_positions || abs_pos == n_tok - 1) ? 1 : 0;
        }

        if (llama_decode(this->ctx, batch) != 0) {
            GENIEX_LOG_ERROR("forward_logits: llama_decode failed");
            rc = GENIEX_ERROR_LLM_GENERATION_FAILED;
            break;
        }

        // Extract logits for this chunk before the next decode overwrites them.
        for (int j = 0; j < n; j++) {
            if (!batch.logits[j]) continue;
            const float* row = llama_get_logits_ith(this->ctx, j);
            if (!row) {
                rc = GENIEX_ERROR_LLM_GENERATION_FAILED;
                break;
            }
            all_logits.insert(all_logits.end(), row, row + n_vocab);
        }
        if (rc != GENIEX_SUCCESS) break;
    }
    llama_batch_free(batch);

    // Leave the KV cache clean regardless of outcome.
    this->reset();

    if (rc != GENIEX_SUCCESS) return rc;

    // malloc so the caller can release with geniex_free() (which calls free()).
    float* buf = static_cast<float*>(malloc(all_logits.size() * sizeof(float)));
    if (!buf) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
    memcpy(buf, all_logits.data(), all_logits.size() * sizeof(float));

    output->logits     = buf;
    output->vocab_size = n_vocab;
    output->n_rows     = all_positions ? n_tok : 1;
    return GENIEX_SUCCESS;
}
}  // namespace geniex

// Private
namespace geniex {

// Speculative (MTP) decode loop. Each step the MTP head drafts up to spec_n_max
// tokens, the target verifies them in one batch, and the accepted prefix is
// committed at once. Mirrors llama.cpp's speculative-simple accounting for a
// single sequence. Assumes the whole prompt was already prefilled on this->ctx
// and this->n_past is the number of prefilled tokens.
//
// id_last is the running committed token that is not yet in the KV cache. It is
// re-decoded (with the drafts) each step and emitted at the top of the next
// step, so every produced token is emitted exactly once.
//
// Partial acceptance relies on plain KV-tail removal (llama_memory_seq_rm),
// which the CPU/GPU/HTP memory backends we target support; the checkpoint dance
// the server uses for recurrent contexts is intentionally omitted.
int32_t LlamaLlm::decode_speculative(const geniex_GenerationConfig& cfg, const std::vector<llama_token>& prompt_ids,
    const std::function<bool(llama_token)>& emit, const std::function<int()>& n_generated, common::Profiler& profiler) {
    const llama_seq_id seq_id  = 0;
    auto*              mem_tgt = llama_get_memory(this->ctx);
    // ngram-* types are self-speculative and leave draft_ctx null.
    auto* mem_dft = this->draft_ctx ? llama_get_memory(this->draft_ctx) : nullptr;

    // Local view of the committed tokens the drafter reads; grows as we accept.
    std::vector<llama_token> prompt = prompt_ids;

    common_speculative_begin(this->spec, seq_id, prompt);

    // Sample the first token from the prefill; keep it separate as id_last.
    llama_token id_last = common_sampler_sample(this->sampler, this->ctx, -1);
    common_sampler_accept(this->sampler, id_last, /* accept_grammar= */ true);

    llama_batch batch = llama_batch_init(this->spec_n_max + 1, /*embd=*/0, /*n_seq_max=*/1);

    int64_t                  draft_n_total    = 0;
    int64_t                  draft_n_accepted = 0;
    int32_t                  res              = GENIEX_SUCCESS;
    bool                     stop             = false;
    std::vector<llama_token> draft;

    while (!stop && res == GENIEX_SUCCESS && n_generated() < cfg.max_tokens) {
        // Emit the running committed token (always valid: sampled by the target).
        if (!emit(id_last)) {
            break;
        }

        // Draft the tokens that (probably) follow id_last.
        draft.clear();
        common_speculative_get_draft_params(this->spec, seq_id) = {
            /* .drafting = */ true,
            /* .n_max    = */ this->spec_n_max,
            /* .n_past   = */ this->n_past,
            /* .id_last  = */ id_last,
            /* .prompt   = */ &prompt,
            /* .result   = */ &draft,
        };
        common_speculative_draft(this->spec);
        draft_n_total += (int64_t)draft.size();

        // Verification batch: [id_last, draft0, draft1, ...], all needing logits.
        common_batch_clear(batch);
        llama_pos pos = this->n_past;
        common_batch_add(batch, id_last, pos++, {seq_id}, /*logits=*/true);
        for (llama_token t : draft) {
            common_batch_add(batch, t, pos++, {seq_id}, /*logits=*/true);
        }

        if (llama_decode(this->ctx, batch) != 0 || !common_speculative_process(this->spec, batch)) {
            res = GENIEX_ERROR_LLM_GENERATION_FAILED;
            break;
        }

        // Accept the longest draft prefix the target agrees with. ids always has
        // at least one entry (the target's own next token); ids.size()-1 drafts
        // were accepted.
        std::vector<llama_token> ids      = common_sampler_sample_and_accept_n(this->sampler, this->ctx, draft);
        const size_t             n_accept = ids.size() - 1;
        draft_n_accepted += (int64_t)n_accept;
        // Only notify the speculator when it actually drafted this step.
        // ngram-* often return an empty draft (no history match yet), which
        // leaves impl_last[seq_id] unset — calling _accept then trips
        // GGML_ASSERT(impl) in common_speculative_accept.
        if (!draft.empty()) {
            common_speculative_accept(this->spec, seq_id, (uint16_t)n_accept);
        }

        // Commit id_last + accepted drafts. Emit the accepted drafts now; the
        // last id becomes the next id_last, emitted at the top of the next step.
        prompt.push_back(id_last);
        for (size_t i = 0; i < n_accept; ++i) {
            if (!emit(ids[i])) {
                stop = true;
                break;
            }
            prompt.push_back(ids[i]);
        }
        this->n_past += (int)n_accept + 1;

        // Drop any rejected draft tail from both KV caches. Nothing to drop when
        // the target agreed with the whole draft, and seq_rm is not free on the
        // HTP backend, so skip it then — same guard as the server's n_rollback.
        if (n_accept < draft.size()) {
            llama_memory_seq_rm(mem_tgt, seq_id, this->n_past, -1);
            if (mem_dft) {
                llama_memory_seq_rm(mem_dft, seq_id, this->n_past, -1);
            }
        }

        id_last = ids.back();
    }

    llama_batch_free(batch);

    profiler.set_draft_stats(draft_n_total, draft_n_accepted);
    return res;
}

void LlamaLlm::set_sampler(const geniex_SamplerConfig* cfg) {
    if (this->sampler) {
        common_sampler_free(this->sampler);
        this->sampler = nullptr;
    }
    common_params_sampling s = build_sampling_params(cfg);
    this->sampler            = common_sampler_init(this->model, s);
}

void LlamaLlm::teardown_speculative() {
    if (this->spec) {
        common_speculative_free(this->spec);
        this->spec = nullptr;
    }
    if (this->draft_ctx) {
        llama_free(this->draft_ctx);
        this->draft_ctx = nullptr;
    }
    if (this->draft_model) {
        llama_model_free(this->draft_model);
        this->draft_model = nullptr;
    }
    this->spec_n_max = 0;
}

int32_t LlamaLlm::setup_speculative(
    const geniex_ModelConfig& config, Device device, const char* device_id, common_params_speculative& spar) {
    const bool needs_draft = std::any_of(spar.types.begin(), spar.types.end(), [](common_speculative_type t) {
        return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP || t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 ||
               t == COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE;
    });
    const bool is_mtp =
        std::find(spar.types.begin(), spar.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != spar.types.end();

    spar.draft.ctx_tgt = this->ctx;
    this->spec_n_max   = spar.draft.n_max;

    if (needs_draft) {
        if (!config.spec_draft_model || config.spec_draft_model[0] == '\0') {
            GENIEX_LOG_ERROR("--spec-type '{}' requires a draft model (--draft-model)", config.spec_type);
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        llama_model_params dmpar     = build_model_params(config, device);
        auto               selection = resolve_devices(device_id);
        if (selection && !selection->empty()) {
            dmpar.devices = selection->data();
        }

        this->draft_model = llama_model_load_from_file(config.spec_draft_model, dmpar);
        if (!this->draft_model) {
            GENIEX_LOG_ERROR("failed to load draft model: {}", config.spec_draft_model);
            return GENIEX_ERROR_COMMON_MODEL_LOAD;
        }

        // Mirrors common_base_params_to_speculative: the draft context inherits
        // the target's params verbatim; only the MTP wiring, the disabled
        // rollback and the single-output limits differ.
        llama_context_params dcpar = build_context_params(config, /*n_ctx_default=*/4096, device);
        if (is_mtp) {
            dcpar.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        }
        dcpar.ctx_other             = this->ctx;
        dcpar.n_rs_seq              = 0;
        dcpar.n_outputs_max         = dcpar.n_seq_max;
        dcpar.n_outputs_max_per_seq = 1;

        this->draft_ctx = llama_init_from_model(this->draft_model, dcpar);
        if (!this->draft_ctx) {
            GENIEX_LOG_ERROR("failed to create draft context");
            return GENIEX_ERROR_COMMON_MODEL_LOAD;
        }
        spar.draft.ctx_dft = this->draft_ctx;
    }

    this->spec = common_speculative_init(spar, /*n_seq=*/1);
    if (!this->spec) {
        GENIEX_LOG_ERROR("failed to initialize speculative context");
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }

    GENIEX_LOG_INFO("speculative decoding enabled: type={}, n_max={}", config.spec_type, this->spec_n_max);
    return GENIEX_SUCCESS;
}

}  // namespace geniex
