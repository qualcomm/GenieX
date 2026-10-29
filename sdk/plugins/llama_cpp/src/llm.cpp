// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "llm.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#include "chat.h"
#include "common.h"
#include "geniex_model.h"
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
            if (device == Device::NPU) {
                htp::set_power_mode(config.power_mode);
            } else if (config.power_mode != GENIEX_POWER_MODE_BURST) {
                GENIEX_LOG_WARN("power_mode is only meaningful on the NPU device; ignoring on this device");
            }
            htp::reacquire_before_load();
        }
    }

    // Resolve the compute-unit alias to a devices[] list for llama.cpp; must
    // outlive mpar (mpar.devices points into selection's buffer).
    auto selection = resolve_devices(input->device_id);
    if (!selection) {
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }
    apply_device_selection(mpar, *selection, device);

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
    if (device == Device::GPU) {
        char arch[32] = {};
        llama_model_meta_val_str(this->model, "general.architecture", arch, sizeof(arch));
        if (strcmp(arch, "gpt-oss") == 0) {
            cpar.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            GENIEX_LOG_INFO("gpt-oss on GPU: flash attention enabled");
        }
    }
#if defined(_WIN32)
    // Gemma 4 26B-A4B (the only MoE gemma4) on X2 Elite NPU: weights already take ~13.8 GiB of the ~15.1 GiB HTP
    // mapping cap, so the default 1024 ubatch compute buffer pushes ctx >= 3584 over it and fastrpc_mmap fails.
    // X Elite runs it fine at 1024 up to 32k ctx. See qcom-ai-hub/geniex#1736.
    if (device == Device::NPU && config.n_ubatch <= 0) {
        char arch[32]    = {};
        char experts[16] = {};
        llama_model_meta_val_str(this->model, "general.architecture", arch, sizeof(arch));
        llama_model_meta_val_str(this->model, "gemma4.expert_count", experts, sizeof(experts));
        if (strcmp(arch, "gemma4") == 0 && atoi(experts) > 1) {
            char* chipset = nullptr;
            geniex_model_detect_chipset(/*offline=*/1, &chipset);
            if (chipset && strcmp(chipset, "qualcomm-snapdragon-x2-elite") == 0) {
                cpar.n_ubatch = 512;
                GENIEX_LOG_INFO("gemma4 MoE on {} NPU: n_ubatch capped to 512", chipset);
            }
            if (chipset) geniex_free(chipset);
        }
    }
#endif
    this->ctx = llama_init_from_model(this->model, cpar);
    if (!this->ctx) {
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }

    ggml_threadpool_params tpp_main  = build_threadpool_params(cpar.n_threads, device);
    ggml_threadpool_params tpp_batch = build_threadpool_params(cpar.n_threads_batch, device);
    int32_t                tp_ret    = this->pools_.attach(this->ctx, tpp_main, tpp_batch);
    if (tp_ret != GENIEX_SUCCESS) {
        return tp_ret;
    }

    warmup_context(this->model, this->ctx);

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
    if (output) {
        output->formatted_text        = nullptr;
        output->chat_template_grammar = {};
    }
    if (!input || !input->messages || !output || input->message_count <= 0) {
        return GENIEX_ERROR_COMMON_INVALID_INPUT;  // error: invalid input
    }

    // Convert geniex_ChatMessage to common_chat_msg
    std::vector<common_chat_msg> common_messages;
    common_messages.reserve(input->message_count);

    for (int32_t i = 0; i < input->message_count; ++i) {
        const geniex_LlmChatMessage& src = input->messages[i];
        if (!src.role) {
            GENIEX_LOG_ERROR("messages[{}] has null role", i);
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
        common_chat_msg msg;
        msg.role = src.role;
        // An assistant turn that only issues tool calls carries no content.
        if (src.content) msg.content = src.content;
        apply_tool_fields(msg, src.tool_calls, src.tool_call_count, src.tool_call_id, src.tool_name);
        common_messages.push_back(std::move(msg));
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
    inputs.parallel_tool_calls   = input->parallel_tool_calls;

    if (input->tools && strlen(input->tools) > 0) {
        inputs.tools = common_chat_tools_parse_oaicompat(common_json::parse(std::string(input->tools)));
    }

    inputs.enable_thinking = input->enable_thinking;

    // Apply chat template and retain its sampler grammar state.
    common_chat_session    session(tmpls.get(), llama_model_get_vocab(this->model), inputs);
    common_params_sampling sampling;
    session.apply_sampling(sampling);

    const std::string prompt      = session.prompt().str();
    char*             prompt_copy = strdup(prompt.c_str());
    if (!prompt_copy) {
        return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
    }
    if (!copy_chat_template_grammar(sampling, &output->chat_template_grammar)) {
        free(prompt_copy);
        return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
    }

    output->formatted_text = prompt_copy;
    return GENIEX_SUCCESS;
}

int32_t LlamaLlm::generate(const geniex_LlmGenerateInput* input, geniex_LlmGenerateOutput* output) {
    // Validate input
    if (!input) return GENIEX_ERROR_COMMON_INVALID_INPUT;

    bool has_input_embd  = input->input_embd != nullptr && input->input_embd_count > 0;
    bool has_input_ids   = input->input_ids != nullptr && input->input_ids_count > 0;
    bool has_prompt_utf8 = input->prompt_utf8 != nullptr;

    if (!has_input_embd && !has_input_ids && !has_prompt_utf8)
        return GENIEX_ERROR_COMMON_INVALID_INPUT;  // error: none of input_embd, input_ids, prompt_utf8 provided

    geniex_GenerationConfig cfg = input->config ? *input->config : geniex_GenerationConfig{};
    cfg.max_tokens              = cfg.max_tokens > 0 ? cfg.max_tokens : 128;

    // Initialzie resources
    this->set_sampler(cfg.sampler_config);
    auto*              mem       = llama_get_memory(this->ctx);
    const llama_vocab* vocab     = llama_model_get_vocab(this->model);
    const int          n_ctx     = llama_n_ctx(this->ctx);
    const int          n_batch   = llama_n_batch(this->ctx);
    const bool         can_shift = llama_memory_can_shift(mem) && !llama_model_is_recurrent(this->model);

    if (has_input_embd) {
        const int32_t expected_dim = llama_model_n_embd_inp(this->model);
        if (input->input_embd_dim != expected_dim) {
            GENIEX_LOG_ERROR("input_embd_dim ({}) does not match model's expected embedding width ({})",
                input->input_embd_dim,
                expected_dim);
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
        if (this->spec) {
            GENIEX_LOG_ERROR("input_embd is not supported together with speculative decoding");
            return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;
        }
        // Embedding turns always fully redecode (see the "update past record"
        // comment near the end of this function), so start from a clean KV
        // cache rather than continuing from whatever a prior turn left behind.
        this->reset();
    }

    // Encode the full prompt (from input_embd, input_ids, or prompt_utf8)
    std::vector<llama_token> prompt_ids;
    if (!has_input_embd) {
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
    }

    // Prefix Match — not applicable to embedding input (nothing to diff against a
    // token history), so an input_embd call always fully redecodes its rows.

    int32_t prompt_len = has_input_embd ? input->input_embd_count : static_cast<int32_t>(prompt_ids.size());
    if (!has_input_embd) {
        int match_len = 0;
        while (match_len < std::min((int)past_prompt_tokens.size(), prompt_len) &&
               past_prompt_tokens[match_len] == prompt_ids[match_len]) {
            match_len++;
        }
        // Re-decode the last prompt token so sampling uses logits for this prompt.
        if (match_len == prompt_len && match_len > 0) {
            --match_len;
        }
        GENIEX_LOG_DEBUG(
            "prefix match: past_prompt_tokens size: {}, prompt_len: {}, "
            "match_len: {}",
            past_prompt_tokens.size(),
            prompt_len,
            match_len);

        if (match_len < (int)this->past_prompt_tokens.size()) {
            if (match_len < this->n_past_global - this->n_past) {
                GENIEX_LOG_WARN("prefix match: matched prefix is outside the cache; resetting state");
                this->reset();
            } else if (!llama_memory_seq_rm(mem, 0, match_len, -1)) {
                GENIEX_LOG_WARN("prefix match: rollback to token {} failed; resetting state", match_len);
                this->reset();
            } else {
                this->n_past        = match_len;
                this->n_past_global = match_len;
            }
            GENIEX_LOG_INFO("prefix match: n_past_global rollback to: {}", this->n_past_global);
            this->past_prompt_tokens.assign(prompt_ids.begin(), prompt_ids.begin() + this->n_past_global);
        }
    }

    std::vector<llama_token> embd_inp;
    if (!has_input_embd) {
        embd_inp.assign(prompt_ids.begin() + this->n_past_global, prompt_ids.end());
    }

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

        if (!llama_memory_seq_rm(mem, 0, n_keep, n_keep + n_discard)) {
            GENIEX_LOG_WARN("Context shifting: failed to discard {} tokens; keeping state", n_discard);
            return 0;
        }
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
            common_batch batch(this->ctx);
            for (int i = 0; i < n_tokens; ++i) {
                batch.add(tokens[i], this->n_past + i, /*seq_id=*/0, /*output=*/i == n_tokens - 1);
            }
            rc = llama_process(this->ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            while (rc == 1 && can_shift && slide_window(n_tokens) > 0) {
                rc = llama_process(this->ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            }
            if (rc == 0 && !common_speculative_process(this->spec, batch)) {
                rc = -1;
            }
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

    // Decode one batch of pre-computed embedding rows and advance n_past.
    // Mirrors process() above but populates llama_batch.embd instead of .token,
    // the same mechanism the VLM/mtmd path uses for vision-encoder output.
    auto process_embd = [&](const float* rows, int n_rows, int32_t overflow_err) -> int32_t {
        llama_batch batch = llama_batch_init(n_rows, input->input_embd_dim, /*n_seq_max=*/1);
        batch.n_tokens    = n_rows;
        std::memcpy(batch.embd, rows, (size_t)n_rows * input->input_embd_dim * sizeof(float));
        for (int i = 0; i < n_rows; ++i) {
            batch.pos[i]       = this->n_past + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i == n_rows - 1);
        }
        int rc = llama_decode(this->ctx, batch);
        while (rc == 1 && can_shift && slide_window(n_rows) > 0) {
            rc = llama_decode(this->ctx, batch);
        }
        llama_batch_free(batch);
        switch (rc) {
            case 0:
                break;
            case 1:
                return overflow_err;
            default:
                return GENIEX_ERROR_LLM_GENERATION_FAILED;
        }
        n_past += n_rows;
        return GENIEX_SUCCESS;
    };

    // Process input (prefilling)

    if (has_input_embd) {
        for (int i = 0; i < input->input_embd_count && res == GENIEX_SUCCESS; i += n_batch) {
            int n_eval = std::min(n_batch, input->input_embd_count - i);
            res        = process_embd(input->input_embd + (size_t)i * input->input_embd_dim,
                n_eval,
                GENIEX_ERROR_LLM_GENERATION_PROMPT_TOO_LONG);
        }
    } else {
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
        int  n = llama_token_to_piece(vocab, id, token_buf, sizeof(token_buf) - 1, 0, /*special=*/true);
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
    if (has_input_embd) {
        // Embedding rows aren't diffable against a token history and their KV
        // isn't representable as past_prompt_tokens, so the KV cache is dropped
        // entirely after an embedding turn: the next call (text, input_ids, or
        // another input_embd) always starts a full redecode from position 0.
        this->reset();
    } else {
        this->n_past_global = prompt_len + generated_tokens.size();
        this->past_prompt_tokens.insert(this->past_prompt_tokens.end(), embd_inp.begin(), embd_inp.end());
        this->past_prompt_tokens.insert(
            this->past_prompt_tokens.end(), generated_tokens.begin(), generated_tokens.end());
    }

    return res;
}

int32_t LlamaLlm::get_model_info(geniex_LlmModelInfo* output) {
    if (!this->model) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
    const llama_vocab* vocab = llama_model_get_vocab(this->model);
    output->vocab_size       = llama_vocab_n_tokens(vocab);
    const llama_token bos    = llama_vocab_bos(vocab);
    output->bos_token        = (bos == LLAMA_TOKEN_NULL) ? -1 : static_cast<int32_t>(bos);
    output->add_bos          = llama_vocab_get_add_bos(vocab) ? 1 : 0;
    output->embd_dim         = llama_model_n_embd_inp(this->model);
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

int32_t LlamaLlm::decide(const geniex_DecideInput* input, geniex_DecideOutput* output) {
    if (!this->model || !this->ctx || !input || !output) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    if (input->image_count != 0 || input->audio_count != 0) return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;

    DecisionPrepared prepared;
    int32_t          rc = prepare_decision(this->model, input, &prepared);
    if (rc != GENIEX_SUCCESS) return rc;
    const llama_vocab* vocab = llama_model_get_vocab(this->model);

    // Decision codes must map to distinct single tokens in the model vocabulary.
    std::vector<llama_token> candidate_tokens;
    for (const auto& code : prepared.candidates) {
        const auto literal = common_tokenize(vocab, code, false, false);
        if (literal.size() != 1 ||
            std::find(candidate_tokens.begin(), candidate_tokens.end(), literal[0]) != candidate_tokens.end()) {
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
        candidate_tokens.push_back(literal[0]);
    }

    // Verify tokenization at this exact prompt boundary, then read one fresh logits row.
    auto score_prompt = [&](const std::string& prompt, std::vector<float>& scores, int32_t& input_tokens) -> int32_t {
        const auto tokens = common_tokenize(vocab, prompt, true, true);
        if (tokens.empty() || tokens.size() > llama_n_ctx(this->ctx)) {
            return GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
        }
        for (size_t i = 0; i < prepared.candidates.size(); ++i) {
            const auto joined = common_tokenize(vocab, prompt + prepared.candidates[i], true, true);
            if (joined.size() != tokens.size() + 1 || !std::equal(tokens.begin(), tokens.end(), joined.begin()) ||
                joined.back() != candidate_tokens[i]) {
                GENIEX_LOG_WARN("decision candidate '{}' is not one token after the prompt", prepared.candidates[i]);
                return GENIEX_ERROR_COMMON_INVALID_INPUT;
            }
        }
        geniex_LlmForwardLogitsInput request{};
        request.input_ids       = tokens.data();
        request.input_ids_count = static_cast<int32_t>(tokens.size());
        geniex_LlmForwardLogitsOutput result{};
        const int32_t                 result_code = forward_logits(&request, &result);
        if (result_code != GENIEX_SUCCESS) return result_code;
        scores.resize(candidate_tokens.size());
        for (size_t i = 0; i < candidate_tokens.size(); ++i) scores[i] = result.logits[candidate_tokens[i]];
        free(result.logits);
        input_tokens = static_cast<int32_t>(tokens.size());
        return GENIEX_SUCCESS;
    };

    std::vector<float> scores;
    int32_t            input_tokens = 0;
    rc                              = score_prompt(prepared.prompt, scores, input_tokens);
    if (rc != GENIEX_SUCCESS) return rc;

    // LEV calibrates multi-pass decisions here; other models return raw option logits.
    std::vector<double> probabilities;
    if (prepared.type == "lev" && (prepared.question_type == "noul" || !prepared.reversed_prompt.empty())) {
        auto softmax = [&](const std::vector<float>& logits) {
            std::vector<double> result(logits.size());
            if (result.empty()) return result;
            const float peak = *std::max_element(logits.begin(), logits.end());
            for (size_t i = 0; i < logits.size(); ++i) {
                result[i] = std::exp((double)(logits[i] - peak) / prepared.temperature);
            }
            double sum = result.front();
            for (size_t i = 1; i < result.size(); ++i) sum += result[i];
            if (!(sum > 0.0) || !std::isfinite(sum)) return std::vector<double>{};
            for (double& value : result) value /= sum;
            return result;
        };
        const auto first = softmax(scores);
        if (first.empty()) return GENIEX_ERROR_COMMON_INVALID_INPUT;
        probabilities.resize(prepared.original_index.size());
        if (prepared.question_type == "noul") {
            // The nine codes represent ratings 0–8; their expectation is P(true).
            double yes = 0.0;
            for (size_t i = 0; i < first.size(); ++i) yes += first[i] * i / (first.size() - 1);
            probabilities[prepared.true_index]     = yes;
            probabilities[1 - prepared.true_index] = 1.0 - yes;
        } else {
            // Average forward and reversed choice probabilities in caller order.
            std::vector<float> reversed_scores;
            int32_t            reversed_tokens = 0;
            rc = score_prompt(prepared.reversed_prompt, reversed_scores, reversed_tokens);
            if (rc != GENIEX_SUCCESS) return rc;
            input_tokens += reversed_tokens;
            const auto second = softmax(reversed_scores);
            if (second.empty()) return GENIEX_ERROR_COMMON_INVALID_INPUT;
            for (size_t i = 0; i < first.size(); ++i) {
                probabilities[prepared.original_index[i]] += first[i] / 2;
                probabilities[prepared.original_index[first.size() - 1 - i]] += second[i] / 2;
            }
        }
    }

    // Return either raw logits or calibrated probabilities, indexed by the caller's options.
    const size_t option_count = prepared.original_index.size();
    if (probabilities.empty()) {
        output->logits = static_cast<float*>(calloc(option_count, sizeof(float)));
        if (!output->logits) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
        for (size_t i = 0; i < option_count; ++i) output->logits[prepared.original_index[i]] = scores[i];
    } else {
        output->probabilities = static_cast<float*>(calloc(option_count, sizeof(float)));
        if (!output->probabilities) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
        for (size_t i = 0; i < option_count; ++i) output->probabilities[i] = static_cast<float>(probabilities[i]);
    }
    output->option_count  = static_cast<int32_t>(option_count);
    output->input_tokens  = input_tokens;
    output->temperature   = prepared.temperature;
    output->decision_type = strdup(prepared.type.c_str());
    if (!output->decision_type) {
        free(output->logits);
        free(output->probabilities);
        std::memset(output, 0, sizeof(*output));
        return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
    }
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

    common_batch batch(this->ctx);

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
        batch.clear();
        llama_pos pos = this->n_past;
        batch.add(id_last, pos++, seq_id, /*output=*/true);
        for (llama_token t : draft) {
            batch.add(t, pos++, seq_id, /*output=*/true);
        }

        if (llama_process(this->ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0 ||
            !common_speculative_process(this->spec, batch)) {
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

    profiler.set_draft_stats(draft_n_total, draft_n_accepted);
    return res;
}

void LlamaLlm::set_sampler(const geniex_SamplerConfig* cfg) {
    if (this->sampler) {
        common_sampler_free(this->sampler);
        this->sampler = nullptr;
    }
    common_params_sampling s = build_sampling_params(cfg);
    if (cfg && cfg->ignore_eos) {
        const llama_vocab* vocab = llama_model_get_vocab(this->model);
        for (llama_token id = 0; id < llama_vocab_n_tokens(vocab); ++id) {
            if (llama_vocab_is_eog(vocab, id)) {
                s.logit_bias.push_back({id, -INFINITY});
            }
        }
    }
    this->sampler = common_sampler_init(this->model, s);
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
        if (selection) {
            apply_device_selection(dmpar, *selection, device);
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
