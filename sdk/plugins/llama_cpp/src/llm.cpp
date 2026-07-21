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

    // FIX: HTP backend patch — only reacquire HTP sessions when we're actually
    // going to use them (npu / hybrid). CPU and GPU targets shouldn't be gated
    // on the ADSP domain's health.
    if (device == Device::NPU) {
        htp::reacquire_before_load();
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

    // Speculative decoding: optional, keyed on a non-empty spec_type ("none" also
    // disables). Failure is non-fatal — we log and fall back to plain decoding.
    if (config.spec_type && config.spec_type[0] != '\0' && strcmp(config.spec_type, "none") != 0) {
        int32_t spec_ret = setup_speculative(config, device, input->device_id);
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

            res = process(&id, 1);
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
    auto*              mem_dft = llama_get_memory(this->draft_ctx);

    // The MTP head reads the target's nextn embeddings; enable them for every
    // decode. This is a static property of the speculator, so set it once.
    llama_set_embeddings(this->ctx, common_speculative_need_embd_nextn(this->spec));

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

        // Drop any rejected draft tail from both KV caches.
        llama_memory_seq_rm(mem_tgt, seq_id, this->n_past, -1);
        llama_memory_seq_rm(mem_dft, seq_id, this->n_past, -1);

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

// Set up speculative decoding from config.spec_type (one or more comma-separated
// llama.cpp type names). draft-model types (draft-mtp / draft-eagle3 / draft-simple) load a
// separate draft GGUF into a context sharing the target KV cache; ngram-* types
// are self-speculative and need no draft model. We keep our own model/context
// builders (not llama.cpp's factory) so the drafter inherits the target's device
// placement — without it the drafter grabs HTP0 and breaks the MTP graph.
int32_t LlamaLlm::setup_speculative(const geniex_ModelConfig& config, Device device, const char* device_id) {
    std::vector<common_speculative_type> types =
        common_speculative_types_from_names(string_split<std::string>(config.spec_type, ','));
    types.erase(std::remove(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_NONE), types.end());
    if (types.empty()) {
        GENIEX_LOG_ERROR("unrecognized --spec-type: '{}'", config.spec_type);
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    const bool needs_draft = std::any_of(types.begin(), types.end(), [](common_speculative_type t) {
        return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP || t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 ||
               t == COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE;
    });

    common_params_speculative spar;
    spar.types       = types;
    spar.draft.n_max = config.spec_n_max > 0 ? config.spec_n_max : 3;
    if (config.spec_n_min > 0) spar.draft.n_min = config.spec_n_min;
    if (config.spec_p_min > 0) spar.draft.p_min = config.spec_p_min;
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

        // Draft context: same tuning as the target, plus the MTP wiring (shared
        // KV via ctx_other, no rollback snapshots).
        llama_context_params dcpar = build_context_params(config, /*n_ctx_default=*/4096, device);
        dcpar.ctx_type             = LLAMA_CONTEXT_TYPE_MTP;
        dcpar.ctx_other            = this->ctx;
        dcpar.n_rs_seq             = 0;

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
