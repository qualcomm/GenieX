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

    // If a prior llama.cpp instance tore down HTP sessions to hand off to
    // QAIRT, the ggml registry still holds cached HTP device pointers whose
    // session context was nulled out. Recreate those sessions now so the
    // upcoming device lookup / model load does not dereference a null
    // session. No-op when HTP is unused or sessions are already live.
    htp::reacquire_before_load();

    const Device             device = classify_device(input->device_id, input->config.n_gpu_layers);
    const geniex_ModelConfig config = build_model_config(input->config, /*n_ctx_default=*/4096, device);

    llama_model_params mpar = build_model_params(config);

    // Check if model path contains "gpt" and "oss" (case insensitive)
    {
        std::string model_path_lower(input->model_path);
        std::transform(model_path_lower.begin(), model_path_lower.end(), model_path_lower.begin(), ::tolower);
        bool is_gpt_oss_model =
            (model_path_lower.find("gpt") != std::string::npos) && (model_path_lower.find("oss") != std::string::npos);

        // Set special token output behavior based on model type
        this->allow_special_tokens = is_gpt_oss_model;

        // Initialize instance-level tensor buffer override array for MoE expert
        // tensors Force specific MoE expert tensors to run on CPU instead of HTP NPU
        // These patterns match: blk.{layer}.ffn_{gate|up|down}_exps.{weight|bias}
        // Only apply this override for GPT OSS models, because mxfp4 data type is not
        // supported in GGML Hexagon
        if (is_gpt_oss_model) {
            // Regex pattern to match all MoE expert tensors
            // This includes: ffn_gate_exps, ffn_up_exps, ffn_down_exps (both .weight
            // and .bias)
            this->tensor_overrides[0]  = {"\\.ffn_(up|down|gate)_exps\\.(weight|bias)", ggml_backend_cpu_buffer_type()};
            this->tensor_overrides[1]  = {nullptr, nullptr};  // Null terminator
            mpar.tensor_buft_overrides = this->tensor_overrides;
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

    // The HTP backend opens FastRPC channels at registry construction time
    // (ggml_hexagon_registry), not per-instance. Those channels live until we
    // explicitly call release_sessions, so we mark the guard whenever HTP is
    // registered — the lifetime to track is the registry, not the
    // device_id/n_gpu_layers selection of this particular load.
    if (htp::htp_backend_present()) {
        htp_guard_.mark_htp();
    }

    this->model = llama_model_load_from_file(input->model_path, mpar);
    if (!this->model) {
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }

    llama_context_params cpar = build_context_params(config, device);

    this->ctx = llama_init_from_model(this->model, cpar);
    if (!this->ctx) {
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }

    ggml_threadpool_params tpp_main  = build_threadpool_params(cpar.n_threads, device);
    ggml_threadpool_params tpp_batch = build_threadpool_params(cpar.n_threads_batch, device);
    int32_t                tp_ret    = create_and_attach_threadpools(this->pools_, this->ctx, tpp_main, tpp_batch);
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
        // Note: if file reading fails, chat_template_str remains nullopt
    }

    this->reset();
    this->reset_sampler();

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

    // Allocate output buffer and copy result
    size_t len       = result.prompt.length();
    char*  formatted = (char*)malloc(len + 1);
    if (!formatted) {
        return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;  // error: memory allocation failed
    }

    std::memcpy(formatted, result.prompt.c_str(), len);
    formatted[len] = '\0';

    output->formatted_text = formatted;
    return GENIEX_SUCCESS;
}

int32_t LlamaLlm::generate(const geniex_LlmGenerateInput* input, geniex_LlmGenerateOutput* output) {
    // Validate input
    if (!input) return GENIEX_ERROR_COMMON_INVALID_INPUT;

    bool has_input_ids   = input->input_ids != nullptr && input->input_ids_count > 0;
    bool has_prompt_utf8 = input->prompt_utf8 != nullptr;

    if (!has_input_ids && !has_prompt_utf8)
        return GENIEX_ERROR_COMMON_INVALID_INPUT;  // error: neither input_ids nor
                                                   // prompt_utf8 provided

    geniex_GenerationConfig cfg = input->config ? *input->config : geniex_GenerationConfig{};
    cfg.max_tokens              = cfg.max_tokens > 0 ? cfg.max_tokens : 128;

    // Initialzie resources
    this->set_sampler(cfg.sampler_config);
    auto*              mem     = llama_get_memory(this->ctx);
    const llama_vocab* vocab   = llama_model_get_vocab(this->model);
    const int          n_ctx   = llama_n_ctx(this->ctx);
    const int          n_batch = llama_n_batch(this->ctx);
    // Sliding the window needs seq_add (aborts on M-RoPE) and a mid-sequence
    // seq_rm (recurrent models reject it; can_shift is a false positive there).
    // When unsupported, skip the slide and let decode report truncation.
    const bool can_shift = llama_memory_can_shift(mem) && !llama_model_is_recurrent(this->model);

    // Encode the full prompt (either from input_ids or prompt_utf8)
    std::vector<llama_token> prompt_ids;
    if (has_input_ids) {
        const int32_t vocab_size = llama_vocab_n_tokens(vocab);
        // Validate token IDs are within vocabulary range
        for (int32_t i = 0; i < input->input_ids_count; i++) {
            if (input->input_ids[i] < 0 || input->input_ids[i] >= vocab_size) {
                GENIEX_LOG_ERROR("token ID out of range: {}", input->input_ids[i]);
                return GENIEX_ERROR_COMMON_INVALID_INPUT;  // error: token ID out of
                                                           // vocabulary range
            }
        }

        prompt_ids.assign(input->input_ids, input->input_ids + input->input_ids_count);
    } else {
        // Use text tokenization path
        try {
            prompt_ids = common_tokenize(vocab, std::string(input->prompt_utf8), true, true);
            {
                // Debug: log prompt tokens
                std::stringstream buffer;
                buffer << "{ ";
                for (size_t i = 0; i < prompt_ids.size(); ++i) {
                    buffer << prompt_ids[i];
                    if (i != prompt_ids.size() - 1) {
                        buffer << ",";
                    }
                }
                buffer << " }";
                GENIEX_LOG_DEBUG("Prompt token IDs:\n{}", buffer.str());
            }
        } catch (const std::exception& e) {
            return GENIEX_ERROR_LLM_TOKENIZATION_FAILED;  // error: prompt encoding failed
        }
    }
    int32_t prompt_len = static_cast<int32_t>(prompt_ids.size());

    // Prefix Match
    int match_len = 0;
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

    // Create input embedding vector from new tokens only
    std::vector<llama_token> embd_inp;
    for (int i = this->n_past_global; i < prompt_len; i++) {
        embd_inp.push_back(prompt_ids[i]);
    }

    // Main loop

    auto process = [&](std::vector<llama_token>& embd) {
        // Context shifting if needed
        if (can_shift && this->n_past + (int)embd.size() >= n_ctx) {
            const int n_past_before = this->n_past;
            const int n_keep        = 4;
            const int needed        = this->n_past + (int)embd.size() - n_ctx + 1;
            int       n_discard     = std::max(this->n_past / 2 - n_keep, needed);
            n_discard               = std::min(n_discard, this->n_past - n_keep);

            if (n_discard > 0) {
                // Context shifting using llama.cpp memory management
                llama_memory_seq_rm(mem, 0, n_keep, n_keep + n_discard);
                llama_memory_seq_add(mem, 0, n_keep + n_discard, this->n_past, -n_discard);
                this->n_past -= n_discard;
            }

            GENIEX_LOG_INFO(
                "Context shifting - discarding {} tokens, n_keep: {}, "
                "this->n_past before: {}, this->n_past after: "
                "{}",
                n_discard,
                n_keep,
                n_past_before,
                this->n_past);
        }

        for (int i = 0; i < (int)embd.size(); i += n_batch) {
            int n_eval = (int)embd.size() - i;
            if (n_eval > n_batch) {
                n_eval = n_batch;
            }

            llama_batch batch = llama_batch_get_one(&embd[i], n_eval);
            // 1 means the batch does not fit the KV cache: report truncation, not a generic failure.
            switch (llama_decode(this->ctx, batch)) {
                case 0:
                    break;
                case 1:
                    return GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
                default:
                    return GENIEX_ERROR_LLM_GENERATION_FAILED;
            }

            n_past += n_eval;
        }

        return GENIEX_SUCCESS;
    };

    // Process input (prefilling)

    common::Profiler profiler;
    profiler.prompt_start();

    bool                     first_token_generated = false;
    int32_t                  res                   = GENIEX_SUCCESS;
    std::vector<llama_token> generated_tokens;
    std::stringstream        full_text;

    std::vector<llama_token> embd;

    for (size_t i = 0; i < embd_inp.size(); ++i) {
        common_sampler_accept(this->sampler,
            embd_inp[i],
            /* accept_grammar= */ false);

        embd.push_back(embd_inp[i]);
        if ((int)embd.size() >= n_batch || i == embd_inp.size() - 1) {
            // A prompt block that does not fit reports truncation; the unified
            // finalize below still emits valid profiler data.
            res = process(embd);
            embd.clear();
            if (res != GENIEX_SUCCESS) {
                break;
            }
        }
    }

    profiler.prompt_end();
    profiler.update_prompt_tokens(prompt_len - this->n_past_global);
    profiler.decode_start();

    // Process output

    while (res == GENIEX_SUCCESS && (int)generated_tokens.size() < cfg.max_tokens) {
        llama_token id = common_sampler_sample(this->sampler, this->ctx, -1);
        common_sampler_accept(this->sampler, id, /* accept_grammar= */ true);

        // Record TTFT on first token generation
        if (!first_token_generated) {
            profiler.record_ttft();
            first_token_generated = true;
            GENIEX_LOG_DEBUG("First token generated, TTFT recorded");
        }

        if (llama_vocab_is_eog(vocab, id)) {
            GENIEX_LOG_DEBUG("EOS token generated, stopping generation");
            profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_EOS);
            break;
        }

        // Convert token to string
        char token_buf[64];
        // Use instance-level setting for special token output
        int n = llama_token_to_piece(vocab, id, token_buf, sizeof(token_buf) - 1, 0, this->allow_special_tokens);
        if (n < 0) {
            res = GENIEX_ERROR_LLM_GENERATION_FAILED;
            break;
        }
        token_buf[n] = '\0';

        // Check stop sequences
        bool stop_matched = false;
        for (int i = 0; i < cfg.stop_count; ++i) {
            if (cfg.stop[i] && strcmp(token_buf, cfg.stop[i]) == 0) {
                GENIEX_LOG_DEBUG("Stop sequence matched: '{}'", cfg.stop[i]);
                profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_STOP_SEQUENCE);
                stop_matched = true;
                break;
            }
        }
        if (stop_matched) {
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

        std::vector<llama_token> embd(1, id);
        // Decode slides the window first when supported, so a single token fits.
        // It only fails to fit when sliding is disabled (M-RoPE / recurrent):
        // that surfaces as a context-length error and keeps the text so far.
        // No break needed — it is the last statement, the while check exits.
        res = process(embd);
    }

    // Set stop reason if not already set
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

    this->n_past_global = prompt_len + generated_tokens.size();
    this->past_prompt_tokens.insert(this->past_prompt_tokens.end(), embd_inp.begin(), embd_inp.end());
    this->past_prompt_tokens.insert(this->past_prompt_tokens.end(), generated_tokens.begin(), generated_tokens.end());

    return res;
}
}  // namespace geniex

// Private
namespace geniex {

void LlamaLlm::reset_sampler() { this->set_sampler(nullptr); }

void LlamaLlm::set_sampler(const geniex_SamplerConfig* cfg) {
    if (this->sampler) {
        common_sampler_free(this->sampler);
        this->sampler = nullptr;
    }
    common_params_sampling s = build_sampling_params(cfg);
    this->sampler            = common_sampler_init(this->model, s);
}

}  // namespace geniex
