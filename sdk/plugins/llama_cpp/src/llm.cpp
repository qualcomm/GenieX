#include "llm.h"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <thread>
#include <vector>

#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "logging.h"
#include "profiler.h"

namespace geniex {

LlamaLlm::~LlamaLlm() {
    if (sampler) common_sampler_free(sampler);
    if (ctx) llama_free(ctx);
    if (model) llama_model_free(model);

    // Free threadpools
    if (threadpool) this->threadpool_free_fn(threadpool);
    if (threadpool_batch) this->threadpool_free_fn(threadpool_batch);
}

int32_t LlamaLlm::create_impl(const ml_LlmCreateInput* input) {
    if (!input) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    if (!input->model_path) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    auto mpar      = llama_model_default_params();
    mpar.use_mmap  = true;
    mpar.use_mlock = false;

    mpar.n_gpu_layers = input->config.n_gpu_layers;

    // Check if model path contains "gpt" and "oss" (case insensitive)
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

    if (input->device_id) {
        // Parse comma-separated device list (e.g., "HTP0,HTP1,HTP2,HTP3")
        std::string                     device_str(input->device_id);
        std::vector<ggml_backend_dev_t> devices;

        size_t start = 0;
        size_t end   = 0;
        while ((end = device_str.find(',', start)) != std::string::npos) {
            std::string dev_name = device_str.substr(start, end - start);
            auto*       dev      = ggml_backend_dev_by_name(dev_name.c_str());
            if (dev) {
                devices.push_back(dev);
                GENIEX_LOG_INFO("Found device: {}", dev_name);
            } else {
                GENIEX_LOG_WARN("Device '{}' not found, skipping", dev_name);
            }
            start = end + 1;
        }
        // Handle last (or only) device name
        std::string last_dev = device_str.substr(start);
        if (!last_dev.empty()) {
            auto* dev = ggml_backend_dev_by_name(last_dev.c_str());
            if (dev) {
                devices.push_back(dev);
                GENIEX_LOG_INFO("Found device: {}", last_dev);
            } else {
                GENIEX_LOG_WARN("Device '{}' not found, skipping", last_dev);
            }
        }

        if (!devices.empty()) {
            ggml_backend_dev_t device_array[9] = {nullptr};
            for (size_t i = 0; i < devices.size() && i < 8; ++i) {
                device_array[i] = devices[i];
            }
            mpar.devices = device_array;
            GENIEX_LOG_INFO("Using {} device(s): {}", devices.size(), input->device_id);
        } else {
            GENIEX_LOG_ERROR("No valid devices found in '{}'", input->device_id);
            return ML_ERROR_COMMON_INVALID_INPUT;
        }
    }

    this->model = llama_model_load_from_file(input->model_path, mpar);
    if (!this->model) {
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

    auto config = input->config;

    // TODO: move this to default llama_paramas func
    auto default_config  = model_config_default();
    auto cpar            = llama_context_default_params();
    cpar.n_ctx           = config.n_ctx > 0 ? config.n_ctx : default_config.n_ctx;
    cpar.n_batch         = config.n_batch > 0 ? config.n_batch : default_config.n_batch;
    cpar.n_ubatch        = config.n_ubatch > 0 ? config.n_ubatch : default_config.n_ubatch;
    cpar.n_seq_max       = config.n_seq_max > 0 ? config.n_seq_max : default_config.n_seq_max;
    cpar.n_threads       = config.n_threads > 0 ? config.n_threads : default_config.n_threads;
    cpar.n_threads_batch = config.n_threads_batch > 0 ? config.n_threads_batch : default_config.n_threads_batch;
    cpar.kv_unified      = true;   // use unified KV cache
    cpar.no_perf         = false;  // enable performance counters

    std::string device_id_str(input->device_id ? input->device_id : "");
    if (device_id_str.find("HTP0") != std::string::npos) {
        cpar.type_k          = GGML_TYPE_Q8_0;
        cpar.type_v          = GGML_TYPE_Q8_0;
        cpar.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    };

    this->ctx = llama_init_from_model(this->model, cpar);
    if (!this->ctx) {
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

    // Create and attach threadpools for better performance
    auto* cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        GENIEX_LOG_ERROR("No CPU backend found");
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

    auto* reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto* ggml_threadpool_new_fn =
        (decltype(ggml_threadpool_new)*)ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto* ggml_threadpool_free_fn =
        (decltype(ggml_threadpool_free)*)ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

    if (!ggml_threadpool_new_fn || !ggml_threadpool_free_fn) {
        GENIEX_LOG_ERROR("Failed to get threadpool functions");
        return ML_ERROR_COMMON_MODEL_LOAD;
    }
    this->threadpool_free_fn = ggml_threadpool_free_fn;

    // Create threadpool parameters matching main.cpp
    struct ggml_threadpool_params tpp_batch = ggml_threadpool_params_default(cpar.n_threads_batch);
    struct ggml_threadpool_params tpp       = ggml_threadpool_params_default(cpar.n_threads);

    // Create batch threadpool if different from main threadpool
    if (cpar.n_threads_batch != cpar.n_threads) {
        this->threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!this->threadpool_batch) {
            GENIEX_LOG_ERROR("Batch threadpool create failed: n_threads {}", tpp_batch.n_threads);
            return ML_ERROR_COMMON_MODEL_LOAD;
        }
        // Start the non-batch threadpool in the paused state
        tpp.paused = true;
    }

    // Create main threadpool
    this->threadpool = ggml_threadpool_new_fn(&tpp);
    if (!this->threadpool) {
        GENIEX_LOG_ERROR("Threadpool create failed: n_threads {}", tpp.n_threads);
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

    // Attach threadpools to context
    llama_attach_threadpool(this->ctx, this->threadpool, this->threadpool_batch);

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

    return ML_SUCCESS;
}

int32_t LlamaLlm::reset() {
    this->n_past_global = 0;
    this->n_past        = 0;
    this->past_prompt_tokens.clear();

    llama_memory_clear(llama_get_memory(this->ctx), /*clear data=*/true);

    return ML_SUCCESS;
}

int32_t LlamaLlm::save_kv_cache(const ml_KvCacheSaveInput* input, ml_KvCacheSaveOutput* _) {
    return llama_state_save_file(this->ctx, input->path, nullptr, 0) ? ML_SUCCESS : ML_ERROR_COMMON_UNKNOWN;
}

int32_t LlamaLlm::load_kv_cache(const ml_KvCacheLoadInput* input, ml_KvCacheLoadOutput* _) {
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

    return ret ? ML_SUCCESS : ML_ERROR_COMMON_UNKNOWN;
}

int32_t LlamaLlm::apply_chat_template(
    const ml_LlmApplyChatTemplateInput* input, ml_LlmApplyChatTemplateOutput* output) {
    if (!input || !input->messages || !output || input->message_count <= 0) {
        return ML_ERROR_COMMON_INVALID_INPUT;  // error: invalid input
    }

    // Convert ml_ChatMessage to common_chat_msg
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
        return ML_ERROR_COMMON_MEMORY_ALLOCATION;  // error: memory allocation failed
    }

    std::memcpy(formatted, result.prompt.c_str(), len);
    formatted[len] = '\0';

    output->formatted_text = formatted;
    return ML_SUCCESS;
}

int32_t LlamaLlm::generate(const ml_LlmGenerateInput* input, ml_LlmGenerateOutput* output) {
    // Validate input
    if (!input) return ML_ERROR_COMMON_INVALID_INPUT;

    bool has_input_ids   = input->input_ids != nullptr && input->input_ids_count > 0;
    bool has_prompt_utf8 = input->prompt_utf8 != nullptr;

    if (!has_input_ids && !has_prompt_utf8)
        return ML_ERROR_COMMON_INVALID_INPUT;  // error: neither input_ids nor
                                               // prompt_utf8 provided

    ml_GenerationConfig cfg = input->config ? *input->config : ml_GenerationConfig{};
    cfg.max_tokens          = cfg.max_tokens > 0 ? cfg.max_tokens : 128;

    // Initialzie resources
    this->set_sampler(cfg.sampler_config);
    auto*              mem     = llama_get_memory(this->ctx);
    const llama_vocab* vocab   = llama_model_get_vocab(this->model);
    const int          n_ctx   = llama_n_ctx(this->ctx);
    const int          n_batch = llama_n_batch(this->ctx);

    // Encode the full prompt (either from input_ids or prompt_utf8)
    std::vector<llama_token> prompt_ids;
    if (has_input_ids) {
        const int32_t vocab_size = llama_vocab_n_tokens(vocab);
        // Validate token IDs are within vocabulary range
        for (int32_t i = 0; i < input->input_ids_count; i++) {
            if (input->input_ids[i] < 0 || input->input_ids[i] >= vocab_size) {
                GENIEX_LOG_ERROR("token ID out of range: {}", input->input_ids[i]);
                return ML_ERROR_COMMON_INVALID_INPUT;  // error: token ID out of
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
            return ML_ERROR_LLM_TOKENIZATION_FAILED;  // error: prompt encoding failed
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
        if (this->n_past + (int)embd.size() >= n_ctx) {
            int       n_past_before = this->n_past;
            int       n_keep        = 4;
            const int n_discard     = this->n_past / 2 - n_keep;

            // Context shifting using llama.cpp memory management
            llama_memory_seq_rm(mem, 0, n_keep, n_keep + n_discard);
            llama_memory_seq_add(mem, 0, n_keep + n_discard, this->n_past, -n_discard);
            this->n_past -= n_discard;

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
            if (llama_decode(this->ctx, batch) != 0) {
                return ML_ERROR_LLM_GENERATION_FAILED;  // error: llama_decode failed
                                                        // during token processing
            }

            n_past += n_eval;
        }

        return ML_SUCCESS;
    };

    // Process input (prefilling)

    common::Profiler profiler;
    profiler.prompt_start();

    std::vector<llama_token> embd;

    for (size_t i = 0; i < embd_inp.size(); ++i) {
        common_sampler_accept(this->sampler,
            embd_inp[i],
            /* accept_grammar= */ false);

        embd.push_back(embd_inp[i]);
        if ((int)embd.size() >= n_batch || i == embd_inp.size() - 1) {
            int32_t res = process(embd);  // Enable detailed logging during prefilling
            if (res != ML_SUCCESS) {
                return res;  // error during processing
            }
            embd.clear();
        }
    }

    profiler.prompt_end();
    profiler.update_prompt_tokens(prompt_len - this->n_past_global);
    profiler.decode_start();

    // Process output

    bool                     first_token_generated = false;
    std::vector<llama_token> generated_tokens;
    std::stringstream        full_text;

    while ((int)generated_tokens.size() < cfg.max_tokens) {
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
            profiler.set_stop_reason(common::StopReason::ML_STOP_REASON_EOS);
            break;
        }

        // Convert token to string
        char token_buf[64];
        // Use instance-level setting for special token output
        int n = llama_token_to_piece(vocab, id, token_buf, sizeof(token_buf) - 1, 0, this->allow_special_tokens);
        if (n < 0) return ML_ERROR_LLM_GENERATION_FAILED;
        token_buf[n] = '\0';

        // Check stop sequences
        bool stop_matched = false;
        for (int i = 0; i < cfg.stop_count; ++i) {
            if (cfg.stop[i] && strcmp(token_buf, cfg.stop[i]) == 0) {
                GENIEX_LOG_DEBUG("Stop sequence matched: '{}'", cfg.stop[i]);
                profiler.set_stop_reason(common::StopReason::ML_STOP_REASON_STOP_SEQUENCE);
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
            profiler.set_stop_reason(common::StopReason::ML_STOP_REASON_USER);
            break;
        }
        full_text << token_buf;

        std::vector<llama_token> embd(1, id);
        int32_t                  res = process(embd);
        if (res != ML_SUCCESS) {
            return res;  // error during processing
        }
    }

    // Set stop reason if not already set
    if ((int)generated_tokens.size() >= cfg.max_tokens) {
        profiler.set_stop_reason(common::StopReason::ML_STOP_REASON_LENGTH);
    }
    profiler.decode_end();
    profiler.update_generated_tokens(generated_tokens.size());
    profiler.to_profile_data(output->profile_data);
    output->full_text = strdup(full_text.str().c_str());  // Initialize to null
                                                          //
    this->n_past_global = prompt_len + generated_tokens.size();
    this->past_prompt_tokens.insert(this->past_prompt_tokens.end(), embd_inp.begin(), embd_inp.end());
    this->past_prompt_tokens.insert(this->past_prompt_tokens.end(), generated_tokens.begin(), generated_tokens.end());

    return ML_SUCCESS;
}
}  // namespace geniex

// Private
namespace geniex {
ml_ModelConfig LlamaLlm::model_config_default(void) {
    auto cfg            = ml_ModelConfig{};
    cfg.n_ctx           = 4096;
    cfg.n_threads       = static_cast<int32_t>(std::thread::hardware_concurrency());
    cfg.n_threads_batch = static_cast<int32_t>(std::thread::hardware_concurrency());
    cfg.n_batch         = 2048;
    cfg.n_ubatch        = 512;
    cfg.n_seq_max       = 1;
    return cfg;
}

void LlamaLlm::reset_sampler() {
    if (this->sampler) {
        common_sampler_free(this->sampler);
        this->sampler = nullptr;
    }
    common_params_sampling s;
    this->sampler = common_sampler_init(this->model, s);
}

void LlamaLlm::set_sampler(const ml_SamplerConfig* cfg) {
    if (this->sampler) {
        common_sampler_free(this->sampler);
        this->sampler = nullptr;
    }

    // Convert ml_SamplerConfig to common_params_sampling
    common_params_sampling s;

    if (cfg) {
        // Apply sampling parameters from the config, using defaults for 0/0.0
        // values This matches the pattern from the obsolete ml-llm.cpp
        // implementation
        s.seed            = (cfg->seed != 0) ? cfg->seed : LLAMA_DEFAULT_SEED;
        s.top_k           = (cfg->top_k != 0) ? cfg->top_k : 40;
        s.top_p           = (cfg->top_p != 0.0f) ? cfg->top_p : 0.95f;
        s.min_p           = (cfg->min_p != 0.0f) ? cfg->min_p : 0.05f;
        s.temp            = (cfg->temperature != 0.0f) ? cfg->temperature : 0.8f;
        s.penalty_repeat  = (cfg->repetition_penalty != 0.0f) ? cfg->repetition_penalty : 1.0f;
        s.penalty_present = (cfg->presence_penalty != 0.0f) ? cfg->presence_penalty : 0.0f;
        s.penalty_freq    = (cfg->frequency_penalty != 0.0f) ? cfg->frequency_penalty : 0.0f;

        // Handle grammar configuration - prioritize grammar_string over
        // grammar_path
        if (cfg->grammar_string && strlen(cfg->grammar_string) > 0) {
            s.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, cfg->grammar_string);
            GENIEX_LOG_DEBUG("Applied grammar string: {}", cfg->grammar_string);
        } else if (cfg->grammar_path && strlen(cfg->grammar_path) > 0) {
            // Read grammar from file
            std::ifstream file(cfg->grammar_path);
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                s.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, buffer.str());
                file.close();
                GENIEX_LOG_DEBUG("Applied grammar from file: {}", cfg->grammar_path);
            } else {
                GENIEX_LOG_ERROR("Failed to read grammar file: {}", cfg->grammar_path);
            }
        }
    }
    // Note: if cfg is null, s will use default values from common_params_sampling
    // constructor

    this->sampler = common_sampler_init(this->model, s);
}

}  // namespace geniex
