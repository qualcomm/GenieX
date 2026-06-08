#include "vlm.h"

#include <algorithm>
#include <cstring>
#include <nlohmann/json.hpp>

#include "chat.h"
#include "common.h"
#include "geniex.h"
#include "htp_session.h"
#include "llama.h"
#include "logging.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "params.h"
#include "profiler.h"

namespace geniex {

LlamaVlm::~LlamaVlm() {
    if (this->ctx) {
        llama_free(this->ctx);
        this->ctx = nullptr;
    }
    if (this->ctx_vision) {
        mtmd_free(this->ctx_vision);
        this->ctx_vision = nullptr;
    }
}

int32_t LlamaVlm::create_impl(const geniex_VlmCreateInput* input) {
    if (!input || !input->model_path) {
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    // See llm.cpp for the rationale behind the HTP session release/reacquire
    // dance. Any llama.cpp class that might load onto HTP must participate.
    htp::reacquire_before_load();

    const Device             device = classify_device(input->device_id, input->config.n_gpu_layers);
    const geniex_ModelConfig config = build_model_config(input->config, /*n_ctx_default=*/16384, device);

    llama_model_params mpar      = build_model_params(config);
    auto               selection = resolve_devices(input->device_id);
    if (!selection) {
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }
    if (!selection->empty()) {
        mpar.devices = selection->data();
    }

    // See llm.cpp for why this is registry-scoped rather than per-device.
    if (htp::htp_backend_present()) {
        htp_guard_.mark_htp();
    }

    this->model = llama_model_load_from_file(input->model_path, mpar);
    if (!this->model) {
        llama_model_free(this->model);
        this->model = nullptr;
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }

    llama_context_params cpar = build_context_params(config, device);

    this->ctx = llama_init_from_model(this->model, cpar);
    if (!this->ctx) {
        llama_model_free(this->model);
        this->model = nullptr;
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }

    ggml_threadpool_params tpp_main  = build_threadpool_params(cpar.n_threads, device);
    ggml_threadpool_params tpp_batch = build_threadpool_params(cpar.n_threads_batch, device);
    int32_t                tp_ret    = create_and_attach_threadpools(this->pools_, this->ctx, tpp_main, tpp_batch);
    if (tp_ret != GENIEX_SUCCESS) {
        return tp_ret;
    }

    // Initialize vision context if mmproj_path provided
    if (input->mmproj_path) {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu             = config.n_gpu_layers > 0;
        mparams.print_timings       = false;
        mparams.n_threads           = 4;
        // Zack TODO: elegant fix this error:  no member named 'verbosity' in 'mtmd_context_params'
        // mparams.verbosity           = GGML_LOG_LEVEL_ERROR;

        this->ctx_vision = mtmd_init_from_file(input->mmproj_path, this->model, mparams);
        // Continue even if vision context fails
        if (this->ctx_vision) {
            this->supports_vision = mtmd_support_vision(this->ctx_vision);
            this->supports_audio  = mtmd_support_audio(this->ctx_vision);
            GENIEX_LOG_INFO("mmproj loaded: vision={}, audio={}", this->supports_vision, this->supports_audio);
        }
    }

    this->reset_sampler();

    return GENIEX_SUCCESS;
}

int32_t LlamaVlm::get_capabilities(geniex_VlmCapabilities* output) {
    if (!output) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    output->supports_vision = this->supports_vision;
    output->supports_audio  = this->supports_audio;
    return GENIEX_SUCCESS;
}

int32_t LlamaVlm::reset() {
    if (!this->ctx) return GENIEX_ERROR_COMMON_INVALID_INPUT;

    this->n_past = 0;
    this->past_text_prefix.clear();
    llama_memory_clear(llama_get_memory(this->ctx), /*clear data=*/true);

    return GENIEX_SUCCESS;
}

int32_t LlamaVlm::apply_chat_template(
    const geniex_VlmApplyChatTemplateInput* input, geniex_VlmApplyChatTemplateOutput* output) {
    if (!this->model || !input || !output || !input->messages || input->message_count <= 0)
        return GENIEX_ERROR_COMMON_INVALID_INPUT;

    // Convert geniex_VlmChatMessage array to vector<common_chat_msg>
    std::vector<common_chat_msg> chat_messages;
    chat_messages.reserve(input->message_count);

    for (int32_t i = 0; i < input->message_count; ++i) {
        common_chat_msg msg;
        if (!this->vlm_message_to_common_chat_msg(&input->messages[i], &msg)) {
            GENIEX_LOG_DEBUG("failed to convert message {} (role={})",
                i,
                input->messages[i].role ? input->messages[i].role : "NULL");
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
        chat_messages.push_back(msg);
        GENIEX_LOG_DEBUG(
            "converted message {} - role={}, content_length={}", i, msg.role.c_str(), msg.content.length());
    }

    common_chat_templates_inputs tmpl_inputs;
    tmpl_inputs.messages              = chat_messages;
    tmpl_inputs.add_generation_prompt = true;
    tmpl_inputs.use_jinja             = true;
    if (input->tools && strlen(input->tools) > 0) {
        tmpl_inputs.tools = common_chat_tools_parse_oaicompat(nlohmann::ordered_json::parse(std::string(input->tools)));
    }

    if (input->enable_thinking) {
        GENIEX_LOG_WARN("thinking mode not supported for llama.cpp VLM; ignoring enable_thinking=true");
    }
    tmpl_inputs.enable_thinking = false;
    GENIEX_LOG_DEBUG("applying chat template with add_generation_prompt=true, use_jinja={}", tmpl_inputs.use_jinja);

    // Apply chat template
    common_chat_templates_ptr tmpls  = common_chat_templates_init(this->model, "");
    auto                      result = common_chat_templates_apply(tmpls.get(), tmpl_inputs);

    if (result.prompt.empty()) {
        GENIEX_LOG_ERROR("chat template application resulted in empty prompt");
        return GENIEX_ERROR_COMMON_FILE_NOT_FOUND;
    }

    // Allocate and copy result
    size_t prompt_length = result.prompt.length();
    char*  output_text   = (char*)malloc(prompt_length + 1);
    if (!output_text) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;

    memcpy(output_text, result.prompt.c_str(), prompt_length);
    output_text[prompt_length] = '\0';

    output->formatted_text = output_text;

    GENIEX_LOG_DEBUG("successfully generated prompt with length={}", prompt_length);
    GENIEX_LOG_DEBUG("result text: {}", output_text);

    return GENIEX_SUCCESS;
}

int32_t LlamaVlm::generate(const geniex_VlmGenerateInput* input, geniex_VlmGenerateOutput* output) {
    if (!this->ctx || !input || !output || !input->prompt_utf8) {
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    common::Profiler profiler;
    profiler.prompt_start();

    int32_t res = GENIEX_SUCCESS;

    geniex_GenerationConfig cfg = input->config ? *input->config : geniex_GenerationConfig{};
    if (cfg.max_tokens <= 0) cfg.max_tokens = 512;

    this->set_sampler(cfg.sampler_config);

    // Prepare multimodal input: collect bitmaps for images and audio
    std::vector<mtmd_bitmap*> bitmaps;
    int                       n_media = 0;

    if (input->config && input->config->image_paths && input->config->image_count > 0 && this->ctx_vision) {
        if (!this->supports_vision) {
            GENIEX_LOG_WARN("model does not support image input; skipping {} image(s)", input->config->image_count);
        } else {
            GENIEX_LOG_DEBUG("processing {} image(s)", input->config->image_count);
            for (int i = 0; i < input->config->image_count; ++i) {
                if (input->config->image_paths[i]) {
                    mtmd_bitmap* bmp =
                        mtmd_helper_bitmap_init_from_file(this->ctx_vision, input->config->image_paths[i]);
                    if (bmp) {
                        bitmaps.push_back(bmp);
                        n_media++;
                        GENIEX_LOG_DEBUG("successfully loaded image {}: {}", i, input->config->image_paths[i]);
                    } else {
                        GENIEX_LOG_DEBUG("failed to load image {}: {}", i, input->config->image_paths[i]);
                    }
                }
            }
        }
    }

    if (input->config && input->config->audio_paths && input->config->audio_count > 0 && this->ctx_vision) {
        if (!this->supports_audio) {
            GENIEX_LOG_WARN(
                "model does not support audio input; skipping {} audio file(s)", input->config->audio_count);
        } else {
            GENIEX_LOG_DEBUG("processing {} audio file(s)", input->config->audio_count);
            for (int i = 0; i < input->config->audio_count; ++i) {
                if (input->config->audio_paths[i]) {
                    mtmd_bitmap* bmp =
                        mtmd_helper_bitmap_init_from_file(this->ctx_vision, input->config->audio_paths[i]);
                    if (bmp) {
                        bitmaps.push_back(bmp);
                        n_media++;
                        GENIEX_LOG_DEBUG("successfully loaded audio {}: {}", i, input->config->audio_paths[i]);
                    } else {
                        GENIEX_LOG_DEBUG("failed to load audio {}: {}", i, input->config->audio_paths[i]);
                    }
                }
            }
        }
    }

    GENIEX_LOG_DEBUG("total media files loaded: {}", n_media);

    // The caller passes the whole conversation re-templated every turn, with the
    // image marker(s) from earlier turns still in it. Image chunks can't be reused
    // across turns (their embeddings are re-encoded and the encoder state isn't a
    // plain token prefix), so the multimodal path clears the KV cache and decodes
    // the full prompt from scratch each turn. n_past is taken authoritatively from
    // mtmd_helper_eval_chunks (token count for text, n_pos for image/audio under
    // M-RoPE), which is the whole point of the fix; the text-only path below still
    // does token-prefix reuse.
    if (this->ctx_vision) {
        GENIEX_LOG_DEBUG("using multimodal (mtmd) path with ctx_vision");

        llama_memory_clear(llama_get_memory(this->ctx), /*clear data=*/true);
        this->n_past = 0;

        mtmd_input_text text;
        text.text          = input->prompt_utf8;
        text.add_special   = true;
        text.parse_special = true;

        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        if (!chunks) {
            for (auto bmp : bitmaps) mtmd_bitmap_free(bmp);
            return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
        }

        int32_t tok_ret = mtmd_tokenize(this->ctx_vision, chunks, &text, (const mtmd_bitmap**)bitmaps.data(), n_media);
        for (auto bmp : bitmaps) mtmd_bitmap_free(bmp);
        if (tok_ret != 0) {
            mtmd_input_chunks_free(chunks);
            GENIEX_LOG_ERROR("mtmd_tokenize failed");
            return GENIEX_ERROR_VLM_GENERATION_FAILED;
        }

        // Evaluate all chunks. A return of 1 means the prompt does not fit the KV
        // cache: report truncation, not a generic failure.
        llama_pos new_n_past = this->n_past;
        switch (mtmd_helper_eval_chunks(
            this->ctx_vision, this->ctx, chunks, this->n_past, 0, llama_n_batch(this->ctx), true, &new_n_past)) {
            case 0:
                profiler.update_prompt_tokens(new_n_past - this->n_past);
                this->n_past = new_n_past;
                break;
            case 1:
                res = GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
                break;
            default:
                mtmd_input_chunks_free(chunks);
                GENIEX_LOG_ERROR("mtmd_helper_eval_chunks failed");
                return GENIEX_ERROR_VLM_GENERATION_FAILED;
        }

        GENIEX_LOG_DEBUG("mtmd eval successful, n_past -> {}", this->n_past);
        mtmd_input_chunks_free(chunks);

        profiler.prompt_end();
        profiler.decode_start();
    } else {
        GENIEX_LOG_DEBUG("using text-only (direct llama) path");

        const llama_vocab*       vocab = llama_model_get_vocab(this->model);
        std::vector<llama_token> prompt_ids;
        try {
            prompt_ids = common_tokenize(vocab, std::string(input->prompt_utf8), true, true);
        } catch (const std::exception&) {
            return GENIEX_ERROR_LLM_TOKENIZATION_FAILED;
        }
        if (prompt_ids.empty()) return GENIEX_ERROR_LLM_TOKENIZATION_FAILED;

        int match_len = 0;
        while (match_len < std::min((int)past_text_prefix.size(), (int)prompt_ids.size()) &&
               past_text_prefix[match_len] == prompt_ids[match_len]) {
            match_len++;
        }
        if (match_len < this->n_past) {
            llama_memory_seq_rm(llama_get_memory(this->ctx), 0, match_len, -1);
            this->n_past = match_len;
        }

        const int n_new = (int)prompt_ids.size() - match_len;
        profiler.prompt_end();
        profiler.update_prompt_tokens(n_new);
        profiler.decode_start();

        if (n_new > 0) {
            llama_batch batch = llama_batch_get_one(prompt_ids.data() + match_len, n_new);
            for (int i = 0; i < batch.n_tokens; ++i) {
                batch.pos[i] = this->n_past + i;
            }
            // 1 means the prompt does not fit the KV cache: report truncation, not a generic failure.
            switch (llama_decode(this->ctx, batch)) {
                case 0:
                    this->n_past += n_new;
                    break;
                case 1:
                    res = GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
                    break;
                default:
                    GENIEX_LOG_ERROR("llama_decode failed");
                    res = GENIEX_ERROR_VLM_GENERATION_FAILED;
                    break;
            }
        }
        this->past_text_prefix = std::move(prompt_ids);
        GENIEX_LOG_DEBUG("text-only decode successful, n_past updated to {}", this->n_past);
    }

    // Generate tokens (common for both multimodal and text-only)
    GENIEX_LOG_DEBUG("starting token generation loop");
    const llama_vocab* vocab = llama_model_get_vocab(this->model);
    char               token_buffer[256];

    // Create reusable batch like mtmd-cli.cpp
    llama_batch batch = llama_batch_init(1, 0, 1);

    std::stringstream full_text;
    int32_t           generated_token_count = 0;
    while (res == GENIEX_SUCCESS && generated_token_count < cfg.max_tokens) {
        llama_token token = common_sampler_sample(this->sampler, this->ctx, -1);

        // Measure TTFT on first token
        profiler.record_ttft();

        // Accept the token first (like mtmd-cli.cpp does)
        common_sampler_accept(this->sampler, token, true);

        if (llama_vocab_is_eog(vocab, token)) {
            GENIEX_LOG_DEBUG("reached end of generation token");
            profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_EOS);
            break;
        }

        int n = llama_token_to_piece(vocab, token, token_buffer, sizeof(token_buffer) - 1, 0, false);
        if (n < 0) n = 0;
        token_buffer[n] = '\0';

        // Check stop sequences
        bool stop_matched = false;
        for (int i = 0; i < cfg.stop_count; ++i) {
            if (cfg.stop[i] && strcmp(token_buffer, cfg.stop[i]) == 0) {
                GENIEX_LOG_DEBUG("Stop sequence matched: '{}'", cfg.stop[i]);
                profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_STOP_SEQUENCE);
                stop_matched = true;
                break;
            }
        }
        if (stop_matched) {
            break;
        }

        generated_token_count++;

        // Call the callback directly (UTF-8 validation is now handled at bridge level)
        if (input->on_token) {
            if (!input->on_token(token_buffer, input->user_data)) {
                GENIEX_LOG_WARN("User callback requested stop during token generation");
                profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_USER);
                break;
            }
        }
        full_text << token_buffer;

        // Decode next token using reusable batch (like mtmd-cli.cpp). 1 means
        // the KV cache is full (truncate); other non-zero values are failures.
        common_batch_clear(batch);
        common_batch_add(batch, token, this->n_past++, {0}, true);
        switch (llama_decode(this->ctx, batch)) {
            case 0:
                break;
            case 1:
                res = GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
                break;
            default:
                res = GENIEX_ERROR_VLM_GENERATION_FAILED;
                break;
        }
    }

    llama_batch_free(batch);

    // Set stop reason if not already set
    if (res == GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH) {
        GENIEX_LOG_WARN("VLM generate: context window ({}) exhausted; truncating", llama_n_ctx(this->ctx));
        profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_LENGTH);
    } else if (generated_token_count >= cfg.max_tokens) {
        profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_LENGTH);
    }

    // Record decode processing end
    profiler.decode_end();
    profiler.update_generated_tokens(generated_token_count);
    profiler.to_profile_data(output->profile_data);

    output->full_text = strdup(full_text.str().c_str());

    GENIEX_LOG_DEBUG("completed generation with {} tokens", generated_token_count);
    return res;
}

}  // namespace geniex

namespace geniex {

void LlamaVlm::reset_sampler() { this->set_sampler(nullptr); }

void LlamaVlm::set_sampler(const geniex_SamplerConfig* cfg) {
    if (this->sampler) {
        common_sampler_free(this->sampler);
        this->sampler = nullptr;
    }
    common_params_sampling s = build_sampling_params(cfg);
    this->sampler            = common_sampler_init(this->model, s);
}

bool LlamaVlm::vlm_message_to_common_chat_msg(const geniex_VlmChatMessage* input, common_chat_msg* output) {
    if (!input || !output) return false;

    // Role is required
    if (!input->role || strlen(input->role) == 0) {
        return false;
    }

    output->role = input->role;

    if (input->contents && input->content_count > 0) {
        int         media_count = 0;
        std::string consolidated_text;

        // First pass: validate types and count media, concatenate text content
        for (int64_t j = 0; j < input->content_count; ++j) {
            // Type is required for each content part
            if (!input->contents[j].type || strlen(input->contents[j].type) == 0) {
                return false;
            }

            if (strcmp(input->contents[j].type, "text") == 0) {
                // Concatenate all text content
                if (input->contents[j].text) {
                    consolidated_text += input->contents[j].text;
                }
            } else {
                // Count non-text content as media
                media_count++;
            }
        }

        // We consolidate all content into a single content, this aligns with mtmd-cli.cpp
        // It would be meaningless to pass "non-text" type to common_chat_template_apply, as all non-text content in the
        // content_parts will be ignored by llama.cpp by the time I am writing this comment.
        std::string final_content;
        for (int i = 0; i < media_count; ++i) {
            final_content += mtmd_default_marker();
        }
        final_content += consolidated_text;

        output->content = final_content;
    }

    return true;
}

}  // namespace geniex
