#include "vlm.h"

#include <cstring>
#include <nlohmann/json.hpp>

#include "chat.h"
#include "common.h"
#include "geniex.h"
#include "llama.h"
#include "logging.h"
#include "mtmd-helper.h"
#include "mtmd.h"
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

    llama_model_params mpar = llama_model_default_params();
    mpar.use_mmap           = true;
    mpar.use_mlock          = false;
    mpar.n_gpu_layers       = input->config.n_gpu_layers;
    if (input->device_id) {
        auto device = ggml_backend_dev_by_name(input->device_id);
        if (!device) {
            // Device not found, log warning and continue with default device
            GENIEX_LOG_ERROR("Device '{}' not found", input->device_id);
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        } else {
            // Create a NULL-terminated array with the device
            static ggml_backend_dev_t device_array[2];
            device_array[0] = device;
            device_array[1] = nullptr;  // NULL-terminated
            mpar.devices    = device_array;
        }
    }

    this->model = llama_model_load_from_file(input->model_path, mpar);
    if (!this->model) {
        llama_model_free(this->model);
        this->model = nullptr;
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }

    llama_context_params cpar = llama_context_default_params();
    cpar.n_ctx                = input->config.n_ctx > 0 ? input->config.n_ctx : 16384;
    cpar.n_batch              = 4096;

    this->ctx = llama_init_from_model(this->model, cpar);
    if (!this->ctx) {
        llama_model_free(this->model);
        this->model = nullptr;
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }

    // Initialize vision context if mmproj_path provided
    if (input->mmproj_path) {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu             = input->config.n_gpu_layers > 0;
        mparams.print_timings       = false;
        mparams.n_threads           = 4;
        // Zack TODO: elegant fix this error:  no member named 'verbosity' in 'mtmd_context_params'
        // mparams.verbosity           = GGML_LOG_LEVEL_ERROR;

        this->ctx_vision = mtmd_init_from_file(input->mmproj_path, this->model, mparams);
        // Continue even if vision context fails
    }

    this->reset_sampler();

    return GENIEX_SUCCESS;
}

int32_t LlamaVlm::reset() {
    if (!this->ctx) return GENIEX_ERROR_COMMON_INVALID_INPUT;

    // Clear memory keeping BOS token (like mtmd-cli.cpp does)
    llama_memory_seq_rm(llama_get_memory(this->ctx), 0, 1, -1);

    // Reset conversation state, setting to n_past = 1 since we preserved the BOS token above.
    // TODO: revisit and verify that setting to 1 is correct. mtmd-cli.cpp in llama-cpp sets it to 0 but setting to 0
    // will cause all test cases to fail.
    this->n_past              = 1;
    this->global_n_past_chars = 0;

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
    tmpl_inputs.use_jinja             = false;
    if (input->tools && strlen(input->tools) > 0) {
        tmpl_inputs.use_jinja = true;  // jinja can be buggy, stay consistent with generation code
        tmpl_inputs.tools = common_chat_tools_parse_oaicompat(nlohmann::ordered_json::parse(std::string(input->tools)));
    }

    tmpl_inputs.enable_thinking = input->enable_thinking;
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

    geniex_GenerationConfig cfg = input->config ? *input->config : geniex_GenerationConfig{};
    if (cfg.max_tokens <= 0) cfg.max_tokens = 512;

    // Prepare multimodal input: collect bitmaps for images and audio
    std::vector<mtmd_bitmap*> bitmaps;
    int                       n_media = 0;

    if (input->config && input->config->image_paths && input->config->image_count > 0 && this->ctx_vision) {
        GENIEX_LOG_DEBUG("processing {} image(s)", input->config->image_count);
        for (int i = 0; i < input->config->image_count; ++i) {
            if (input->config->image_paths[i]) {
                mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(this->ctx_vision, input->config->image_paths[i]);
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

    if (input->config && input->config->audio_paths && input->config->audio_count > 0 && this->ctx_vision) {
        GENIEX_LOG_DEBUG("processing %d audio file(s)", input->config->audio_count);
        for (int i = 0; i < input->config->audio_count; ++i) {
            if (input->config->audio_paths[i]) {
                mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(this->ctx_vision, input->config->audio_paths[i]);
                if (bmp) {
                    bitmaps.push_back(bmp);
                    n_media++;
                    GENIEX_LOG_DEBUG("successfully loaded audio %d: %s", i, input->config->audio_paths[i]);
                } else {
                    GENIEX_LOG_DEBUG("failed to load audio %d: %s", i, input->config->audio_paths[i]);
                }
            }
        }
    }

    GENIEX_LOG_DEBUG("total media files loaded: {}", n_media);

    // Get the full prompt length for tracking
    const int32_t full_prompt_len = (int32_t)strlen(input->prompt_utf8);
    GENIEX_LOG_DEBUG("full prompt length: {}, global_text_pos: {}", full_prompt_len, this->global_n_past_chars);

    // Extract only the new text portion that hasn't been processed yet
    std::string new_text_portion;
    if (this->global_n_past_chars < full_prompt_len) {
        new_text_portion = std::string(input->prompt_utf8 + this->global_n_past_chars);
        GENIEX_LOG_DEBUG("new text portion length: {}", new_text_portion.length());
    } else {
        GENIEX_LOG_DEBUG("no new text to process (global_text_pos >= full_prompt_len)");
    }

    // Use mtmd path when ctx_vision is available, fallback to direct llama path otherwise
    if (this->ctx_vision) {
        GENIEX_LOG_DEBUG("using multimodal (mtmd) path with ctx_vision");

        // Only process if there's new text content
        if (!new_text_portion.empty()) {
            // prompt_utf8 already has chat template and media markers applied
            mtmd_input_text text;
            text.text          = new_text_portion.c_str();
            text.add_special   = this->n_past == 0;  // add BOS only on first message
            text.parse_special = true;

            mtmd_input_chunks* chunks = mtmd_input_chunks_init();
            if (!chunks) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;

            int32_t res = mtmd_tokenize(this->ctx_vision, chunks, &text, (const mtmd_bitmap**)bitmaps.data(), n_media);
            if (res != 0) {
                mtmd_input_chunks_free(chunks);
                for (auto bmp : bitmaps) {
                    if (bmp) mtmd_bitmap_free(bmp);
                }
                GENIEX_LOG_ERROR("mtmd_tokenize failed");
                return GENIEX_ERROR_VLM_GENERATION_FAILED;
            }

            GENIEX_LOG_DEBUG("mtmd_tokenize successful");

            // Clear bitmaps (like eval_message does)
            for (auto bmp : bitmaps) mtmd_bitmap_free(bmp);

            // Evaluate chunks (like eval_message does)
            llama_pos new_n_past;
            if (mtmd_helper_eval_chunks(this->ctx_vision,
                    this->ctx,
                    chunks,
                    this->n_past,
                    0,
                    llama_n_batch(this->ctx),
                    true,
                    &new_n_past)) {
                mtmd_input_chunks_free(chunks);
                GENIEX_LOG_ERROR("mtmd_helper_eval_chunks failed");
                return GENIEX_ERROR_VLM_GENERATION_FAILED;
            }

            GENIEX_LOG_DEBUG("mtmd_helper_eval_chunks successful, n_past: {} -> {}", this->n_past, new_n_past);
            profiler.update_prompt_tokens(new_n_past - this->n_past);
            this->n_past = new_n_past;
            mtmd_input_chunks_free(chunks);
        } else {
            // No new content to process, just clear bitmaps
            for (auto bmp : bitmaps) mtmd_bitmap_free(bmp);
            GENIEX_LOG_DEBUG("no new text content, skipping mtmd processing");
        }

        profiler.prompt_end();
        profiler.decode_start();
    } else {
        GENIEX_LOG_DEBUG("using text-only (direct llama) path");

        // Only process if there's new text content
        if (!new_text_portion.empty()) {
            // Fallback to direct llama path when ctx_vision is not available
            const llama_vocab* vocab = llama_model_get_vocab(this->model);

            // Get required length
            int32_t needed = llama_tokenize(
                vocab, new_text_portion.c_str(), (int32_t)new_text_portion.length(), nullptr, 0, true, true);
            if (needed < 0) needed = -needed;
            if (needed == 0) return GENIEX_ERROR_LLM_TOKENIZATION_FAILED;

            // Allocate and tokenize
            int32_t* prompt_tokens = (int32_t*)malloc(sizeof(int32_t) * needed);
            if (!prompt_tokens) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;

            int32_t prompt_len = llama_tokenize(
                vocab, new_text_portion.c_str(), (int32_t)new_text_portion.length(), prompt_tokens, needed, true, true);
            if (prompt_len < 0) {
                free(prompt_tokens);
                return GENIEX_ERROR_LLM_TOKENIZATION_FAILED;
            }

            GENIEX_LOG_DEBUG("_ml_vlm_generate_internal: Tokenized new text portion into {} tokens", prompt_len);

            // Record prompt processing end and decode start
            profiler.prompt_end();
            profiler.update_prompt_tokens(prompt_len);
            profiler.decode_start();

            llama_batch batch = llama_batch_get_one(prompt_tokens, prompt_len);
            // Set positions based on current n_past
            for (int i = 0; i < batch.n_tokens; ++i) {
                batch.pos[i] = this->n_past + i;
            }

            if (llama_decode(this->ctx, batch) != 0) {
                free(prompt_tokens);
                GENIEX_LOG_ERROR("llama_decode failed");
                return GENIEX_ERROR_VLM_GENERATION_FAILED;
            }
            free(prompt_tokens);
            this->n_past += prompt_len;
            GENIEX_LOG_DEBUG("llama_decode successful, n_past updated to {}", this->n_past);
        } else {
            GENIEX_LOG_DEBUG("no new text content, skipping direct llama processing");
        }
    }

    // Generate tokens (common for both multimodal and text-only)
    GENIEX_LOG_DEBUG("starting token generation loop");
    const llama_vocab* vocab = llama_model_get_vocab(this->model);
    char               token_buffer[256];

    // Create reusable batch like mtmd-cli.cpp
    llama_batch batch = llama_batch_init(1, 0, 1);

    std::stringstream full_text;
    int32_t           generated_token_count = 0;
    for (int i = 0; i < cfg.max_tokens; ++i) {
        llama_token token = common_sampler_sample(this->sampler, this->ctx, -1);

        // Measure TTFT on first token
        profiler.record_ttft();

        // Accept the token first (like mtmd-cli.cpp does)
        common_sampler_accept(this->sampler, token, true);

        if (llama_vocab_is_eog(vocab, token)) {
            GENIEX_LOG_DEBUG("reached end of generation token at step {}", i);
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
                GENIEX_LOG_DEBUG("generation stopped by callback or stop sequence at step {}", i);
                GENIEX_LOG_WARN("User callback requested stop during token generation");
                profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_USER);
                break;
            }
        }
        full_text << token_buffer;

        // Decode next token using reusable batch (like mtmd-cli.cpp)
        common_batch_clear(batch);
        common_batch_add(batch, token, this->n_past++, {0}, true);
        if (llama_decode(this->ctx, batch) != 0) break;
    }

    if (generated_token_count >= cfg.max_tokens) {
        GENIEX_LOG_DEBUG("reached max_tokens limit ({})", cfg.max_tokens);
        profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_LENGTH);
    }

    // Clean up batch
    llama_batch_free(batch);

    // Record decode processing end
    profiler.decode_end();
    profiler.update_generated_tokens(generated_token_count);
    profiler.to_profile_data(output->profile_data);

    // Generate output text
    if (generated_token_count == 0) {
        // output->full_text = (char*)malloc(1);
        // if (!output->full_text) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
        // output->full_text[0] = '\0';
        GENIEX_LOG_DEBUG("no tokens generated, returning empty string");
        return GENIEX_SUCCESS;
    } else {
        auto full_text_str        = full_text.str();
        output->full_text         = strdup(full_text_str.c_str());
        this->global_n_past_chars = full_prompt_len + full_text_str.length();

        GENIEX_LOG_DEBUG("updated global_text_pos to {} (prompt_len={} + generated_len={})",
            this->global_n_past_chars,
            full_prompt_len,
            full_text_str.length());
    }

    GENIEX_LOG_DEBUG("completed generation with {} tokens (no text output requested)", generated_token_count);
    return GENIEX_SUCCESS;
}

}  // namespace geniex

namespace geniex {

void LlamaVlm::reset_sampler() {
    // Free existing sampler
    if (this->sampler) {
        common_sampler_free(this->sampler);
        this->sampler = nullptr;
    }

    // Use default values from common_params_sampling (like ml-llm.cpp)
    common_params_sampling s;
    this->sampler = common_sampler_init(this->model, s);
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
