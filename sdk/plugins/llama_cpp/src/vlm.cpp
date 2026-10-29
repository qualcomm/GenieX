// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "vlm.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

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

namespace {
constexpr size_t  kOpenClSmallAllocLimit       = size_t{1} << 30;
constexpr int32_t kOpenClSmallAllocImageTokens = 1024;
}  // namespace

LlamaVlm::~LlamaVlm() {
    // ctx_vision and ctx hold pointers into model; free them first.
    if (this->ctx_vision) {
        mtmd_free(this->ctx_vision);
        this->ctx_vision = nullptr;
    }
    if (this->ctx) {
        llama_free(this->ctx);
        this->ctx = nullptr;
    }
    if (this->model) {
        llama_model_free(this->model);
        this->model = nullptr;
    }
}

int32_t LlamaVlm::create(const geniex_VlmCreateInput* input) {
    if (!input || !input->model_path) {
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    const Device              device = classify_device(input->device_id, input->config.n_gpu_layers);
    const geniex_ModelConfig& config = input->config;

    // See llm.cpp: reacquire whenever the HTP backend is registered, since
    // any llama.cpp load walks the registry's device list and a stale session
    // pointer left from a prior release will crash the load on cpu / gpu too.
    if (htp::htp_backend_present()) {
        if (device == Device::NPU) {
            htp::set_power_mode(config.power_mode);
        } else if (config.power_mode != GENIEX_POWER_MODE_BURST) {
            GENIEX_LOG_WARN("power_mode is only meaningful on the NPU device; ignoring on this device");
        }
        htp::reacquire_before_load();
    }

    llama_model_params mpar      = build_model_params(config, device);
    auto               selection = resolve_devices(input->device_id);
    if (!selection) {
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }
    select_iq9_npu_devices(*selection, input->model_path);

    // Pick the mmproj device from the caller's explicit selection, before apply_device_selection fills in defaults.
    const bool explicit_selection = !selection->empty();
    apply_device_selection(mpar, *selection, device);

    ggml_backend_dev_t vision_device = nullptr;
    if (input->vit_device_id && input->vit_device_id[0] != '\0') {
        vision_device = ggml_backend_dev_by_name(input->vit_device_id);
        if (!vision_device) {
            GENIEX_LOG_ERROR("Vision device '{}' not found", input->vit_device_id);
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
        GENIEX_LOG_INFO("Using vision device override: {}", input->vit_device_id);
    } else if (explicit_selection) {
        vision_device = resolve_vision_device(*selection);
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

    // 16384 is fine on CPU/GPU, but on IQ9 even an otherwise-safe n_ubatch still overflows HTP's
    // fastrpc/CMA pool once n_ctx (and thus the KV cache) gets that large. See qcom-ai-hub/geniex#1683.
    const int32_t        n_ctx_default = device == Device::NPU ? 4096 : 16384;
    llama_context_params cpar =
        build_context_params(config, n_ctx_default, device, nullptr, OutputReservation::LastPerSequence);
    this->ctx = llama_init_from_model(this->model, cpar);
    if (!this->ctx) {
        llama_model_free(this->model);
        this->model = nullptr;
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }

    ggml_threadpool_params tpp_main  = build_threadpool_params(cpar.n_threads, device);
    ggml_threadpool_params tpp_batch = build_threadpool_params(cpar.n_threads_batch, device);
    int32_t                tp_ret    = this->pools_.attach(this->ctx, tpp_main, tpp_batch);
    if (tp_ret != GENIEX_SUCCESS) {
        return tp_ret;
    }

    warmup_context(this->model, this->ctx);

    // Initialize vision context if mmproj_path provided
    if (input->mmproj_path) {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu             = false;
        if (vision_device) {
            mparams.use_gpu = true;
            mparams.device  = vision_device;
        }
        mparams.print_timings   = false;
        mparams.n_threads       = cpar.n_threads;
        mparams.flash_attn_type = cpar.flash_attn_type;
        if (vision_device && std::strncmp(ggml_backend_dev_name(vision_device), "GPUOpenCL", 9) == 0 &&
            ggml_backend_buft_get_max_size(ggml_backend_dev_buffer_type(vision_device)) <= kOpenClSmallAllocLimit) {
            // The fixed warmup graph needs a multi-GiB attention buffer; mtmd ignores the failed reserve and crashes.
            mparams.warmup           = false;
            mparams.image_max_tokens = kOpenClSmallAllocImageTokens;
        }
        // Zack TODO: elegant fix this error:  no member named 'verbosity' in 'mtmd_context_params'
        // mparams.verbosity           = GGML_LOG_LEVEL_ERROR;

        this->ctx_vision = mtmd_init_from_file(input->mmproj_path, this->model, mparams);
        // A spare HTP session may be unavailable on some targets (e.g. SM8850 cannot reserve a second virtual
        // session). Retry on the LM's own device, then on CPU, instead of dropping the encoder.
        if (!this->ctx_vision && vision_device && !selection->empty() && vision_device != selection->front()) {
            GENIEX_LOG_WARN("mtmd failed to initialize the vision encoder on {}; retrying on {}",
                ggml_backend_dev_name(vision_device),
                ggml_backend_dev_name(selection->front()));
            mparams.device   = selection->front();
            this->ctx_vision = mtmd_init_from_file(input->mmproj_path, this->model, mparams);
        }
        if (!this->ctx_vision && vision_device) {
            GENIEX_LOG_WARN("mtmd failed to initialize the vision encoder on HTP; falling back to CPU");
            mparams.use_gpu  = false;
            mparams.device   = nullptr;
            this->ctx_vision = mtmd_init_from_file(input->mmproj_path, this->model, mparams);
        }
        // Continue even if vision context fails
        if (this->ctx_vision) {
            this->supports_vision = mtmd_support_vision(this->ctx_vision);
            this->supports_audio  = mtmd_support_audio(this->ctx_vision);
            GENIEX_LOG_INFO("mmproj loaded: vision={}, audio={}", this->supports_vision, this->supports_audio);
        }
    }

    this->set_sampler(nullptr);

    return GENIEX_SUCCESS;
}

int32_t LlamaVlm::get_capabilities(geniex_VlmCapabilities* output) {
    if (!output) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    output->supports_vision = this->supports_vision;
    output->supports_audio  = this->supports_audio;
    return GENIEX_SUCCESS;
}

int32_t LlamaVlm::decide(const geniex_DecideInput* input, geniex_DecideOutput* output) {
    if (!this->ctx || !this->model || !input || !output) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    DecisionPrepared prepared;
    int32_t          rc = prepare_decision(this->model, input, &prepared);
    if (rc != GENIEX_SUCCESS) return rc;
    if (prepared.type == "lev") return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    const std::string&              prompt     = prepared.prompt;
    const std::vector<std::string>& candidates = prepared.candidates;
    std::vector<float>              candidate_logits;
    int32_t                         input_tokens = 0;
    // Each decision starts with empty KV state; restore that state even on failure.
    this->reset();
    struct ResetGuard {
        LlamaVlm* self;
        ~ResetGuard() { self->reset(); }
    } reset_guard{this};

    const llama_vocab* vocab         = llama_model_get_vocab(this->model);
    const auto         prompt_tokens = common_tokenize(vocab, prompt, true, true);
    if (prompt_tokens.empty() || prompt_tokens.size() > static_cast<size_t>(llama_n_ctx(this->ctx))) {
        return GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
    }
    // Check that every answer code remains one unique token after this prompt.
    std::vector<llama_token> candidate_tokens;
    for (const auto& candidate : candidates) {
        if (candidate.empty()) return GENIEX_ERROR_COMMON_INVALID_INPUT;
        const auto literal = common_tokenize(vocab, candidate, false, false);
        const auto joined  = common_tokenize(vocab, prompt + candidate, true, true);
        if (literal.size() != 1 || joined.size() != prompt_tokens.size() + 1 ||
            !std::equal(prompt_tokens.begin(), prompt_tokens.end(), joined.begin()) || joined.back() != literal[0] ||
            std::find(candidate_tokens.begin(), candidate_tokens.end(), literal[0]) != candidate_tokens.end()) {
            GENIEX_LOG_WARN("vlm decision: candidate '{}' is not a unique single token after the prompt", candidate);
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
        candidate_tokens.push_back(literal[0]);
    }

    // mtmd binds bitmaps to prompt markers in order, so extra markers are unsafe.
    const std::string marker  = mtmd_default_marker();
    size_t            markers = 0;
    for (size_t pos = 0; (pos = prompt.find(marker, pos)) != std::string::npos; pos += marker.size()) ++markers;
    const size_t media_count = static_cast<size_t>(input->image_count + input->audio_count);
    if (markers != media_count) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    if (media_count > 0 && !this->ctx_vision) return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    if (input->image_count > 0 && !this->supports_vision) return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    if (input->audio_count > 0 && !this->supports_audio) return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;

    candidate_logits.resize(candidate_tokens.size());
    if (media_count == 0) {
        // Text-only VLM decisions can decode directly without an mtmd context.
        const int32_t n_batch = llama_n_batch(this->ctx);
        common_batch  batch(this->ctx);
        int32_t       result = GENIEX_SUCCESS;
        for (size_t start = 0; start < prompt_tokens.size(); start += n_batch) {
            const int32_t count = std::min<int32_t>(n_batch, prompt_tokens.size() - start);
            batch.clear();
            for (int32_t i = 0; i < count; ++i) {
                const size_t position = start + i;
                batch.add(
                    prompt_tokens[position], static_cast<llama_pos>(position), 0, position + 1 == prompt_tokens.size());
            }
            const int32_t rc = llama_process(this->ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (rc != 0) {
                result = rc == 1 ? GENIEX_ERROR_LLM_GENERATION_PROMPT_TOO_LONG : GENIEX_ERROR_VLM_GENERATION_FAILED;
                break;
            }
        }
        const float* row = result == GENIEX_SUCCESS ? llama_get_logits_ith(this->ctx, -1) : nullptr;
        if (!row) result = result == GENIEX_SUCCESS ? GENIEX_ERROR_VLM_GENERATION_FAILED : result;
        if (result == GENIEX_SUCCESS) {
            for (size_t i = 0; i < candidate_tokens.size(); ++i) candidate_logits[i] = row[candidate_tokens[i]];
            input_tokens = static_cast<int32_t>(prompt_tokens.size());
        }
        if (result != GENIEX_SUCCESS) return result;
    } else {
        // Decode image and audio bitmaps through mtmd before reading logits.
        std::vector<mtmd_bitmap*> bitmaps;
        auto                      free_bitmaps = [&]() {
            for (auto* bitmap : bitmaps) mtmd_bitmap_free(bitmap);
            bitmaps.clear();
        };
        for (int32_t i = 0; i < input->image_count; ++i) {
            auto loaded = mtmd_helper_bitmap_init_from_file(
                this->ctx_vision, input->image_paths[i], false, mtmd_helper_init_opt_default());
            if (!loaded.bitmap) {
                free_bitmaps();
                return GENIEX_ERROR_VLM_IMAGE_LOAD;
            }
            bitmaps.push_back(loaded.bitmap);
        }
        for (int32_t i = 0; i < input->audio_count; ++i) {
            auto loaded = mtmd_helper_bitmap_init_from_file(
                this->ctx_vision, input->audio_paths[i], false, mtmd_helper_init_opt_default());
            if (!loaded.bitmap) {
                free_bitmaps();
                return GENIEX_ERROR_VLM_AUDIO_LOAD;
            }
            bitmaps.push_back(loaded.bitmap);
        }
        // One mtmd tokenization and forward pass covers both text and all media.
        mtmd_input_text    text{prompt.c_str(), prompt.size(), true, true};
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        if (!chunks) {
            free_bitmaps();
            return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
        }
        const int32_t tokenized =
            mtmd_tokenize(this->ctx_vision, chunks, &text, (const mtmd_bitmap**)bitmaps.data(), bitmaps.size());
        free_bitmaps();
        if (tokenized != 0) {
            mtmd_input_chunks_free(chunks);
            return GENIEX_ERROR_VLM_GENERATION_FAILED;
        }
        llama_pos n_past      = 0;
        bool      have_logits = false;
        int32_t   result      = GENIEX_SUCCESS;
        // The final text chunk must request logits for the next decision token.
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            const mtmd_input_chunk* chunk       = mtmd_input_chunks_get(chunks, i);
            const bool              is_media    = mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_TEXT;
            const bool              is_last     = i + 1 == mtmd_input_chunks_size(chunks);
            llama_pos               next_n_past = n_past;
            int32_t                 rc;
            if (is_media) {
                if (mtmd_encode_chunk(this->ctx_vision, chunk) != 0) {
                    result = GENIEX_ERROR_VLM_GENERATION_FAILED;
                    break;
                }
                rc = mtmd_helper_decode_image_chunk(this->ctx_vision,
                    this->ctx,
                    chunk,
                    mtmd_get_output_embd(this->ctx_vision),
                    n_past,
                    0,
                    llama_n_batch(this->ctx),
                    &next_n_past,
                    nullptr,
                    nullptr);
            } else {
                rc = mtmd_helper_eval_chunk_single(
                    this->ctx_vision, this->ctx, chunk, n_past, 0, llama_n_batch(this->ctx), is_last, &next_n_past);
                have_logits = is_last;
            }
            if (rc != 0) {
                result = rc == 1 ? GENIEX_ERROR_LLM_GENERATION_PROMPT_TOO_LONG : GENIEX_ERROR_VLM_GENERATION_FAILED;
                break;
            }
            input_tokens += static_cast<int32_t>(mtmd_input_chunk_get_n_tokens(chunk));
            n_past = next_n_past;
        }
        if (result == GENIEX_SUCCESS && !have_logits) result = GENIEX_ERROR_COMMON_INVALID_INPUT;
        if (result == GENIEX_SUCCESS) {
            const float* row = llama_get_logits_ith(this->ctx, -1);
            if (!row)
                result = GENIEX_ERROR_VLM_GENERATION_FAILED;
            else
                for (size_t i = 0; i < candidate_tokens.size(); ++i) candidate_logits[i] = row[candidate_tokens[i]];
        }
        mtmd_input_chunks_free(chunks);
        if (result != GENIEX_SUCCESS) return result;
    }
    // Restore caller option order before exposing the logits through the C API.
    float* logits = static_cast<float*>(calloc(prepared.candidates.size(), sizeof(float)));
    if (!logits) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
    for (size_t i = 0; i < prepared.original_index.size(); ++i)
        logits[prepared.original_index[i]] = candidate_logits[i];
    output->logits        = logits;
    output->option_count  = static_cast<int32_t>(prepared.original_index.size());
    output->input_tokens  = input_tokens;
    output->temperature   = prepared.temperature;
    output->decision_type = strdup(prepared.type.c_str());
    if (!output->decision_type) {
        free(output->logits);
        std::memset(output, 0, sizeof(*output));
        return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
    }
    return GENIEX_SUCCESS;
}

int32_t LlamaVlm::reset() {
    if (!this->ctx) return GENIEX_ERROR_COMMON_INVALID_INPUT;

    // Hybrid/recurrent models require a full KV cache clear; partial trimming can
    // leave cache state inconsistent with n_past. Keep this aligned with LlamaLlm::reset().
    llama_memory_clear(llama_get_memory(this->ctx), /*clear data=*/true);

    this->n_past = 0;
    this->past_prompt.clear();
    this->past_gen.clear();

    return GENIEX_SUCCESS;
}

int32_t LlamaVlm::apply_chat_template(
    const geniex_VlmApplyChatTemplateInput* input, geniex_VlmApplyChatTemplateOutput* output) {
    if (output) {
        output->formatted_text        = nullptr;
        output->chat_template_grammar = {};
    }
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
    tmpl_inputs.parallel_tool_calls   = input->parallel_tool_calls;
    if (input->tools && strlen(input->tools) > 0) {
        tmpl_inputs.tools = common_chat_tools_parse_oaicompat(common_json::parse(std::string(input->tools)));
    }

    if (input->enable_thinking) {
        GENIEX_LOG_WARN("thinking mode not supported for llama.cpp VLM; ignoring enable_thinking=true");
    }
    tmpl_inputs.enable_thinking = false;
    GENIEX_LOG_DEBUG("applying chat template with add_generation_prompt=true, use_jinja={}", tmpl_inputs.use_jinja);

    // Apply chat template and retain its sampler grammar state.
    common_chat_templates_ptr tmpls = common_chat_templates_init(this->model, "");
    common_chat_session       session(tmpls.get(), llama_model_get_vocab(this->model), tmpl_inputs);
    common_params_sampling    sampling;
    session.apply_sampling(sampling);

    const std::string prompt_str = session.prompt().str();
    if (prompt_str.empty()) {
        GENIEX_LOG_ERROR("chat template application resulted in empty prompt");
        return GENIEX_ERROR_COMMON_FILE_NOT_FOUND;
    }

    if (!copy_chat_template_grammar(sampling, &output->chat_template_grammar)) {
        return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
    }

    // Allocate and copy result
    size_t prompt_length = prompt_str.length();
    char*  output_text   = (char*)malloc(prompt_length + 1);
    if (!output_text) {
        free_chat_template_grammar(&output->chat_template_grammar);
        return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
    }

    memcpy(output_text, prompt_str.c_str(), prompt_length);
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

    int32_t res = GENIEX_SUCCESS;

    geniex_GenerationConfig cfg = input->config ? *input->config : geniex_GenerationConfig{};
    if (cfg.max_tokens <= 0) cfg.max_tokens = 512;

    this->set_sampler(cfg.sampler_config);

    // Bitmap loading and tokenization fall outside both media_time and
    // prompt_time (only the chunk loop below is timed), so they show up in ttft.
    std::vector<mtmd_bitmap*> bitmaps;
    int                       n_media      = 0;
    auto                      free_bitmaps = [&]() {
        for (auto* bmp : bitmaps) mtmd_bitmap_free(bmp);
        bitmaps.clear();
    };

    if (input->config && input->config->image_paths && input->config->image_count > 0 && this->ctx_vision) {
        if (!this->supports_vision) {
            GENIEX_LOG_WARN("model does not support image input; skipping {} image(s)", input->config->image_count);
        } else {
            GENIEX_LOG_DEBUG("processing {} image(s)", input->config->image_count);
            for (int i = 0; i < input->config->image_count; ++i) {
                if (input->config->image_paths[i]) {
                    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(
                        this->ctx_vision, input->config->image_paths[i], false, mtmd_helper_init_opt_default())
                                           .bitmap;
                    if (bmp) {
                        bitmaps.push_back(bmp);
                        n_media++;
                        GENIEX_LOG_DEBUG("successfully loaded image {}: {}", i, input->config->image_paths[i]);
                    } else {
                        GENIEX_LOG_ERROR("failed to load image {}: {}", i, input->config->image_paths[i]);
                        free_bitmaps();
                        return GENIEX_ERROR_VLM_IMAGE_LOAD;
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
                    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(
                        this->ctx_vision, input->config->audio_paths[i], false, mtmd_helper_init_opt_default())
                                           .bitmap;
                    if (bmp) {
                        bitmaps.push_back(bmp);
                        n_media++;
                        GENIEX_LOG_DEBUG("successfully loaded audio {}: {}", i, input->config->audio_paths[i]);
                    } else {
                        GENIEX_LOG_ERROR("failed to load audio {}: {}", i, input->config->audio_paths[i]);
                        free_bitmaps();
                        return GENIEX_ERROR_VLM_AUDIO_LOAD;
                    }
                }
            }
        }
    }

    GENIEX_LOG_DEBUG("total media files loaded: {}", n_media);

    // Incremental text to feed (see vlm.h). Mismatch breaks append-only — fail.
    const std::string full_prompt(input->prompt_utf8);

    std::string new_text_portion;
    if (this->n_past == 0 || this->past_prompt.empty()) {
        new_text_portion = full_prompt;
    } else {
        size_t lcp     = 0;
        size_t lcp_max = std::min(full_prompt.size(), this->past_prompt.size());
        while (lcp < lcp_max && full_prompt[lcp] == this->past_prompt[lcp]) ++lcp;

        const size_t reuse_end = lcp + this->past_gen.size();
        const bool   gen_matches =
            reuse_end <= full_prompt.size() && full_prompt.compare(lcp, this->past_gen.size(), this->past_gen) == 0;

        if (!gen_matches) {
            GENIEX_LOG_ERROR("prefix reuse failed: prompt[{}:{}] does not match last generation (|G|={})",
                lcp,
                reuse_end,
                this->past_gen.size());
            return GENIEX_ERROR_VLM_PREFIX_REUSE_FAILED;
        }

        new_text_portion = full_prompt.substr(reuse_end);
        GENIEX_LOG_DEBUG(
            "prefix reuse: |A|={}, |G|={}, increment={} bytes", lcp, this->past_gen.size(), new_text_portion.size());
    }

    // Use mtmd path when ctx_vision is available, fallback to direct llama path otherwise
    if (this->ctx_vision) {
        GENIEX_LOG_DEBUG("using multimodal (mtmd) path with ctx_vision");

        // Only process if there's new text content
        if (!new_text_portion.empty()) {
            // prompt_utf8 already has chat template and media markers applied
            mtmd_input_text text;
            text.text          = new_text_portion.c_str();
            text.text_len      = new_text_portion.length();
            text.add_special   = this->n_past == 0;  // add BOS only on first message
            text.parse_special = true;

            mtmd_input_chunks* chunks = mtmd_input_chunks_init();
            if (!chunks) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;

            int32_t tok_ret =
                mtmd_tokenize(this->ctx_vision, chunks, &text, (const mtmd_bitmap**)bitmaps.data(), n_media);
            for (auto bmp : bitmaps) {
                if (bmp) mtmd_bitmap_free(bmp);
            }
            if (tok_ret != 0) {
                mtmd_input_chunks_free(chunks);
                GENIEX_LOG_ERROR("mtmd_tokenize failed");
                return GENIEX_ERROR_VLM_GENERATION_FAILED;
            }

            // Time each phase separately: encoder → media_time, decode/prefill →
            // prompt_time. prompt_tokens counts text + media tokens.
            const size_t  n_chunks      = mtmd_input_chunks_size(chunks);
            const int32_t n_batch       = llama_n_batch(this->ctx);
            llama_pos     n_past_cur    = this->n_past;
            uint32_t      prompt_tokens = 0;
            for (size_t i = 0; i < n_chunks && res == GENIEX_SUCCESS; ++i) {
                const mtmd_input_chunk* chunk    = mtmd_input_chunks_get(chunks, i);
                const bool              is_media = mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_TEXT;
                const bool              is_last  = (i == n_chunks - 1);
                // Token count, not KV positions (differ under M-RoPE; KV uses new_n_past).
                const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);

                llama_pos new_n_past = n_past_cur;
                int32_t   ret;
                if (is_media) {
                    profiler.media_start();
                    ret = mtmd_encode_chunk(this->ctx_vision, chunk);
                    profiler.media_end();
                    if (ret != 0) {
                        // encode returns 1 on a generic error, not KV-full: fail, don't truncate.
                        GENIEX_LOG_ERROR("media chunk encoding failed");
                        res = GENIEX_ERROR_VLM_GENERATION_FAILED;
                        break;
                    }
                    float* embd = mtmd_get_output_embd(this->ctx_vision);
                    profiler.prompt_start();
                    ret = mtmd_helper_decode_image_chunk(this->ctx_vision,
                        this->ctx,
                        chunk,
                        embd,
                        n_past_cur,
                        0,
                        n_batch,
                        &new_n_past,
                        nullptr,
                        nullptr);
                    profiler.prompt_end();
                } else {
                    profiler.prompt_start();
                    ret = mtmd_helper_eval_chunk_single(
                        this->ctx_vision, this->ctx, chunk, n_past_cur, 0, n_batch, is_last, &new_n_past);
                    profiler.prompt_end();
                }

                // This is prefill: a return of 1 (KV cache full) means the prompt
                // itself is too long, not a generic failure.
                switch (ret) {
                    case 0:
                        prompt_tokens += (uint32_t)n_tokens;
                        n_past_cur = new_n_past;
                        break;
                    case 1:
                        res = GENIEX_ERROR_LLM_GENERATION_PROMPT_TOO_LONG;
                        break;
                    default:
                        GENIEX_LOG_ERROR("chunk evaluation failed");
                        res = GENIEX_ERROR_VLM_GENERATION_FAILED;
                        break;
                }
            }
            if (res == GENIEX_SUCCESS) {
                profiler.update_prompt_tokens(prompt_tokens);
                this->n_past = n_past_cur;
            }
            mtmd_input_chunks_free(chunks);
        } else {
            // No new content to process, just clear bitmaps
            for (auto bmp : bitmaps) mtmd_bitmap_free(bmp);
            GENIEX_LOG_DEBUG("no new text content, skipping mtmd processing");
        }

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
            int32_t* prompt_tokens = (int32_t*)calloc(needed, sizeof(int32_t));
            if (!prompt_tokens) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;

            int32_t prompt_len = llama_tokenize(
                vocab, new_text_portion.c_str(), (int32_t)new_text_portion.length(), prompt_tokens, needed, true, true);
            if (prompt_len < 0) {
                free(prompt_tokens);
                return GENIEX_ERROR_LLM_TOKENIZATION_FAILED;
            }

            GENIEX_LOG_DEBUG("_ml_vlm_generate_internal: Tokenized new text portion into {} tokens", prompt_len);

            profiler.update_prompt_tokens(prompt_len);

            llama_batch batch = llama_batch_get_one(prompt_tokens, prompt_len);
            // Set positions based on current n_past
            for (int i = 0; i < batch.n_tokens; ++i) {
                batch.pos[i] = this->n_past + i;
            }

            profiler.prompt_start();
            int32_t decode_ret = llama_decode(this->ctx, batch);
            profiler.prompt_end();
            profiler.decode_start();
            free(prompt_tokens);
            // 1 means the prompt does not fit the KV cache: since this is
            // prefill, the prompt itself is too long, not a generic failure.
            switch (decode_ret) {
                case 0:
                    this->n_past += prompt_len;
                    break;
                case 1:
                    res = GENIEX_ERROR_LLM_GENERATION_PROMPT_TOO_LONG;
                    break;
                default:
                    GENIEX_LOG_ERROR("llama_decode failed");
                    res = GENIEX_ERROR_VLM_GENERATION_FAILED;
                    break;
            }
        } else {
            GENIEX_LOG_DEBUG("no new text content, skipping direct llama processing");
        }
    }

    // Generate tokens (common for both multimodal and text-only)
    GENIEX_LOG_DEBUG("starting token generation loop");
    const llama_vocab* vocab = llama_model_get_vocab(this->model);
    char               token_buffer[256];

    // Create reusable batch like mtmd-cli.cpp
    common_batch batch(this->ctx);

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

        int n = llama_token_to_piece(vocab, token, token_buffer, sizeof(token_buffer) - 1, 0, /*special=*/true);
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
        batch.clear();
        batch.add(token, this->n_past++, /*seq_id=*/0, /*output=*/true);
        switch (llama_process(this->ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
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

    auto full_text_str = full_text.str();
    output->full_text  = strdup(full_text_str.c_str());
    if (generated_token_count > 0) {
        // Record this turn so the next can reuse A+T+G (see vlm.h).
        this->past_prompt = full_prompt;
        this->past_gen    = full_text_str;
    }

    GENIEX_LOG_DEBUG("completed generation with {} tokens", generated_token_count);
    return res;
}

}  // namespace geniex

namespace geniex {

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
    apply_tool_fields(*output, input->tool_calls, input->tool_call_count, input->tool_call_id, input->tool_name);

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
