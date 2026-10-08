// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "vlm.h"

#include <algorithm>
#include <cstring>
#include <memory>

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

static std::string media_marker(size_t index) { return "<__geniex_media_" + std::to_string(index) + "__>"; }

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

    if (!selection->empty()) {
        mpar.devices = selection->data();
    }

    ggml_backend_dev_t vision_device = nullptr;
    if (input->vit_device_id && input->vit_device_id[0] != '\0') {
        vision_device = ggml_backend_dev_by_name(input->vit_device_id);
        if (!vision_device) {
            GENIEX_LOG_ERROR("Vision device '{}' not found", input->vit_device_id);
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
        GENIEX_LOG_INFO("Using vision device override: {}", input->vit_device_id);
    } else if (!selection->empty()) {
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
    llama_context_params cpar          = build_context_params(config, n_ctx_default, device);
    this->ctx                          = llama_init_from_model(this->model, cpar);
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
        // Zack TODO: elegant fix this error:  no member named 'verbosity' in 'mtmd_context_params'
        // mparams.verbosity           = GGML_LOG_LEVEL_ERROR;

        this->ctx_vision = mtmd_init_from_file(input->mmproj_path, this->model, mparams);
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

int32_t LlamaVlm::reset() {
    if (!this->ctx) return GENIEX_ERROR_COMMON_INVALID_INPUT;

    // Hybrid/recurrent models require a full KV cache clear; partial trimming can
    // leave cache state inconsistent with n_past. Keep this aligned with LlamaLlm::reset().
    llama_memory_clear(llama_get_memory(this->ctx), /*clear data=*/true);

    this->n_past = 0;
    this->cached_chunks.clear();
    this->cached_media.clear();
    this->pending = {};

    return GENIEX_SUCCESS;
}

int32_t LlamaVlm::apply_chat_template(
    const geniex_VlmApplyChatTemplateInput* input, geniex_VlmApplyChatTemplateOutput* output) {
    this->pending = {};
    if (!this->model || !input || !output || !input->messages || input->message_count <= 0)
        return GENIEX_ERROR_COMMON_INVALID_INPUT;

    // Convert geniex_VlmChatMessage array to vector<common_chat_msg>
    std::vector<common_chat_msg> chat_messages;
    chat_messages.reserve(input->message_count);
    std::vector<Media> media;

    for (int32_t i = 0; i < input->message_count; ++i) {
        if (input->messages[i].content_count > 0 && !input->messages[i].contents) {
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
        common_chat_msg msg;
        if (!this->vlm_message_to_common_chat_msg(&input->messages[i], &msg, media.size())) {
            GENIEX_LOG_DEBUG("failed to convert message {} (role={})",
                i,
                input->messages[i].role ? input->messages[i].role : "NULL");
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
        chat_messages.push_back(msg);
        for (int64_t j = 0; j < input->messages[i].content_count; ++j) {
            const auto& part = input->messages[i].contents[j];
            if (strcmp(part.type, "image") == 0 || strcmp(part.type, "audio") == 0) {
                Media item;
                item.kind = strcmp(part.type, "image") == 0 ? VlmPrefixChunk::Kind::Image : VlmPrefixChunk::Kind::Audio;
                item.id   = part.media_id && part.media_id[0] ? part.media_id : (part.text ? part.text : "");
                item.path = part.text ? part.text : "";
                item.explicit_id = part.media_id && part.media_id[0];
                media.push_back(std::move(item));
            }
        }
        GENIEX_LOG_DEBUG(
            "converted message {} - role={}, content_length={}", i, msg.role.c_str(), msg.content.length());
    }

    common_chat_templates_inputs tmpl_inputs;
    tmpl_inputs.messages              = chat_messages;
    tmpl_inputs.add_generation_prompt = true;
    tmpl_inputs.use_jinja             = true;
    if (input->tools && strlen(input->tools) > 0) {
        tmpl_inputs.tools = common_chat_tools_parse_oaicompat(common_json::parse(std::string(input->tools)));
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

    std::vector<std::pair<size_t, size_t>> markers;
    for (size_t i = 0; i < media.size(); ++i) {
        const auto marker = media_marker(i);
        for (size_t at = 0; (at = result.prompt.find(marker, at)) != std::string::npos; at += marker.size()) {
            markers.emplace_back(at, i);
        }
    }
    std::sort(markers.begin(), markers.end());
    std::vector<Media> rendered_media;
    for (const auto& entry : markers) rendered_media.push_back(media[entry.second]);
    for (auto it = markers.rbegin(); it != markers.rend(); ++it) {
        result.prompt.replace(it->first, media_marker(it->second).size(), mtmd_default_marker());
    }

    // Allocate and copy result
    size_t prompt_length = result.prompt.length();
    char*  output_text   = (char*)malloc(prompt_length + 1);
    if (!output_text) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;

    memcpy(output_text, result.prompt.c_str(), prompt_length);
    output_text[prompt_length] = '\0';

    output->formatted_text = output_text;
    this->pending          = {result.prompt, std::move(rendered_media)};

    GENIEX_LOG_DEBUG("successfully generated prompt with length={}", prompt_length);
    GENIEX_LOG_DEBUG("result text: {}", output_text);

    return GENIEX_SUCCESS;
}

int32_t LlamaVlm::generate(const geniex_VlmGenerateInput* input, geniex_VlmGenerateOutput* output) {
    struct ResetOnFailure {
        LlamaVlm& vlm;
        bool      committed = false;
        ~ResetOnFailure() {
            if (!committed) vlm.reset();
        }
    } transaction{*this};
    if (!this->ctx || !input || !output || !input->prompt_utf8) {
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    common::Profiler profiler;

    int32_t res = GENIEX_SUCCESS;

    geniex_GenerationConfig cfg = input->config ? *input->config : geniex_GenerationConfig{};
    if (cfg.max_tokens <= 0) cfg.max_tokens = 512;

    this->set_sampler(cfg.sampler_config);

    const std::string  full_prompt(input->prompt_utf8);
    std::vector<Media> media;
    if (this->pending.prompt == full_prompt) {
        media = std::move(this->pending.media);
    } else if (input->config) {
        for (int i = 0; i < cfg.image_count; ++i) {
            if (cfg.image_paths && cfg.image_paths[i]) {
                media.push_back({VlmPrefixChunk::Kind::Image, cfg.image_paths[i], cfg.image_paths[i]});
            }
        }
        for (int i = 0; i < cfg.audio_count; ++i) {
            if (cfg.audio_paths && cfg.audio_paths[i]) {
                media.push_back({VlmPrefixChunk::Kind::Audio, cfg.audio_paths[i], cfg.audio_paths[i]});
            }
        }
    }
    this->pending = {};

    const std::string marker       = mtmd_default_marker();
    size_t            marker_count = 0;
    for (size_t at = 0; (at = full_prompt.find(marker, at)) != std::string::npos; at += marker.size()) {
        ++marker_count;
    }
    if (marker_count != media.size() || (marker_count && !this->ctx_vision)) {
        GENIEX_LOG_ERROR("VLM prompt media markers do not match the available media identities");
        return GENIEX_ERROR_VLM_PREFIX_REUSE_FAILED;
    }

    const size_t image_count = std::count_if(
        media.begin(), media.end(), [](const Media& item) { return item.kind == VlmPrefixChunk::Kind::Image; });
    size_t next_image = 0;
    size_t next_audio = 0;
    for (auto& item : media) {
        const bool         image      = item.kind == VlmPrefixChunk::Kind::Image;
        size_t&            next       = image ? next_image : next_audio;
        const char* const* paths      = image ? cfg.image_paths : cfg.audio_paths;
        const int          count      = image ? cfg.image_count : cfg.audio_count;
        const size_t       total      = image ? image_count : media.size() - image_count;
        const bool         full_paths = count > 0 && static_cast<size_t>(count) == total;
        if (full_paths && paths && paths[next]) {
            item.path = paths[next++];
            if (!item.explicit_id) item.id = item.path;
        }
        auto old = std::find_if(this->cached_media.begin(), this->cached_media.end(), [&](const Media& m) {
            return m.kind == item.kind && !item.id.empty() && m.id == item.id;
        });
        if (old != this->cached_media.end()) {
            if (item.path.empty()) item.path = old->path;
            item.width         = old->width;
            item.height        = old->height;
            item.audio_samples = old->audio_samples;
        } else {
            if (!full_paths && paths && count > 0 && next < static_cast<size_t>(count) && paths[next]) {
                item.path = paths[next++];
                if (!item.explicit_id) item.id = item.path;
            }
            if (item.path.empty()) item.path = item.id;
        }
        if (item.id.empty()) item.id = item.path;
        if (item.id.empty()) {
            GENIEX_LOG_ERROR("VLM media identity is missing");
            return GENIEX_ERROR_VLM_PREFIX_REUSE_FAILED;
        }
    }

    using Bitmap  = std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>;
    using MediaId = std::pair<VlmPrefixChunk::Kind, std::string>;
    std::vector<Bitmap>             bitmaps;
    std::vector<const mtmd_bitmap*> bitmap_ptrs;
    auto                            make_bitmaps = [&](bool all_real, const std::vector<MediaId>& real_ids) -> int32_t {
        for (size_t i = 0; i < media.size(); ++i) {
            auto&      item = media[i];
            const bool old  = item.width || item.audio_samples;
            const bool real =
                all_real || !old ||
                std::find(real_ids.begin(), real_ids.end(), MediaId{item.kind, item.id}) != real_ids.end();
            if (i < bitmaps.size() && (!real || mtmd_bitmap_get_data(bitmaps[i].get()))) continue;
            mtmd_bitmap* raw = nullptr;
            if (real) {
                if (!item.path.empty()) {
                    raw = mtmd_helper_bitmap_init_from_file(
                        this->ctx_vision, item.path.c_str(), false, mtmd_helper_init_opt_default())
                              .bitmap;
                }
                if (!raw) {
                    GENIEX_LOG_ERROR("VLM media required for prefill: {}", item.id);
                    return GENIEX_ERROR_VLM_PREFIX_REUSE_FAILED;
                }
                if (mtmd_bitmap_is_audio(raw) != (item.kind == VlmPrefixChunk::Kind::Audio)) {
                    mtmd_bitmap_free(raw);
                    return GENIEX_ERROR_COMMON_INVALID_INPUT;
                }
                item.width  = mtmd_bitmap_get_nx(raw);
                item.height = mtmd_bitmap_get_ny(raw);
                item.audio_samples =
                    item.kind == VlmPrefixChunk::Kind::Audio ? mtmd_bitmap_get_n_bytes(raw) / sizeof(float) : 0;
            } else if (item.kind == VlmPrefixChunk::Kind::Audio) {
                raw = mtmd_bitmap_init_from_audio(item.audio_samples, nullptr);
            } else {
                raw = mtmd_bitmap_init(item.width, item.height, nullptr);
            }
            if (!raw) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
            mtmd_bitmap_set_id(raw, item.id.c_str());
            if (i < bitmaps.size()) {
                bitmaps[i].reset(raw);
                bitmap_ptrs[i] = raw;
            } else {
                bitmaps.emplace_back(raw, mtmd_bitmap_free);
                bitmap_ptrs.push_back(raw);
            }
        }
        return GENIEX_SUCCESS;
    };

    auto describe = [](const mtmd_input_chunks* chunks) {
        std::vector<VlmPrefixChunk> result;
        result.reserve(mtmd_input_chunks_size(chunks));
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            const auto*    chunk = mtmd_input_chunks_get(chunks, i);
            VlmPrefixChunk part;
            const auto     type = mtmd_input_chunk_get_type(chunk);
            if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                part.kind          = VlmPrefixChunk::Kind::Text;
                size_t      count  = 0;
                const auto* tokens = mtmd_input_chunk_get_tokens_text(chunk, &count);
                part.tokens.assign(tokens, tokens + count);
            } else {
                part.kind =
                    type == MTMD_INPUT_CHUNK_TYPE_AUDIO ? VlmPrefixChunk::Kind::Audio : VlmPrefixChunk::Kind::Image;
                const char* id = mtmd_input_chunk_get_id(chunk);
                if (id) part.media_id = id;
            }
            result.push_back(std::move(part));
        }
        return result;
    };

    using Chunks  = std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>;
    auto tokenize = [&](Chunks& chunks, std::vector<VlmPrefixChunk>& parts) -> int32_t {
        chunks.reset(mtmd_input_chunks_init());
        if (!chunks) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
        mtmd_input_text text{full_prompt.c_str(), full_prompt.size(), true, true};
        if (mtmd_tokenize(this->ctx_vision, chunks.get(), &text, bitmap_ptrs.data(), bitmap_ptrs.size())) {
            GENIEX_LOG_ERROR("VLM prompt tokenization failed");
            return GENIEX_ERROR_VLM_GENERATION_FAILED;
        }
        parts = describe(chunks.get());
        return GENIEX_SUCCESS;
    };

    Chunks                      chunks(nullptr, mtmd_input_chunks_free);
    std::vector<VlmPrefixChunk> parts;
    if (this->ctx_vision) {
        res = make_bitmaps(false, {});
        if (res != GENIEX_SUCCESS) return res;
        res = tokenize(chunks, parts);
        if (res != GENIEX_SUCCESS) return res;
    } else {
        const llama_vocab* vocab = llama_model_get_vocab(this->model);
        int                needed =
            llama_tokenize(vocab, full_prompt.c_str(), static_cast<int>(full_prompt.size()), nullptr, 0, true, true);
        if (needed < 0) needed = -needed;
        if (needed <= 0) return GENIEX_ERROR_LLM_TOKENIZATION_FAILED;
        VlmPrefixChunk part;
        part.kind = VlmPrefixChunk::Kind::Text;
        part.tokens.resize(needed);
        int count = llama_tokenize(
            vocab, full_prompt.c_str(), static_cast<int>(full_prompt.size()), part.tokens.data(), needed, true, true);
        if (count < 0) return GENIEX_ERROR_LLM_TOKENIZATION_FAILED;
        part.tokens.resize(count);
        parts.push_back(std::move(part));
    }
    if (parts.empty()) return GENIEX_ERROR_LLM_TOKENIZATION_FAILED;

    auto match = vlm_prefill_boundary(this->cached_chunks, parts);

    if (this->ctx_vision) {
        std::vector<MediaId> required;
        for (size_t i = match.next_chunk; i < parts.size(); ++i) {
            if (parts[i].kind != VlmPrefixChunk::Kind::Text) required.emplace_back(parts[i].kind, parts[i].media_id);
        }
        if (match.kv_pos == 0 || !required.empty()) {
            res = make_bitmaps(match.kv_pos == 0, required);
            if (res != GENIEX_SUCCESS) return res;
            res = tokenize(chunks, parts);
            if (res != GENIEX_SUCCESS) return res;
            const auto verified = vlm_prefill_boundary(this->cached_chunks, parts);
            if (verified.next_chunk != match.next_chunk || verified.text_offset != match.text_offset ||
                verified.kv_pos != match.kv_pos) {
                // Loading real media can change the placeholder's chunk layout.
                match = {};
                res   = make_bitmaps(true, {});
                if (res != GENIEX_SUCCESS) return res;
                res = tokenize(chunks, parts);
                if (res != GENIEX_SUCCESS) return res;
            }
        }
    }

    llama_memory_t memory = llama_get_memory(this->ctx);
    if (match.kv_pos == 0) {
        llama_memory_clear(memory, true);
    } else if (!llama_memory_seq_rm(memory, 0, match.kv_pos, -1)) {
        if (this->ctx_vision) {
            res = make_bitmaps(true, {});
            if (res != GENIEX_SUCCESS) return res;
            res = tokenize(chunks, parts);
            if (res != GENIEX_SUCCESS) return res;
        }
        llama_memory_clear(memory, true);
        match = {};
    }

    std::vector<VlmPrefixChunk> evaluated;
    evaluated.reserve(parts.size());
    for (size_t i = 0; i < match.next_chunk; ++i) evaluated.push_back(this->cached_chunks[i]);
    int32_t       position      = match.kv_pos;
    uint32_t      prompt_tokens = 0;
    const int32_t n_batch       = llama_n_batch(this->ctx);
    for (size_t i = match.next_chunk; i < parts.size() && res == GENIEX_SUCCESS; ++i) {
        auto&         part   = parts[i];
        const bool    last   = i + 1 == parts.size();
        const size_t  offset = i == match.next_chunk ? match.text_offset : 0;
        const int32_t begin  = offset ? this->cached_chunks[i].kv_begin : position;
        if (part.kind == VlmPrefixChunk::Kind::Text) {
            llama_batch batch = llama_batch_init(n_batch, 0, 1);
            for (size_t t = offset; t < part.tokens.size() && res == GENIEX_SUCCESS;) {
                batch.n_tokens = 0;
                while (t < part.tokens.size() && batch.n_tokens < n_batch) {
                    const int j        = batch.n_tokens++;
                    batch.token[j]     = part.tokens[t++];
                    batch.pos[j]       = position + j;
                    batch.n_seq_id[j]  = 1;
                    batch.seq_id[j][0] = 0;
                    batch.logits[j]    = false;
                }
                if (last && t == part.tokens.size()) batch.logits[batch.n_tokens - 1] = true;
                profiler.prompt_start();
                const int ret = llama_decode(this->ctx, batch);
                profiler.prompt_end();
                if (ret == 0) {
                    position += batch.n_tokens;
                    prompt_tokens += batch.n_tokens;
                } else {
                    res = ret == 1 ? GENIEX_ERROR_LLM_GENERATION_PROMPT_TOO_LONG : GENIEX_ERROR_VLM_GENERATION_FAILED;
                }
            }
            llama_batch_free(batch);
        } else {
            const auto* chunk = mtmd_input_chunks_get(chunks.get(), i);
            profiler.media_start();
            const int encode_ret = mtmd_encode_chunk(this->ctx_vision, chunk);
            profiler.media_end();
            if (encode_ret != 0) {
                res = GENIEX_ERROR_VLM_GENERATION_FAILED;
            } else {
                llama_pos new_position = position;
                profiler.prompt_start();
                const int ret = mtmd_helper_decode_image_chunk(this->ctx_vision,
                    this->ctx,
                    chunk,
                    mtmd_get_output_embd(this->ctx_vision),
                    position,
                    0,
                    n_batch,
                    &new_position,
                    nullptr,
                    nullptr);
                profiler.prompt_end();
                if (ret == 0) {
                    position = new_position;
                    prompt_tokens += static_cast<uint32_t>(mtmd_input_chunk_get_n_tokens(chunk));
                } else {
                    res = ret == 1 ? GENIEX_ERROR_LLM_GENERATION_PROMPT_TOO_LONG : GENIEX_ERROR_VLM_GENERATION_FAILED;
                }
            }
        }
        part.kv_begin = begin;
        part.kv_end   = position;
        if (res == GENIEX_SUCCESS) evaluated.push_back(part);
    }
    if (res != GENIEX_SUCCESS) {
        return res;
    }
    this->n_past = position;
    profiler.update_prompt_tokens(prompt_tokens);
    profiler.decode_start();

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
        common_batch_clear(batch);
        common_batch_add(batch, token, this->n_past, {0}, true);
        switch (llama_decode(this->ctx, batch)) {
            case 0:
                if (evaluated.back().kind != VlmPrefixChunk::Kind::Text) {
                    evaluated.push_back({VlmPrefixChunk::Kind::Text, {}, "", this->n_past, this->n_past});
                }
                evaluated.back().tokens.push_back(token);
                evaluated.back().kv_end = this->n_past + 1;
                ++this->n_past;
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

    auto full_text_str = full_text.str();
    output->full_text  = strdup(full_text_str.c_str());
    if (!output->full_text) return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
    if (res == GENIEX_SUCCESS) {
        this->cached_chunks   = std::move(evaluated);
        this->cached_media    = std::move(media);
        transaction.committed = true;
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

bool LlamaVlm::vlm_message_to_common_chat_msg(
    const geniex_VlmChatMessage* input, common_chat_msg* output, size_t media_offset) {
    if (!input || !output) return false;

    // Role is required
    if (!input->role || strlen(input->role) == 0) {
        return false;
    }

    output->role = input->role;
    apply_tool_fields(*output, input->tool_calls, input->tool_call_count, input->tool_call_id, input->tool_name);

    if (input->contents && input->content_count > 0) {
        std::string final_content;
        for (int64_t j = 0; j < input->content_count; ++j) {
            if (!input->contents[j].type || strlen(input->contents[j].type) == 0) {
                return false;
            }
            if (strcmp(input->contents[j].type, "text") == 0) {
                if (input->contents[j].text) {
                    if (strstr(input->contents[j].text, "<__geniex_media_")) return false;
                    final_content += input->contents[j].text;
                }
            } else if (strcmp(input->contents[j].type, "image") == 0 || strcmp(input->contents[j].type, "audio") == 0) {
                final_content += media_marker(media_offset++);
            } else {
                return false;
            }
        }
        output->content = final_content;
    }

    return true;
}

}  // namespace geniex
