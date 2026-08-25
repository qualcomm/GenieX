// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "geniex.h"
#include "logging.h"
#include "plugin/ILlm.h"
#include "profile.h"
#include "registry.h"
#include "utils.h"

using namespace geniex;

int32_t geniex_llm_create(const geniex_LlmCreateInput* input, geniex_LLM** out_handle) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = geniex::Registry::instance().get<geniex::ILlm>(input->plugin_id);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_SUPPORTED;
        int32_t res = backend->create(input);
        if (res != GENIEX_SUCCESS) {
            delete backend;
        } else {
            *out_handle = reinterpret_cast<geniex_LLM*>(backend);
        }
        return res;
    } catch (const PluginNotFoundException& e) {
        GENIEX_LOG_ERROR("plugin not found");
        return GENIEX_ERROR_COMMON_PLUGIN_INVALID;
    } catch (const PluginLoadException& e) {
        GENIEX_LOG_ERROR("plugin load error");
        return GENIEX_ERROR_COMMON_PLUGIN_LOAD;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("creating llm error: {}", e.what());
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
}

int32_t geniex_llm_destroy(geniex_LLM* h) {
    GENIEX_LOG_TRACE("llm destroy");

    try {
        auto backend = reinterpret_cast<ILlm*>(h);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        delete backend;
        return GENIEX_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("destroy llm error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_llm_reset(geniex_LLM* h) {
    GENIEX_LOG_TRACE("llm reset");

    try {
        auto backend = reinterpret_cast<ILlm*>(h);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        return backend->reset();
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("reset llm error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_llm_save_kv_cache(
    geniex_LLM* h, const geniex_KvCacheSaveInput* input, geniex_KvCacheSaveOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<ILlm*>(h);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        auto result = backend->save_kv_cache(input, output);
        GENIEX_LOG_TRACE("{}: {}", static_cast<geniex_ErrorCode>(result), output);
        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("llm save kv cache error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_llm_load_kv_cache(
    geniex_LLM* h, const geniex_KvCacheLoadInput* input, geniex_KvCacheLoadOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<ILlm*>(h);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        auto result = backend->load_kv_cache(input, output);
        GENIEX_LOG_TRACE("{}: {}", static_cast<geniex_ErrorCode>(result), output);
        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("llm load kv cache error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_llm_apply_chat_template(
    geniex_LLM* h, const geniex_LlmApplyChatTemplateInput* input, geniex_LlmApplyChatTemplateOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<ILlm*>(h);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        auto result = backend->apply_chat_template(input, output);
        GENIEX_LOG_TRACE("{}: {}", static_cast<geniex_ErrorCode>(result), output);
        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("llm apply chat template error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_llm_generate(geniex_LLM* h, const geniex_LlmGenerateInput* input, geniex_LlmGenerateOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<ILlm*>(h);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;

        // Wrap the user's callback with UTF-8 validation if a callback is provided
        std::unique_ptr<Utf8CallbackWrapper> wrapper;
        geniex_LlmGenerateInput              modified_input;

        if (input->on_token) {
            // Create wrapper that accumulates incomplete UTF-8 sequences
            wrapper                     = std::make_unique<Utf8CallbackWrapper>();
            wrapper->original_callback  = input->on_token;
            wrapper->original_user_data = input->user_data;

            // Create a modified input with our wrapped callback
            modified_input           = *input;
            modified_input.on_token  = get_utf8_callback_wrapper();
            modified_input.user_data = wrapper.get();

            int32_t result = backend->generate(&modified_input, output);
            calculate_profile_data(output->profile_data);
            GENIEX_LOG_TRACE("{}: {}", static_cast<geniex_ErrorCode>(result), output);

            // Flush any remaining incomplete UTF-8 sequence after generation completes
            wrapper->flush();

            return result;
        } else {
            // No callback, pass through directly
            int32_t result = backend->generate(input, output);
            calculate_profile_data(output->profile_data);
            GENIEX_LOG_TRACE("{}: {}", static_cast<geniex_ErrorCode>(result), output);
            return result;
        }
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("llm generate error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_llm_get_model_info(geniex_LLM* h, geniex_LlmModelInfo* output) {
    GENIEX_LOG_TRACE("llm get_model_info");

    if (!output) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    std::memset(output, 0, sizeof(*output));

    try {
        auto backend = reinterpret_cast<ILlm*>(h);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        return backend->get_model_info(output);
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("llm get_model_info error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

// Reduce full-vocab row-major logits to the top-N per row (descending logit),
// writing new logits/token_ids buffers. Selection mirrors the O(vocab * top_n)
// partial scan the benchmark used before this moved into the SDK. Frees the
// plugin's full-vocab buffer and repoints output at the reduced buffers.
static int32_t reduce_forward_logits_top_n(geniex_LlmForwardLogitsOutput* output, int32_t top_n) {
    const int32_t n_rows     = output->n_rows;
    const int32_t vocab_size = output->vocab_size;
    const int32_t row_width  = top_n < vocab_size ? top_n : vocab_size;

    float*            red_logits = static_cast<float*>(std::malloc((size_t)n_rows * row_width * sizeof(float)));
    int32_t*          red_ids    = static_cast<int32_t*>(std::malloc((size_t)n_rows * row_width * sizeof(int32_t)));
    std::vector<char> taken(vocab_size);
    if (!red_logits || !red_ids) {
        std::free(red_logits);
        std::free(red_ids);
        return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
    }

    for (int32_t r = 0; r < n_rows; ++r) {
        const float* row = output->logits + (size_t)r * vocab_size;
        std::fill(taken.begin(), taken.end(), 0);
        for (int32_t k = 0; k < row_width; ++k) {
            int32_t best = -1;
            for (int32_t v = 0; v < vocab_size; ++v) {
                if (taken[v]) continue;
                if (best < 0 || row[v] > row[best]) best = v;
            }
            taken[best]                           = 1;
            red_logits[(size_t)r * row_width + k] = row[best];
            red_ids[(size_t)r * row_width + k]    = best;
        }
    }

    std::free(output->logits);
    output->logits    = red_logits;
    output->token_ids = red_ids;
    output->row_width = row_width;
    return GENIEX_SUCCESS;
}

int32_t geniex_llm_forward_logits(
    geniex_LLM* h, const geniex_LlmForwardLogitsInput* input, geniex_LlmForwardLogitsOutput* output) {
    GENIEX_LOG_TRACE("llm forward_logits");

    if (!input || !output) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    if (!input->input_ids || input->input_ids_count <= 0) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    if (input->top_n < 0) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    std::memset(output, 0, sizeof(*output));

    // No UTF-8 callback wrapping here: this path returns raw floats, not decoded text.
    try {
        auto backend = reinterpret_cast<ILlm*>(h);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;

        // Plugins always produce full-vocab logits; the bridge does the top-N
        // reduction so both backends share one implementation.
        int32_t rc = backend->forward_logits(input, output);
        if (rc != GENIEX_SUCCESS) return rc;

        output->row_width = output->vocab_size;
        if (input->top_n > 0 && output->vocab_size > 0) {
            rc = reduce_forward_logits_top_n(output, input->top_n);
            if (rc != GENIEX_SUCCESS) {
                std::free(output->logits);
                std::memset(output, 0, sizeof(*output));
            }
        }
        return rc;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("llm forward_logits error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}
