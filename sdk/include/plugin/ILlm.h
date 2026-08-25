// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "geniex.h"

namespace geniex {

class ILlm {
   public:
    virtual ~ILlm() = default;

    /**
     * @brief Create and initialize the LLM model
     * @param input The creation input parameters
     * @return ML error code (GENIEX_SUCCESS on success, negative on failure)
     */
    virtual int32_t create(const geniex_LlmCreateInput* input) = 0;

    virtual int32_t reset() = 0;

    virtual int32_t save_kv_cache(const geniex_KvCacheSaveInput*, geniex_KvCacheSaveOutput*) = 0;
    virtual int32_t load_kv_cache(const geniex_KvCacheLoadInput*, geniex_KvCacheLoadOutput*) = 0;

    virtual int32_t apply_chat_template(
        const geniex_LlmApplyChatTemplateInput*, geniex_LlmApplyChatTemplateOutput*) = 0;

    virtual int32_t generate(const geniex_LlmGenerateInput*, geniex_LlmGenerateOutput*) = 0;

    /**
     * @brief Report static model metadata (vocab size, BOS handling).
     *
     * Default returns PARAM_NOT_SUPPORTED so plugins that cannot expose this
     * information keep building. Plugins able to report it MUST override.
     */
    virtual int32_t get_model_info(geniex_LlmModelInfo*) { return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED; }

    /**
     * @brief Run a single forward pass and return raw logits (no sampling/decode).
     *
     * Default returns PARAM_NOT_SUPPORTED so plugins that cannot expose logits
     * keep building. Plugins able to produce them MUST override.
     */
    virtual int32_t forward_logits(const geniex_LlmForwardLogitsInput*, geniex_LlmForwardLogitsOutput*) {
        return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }
};

}  // namespace geniex
