// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "geniex.h"

namespace geniex {

class IVlm {
   public:
    virtual ~IVlm() = default;

    /**
     * @brief Create and initialize the VLM model
     * @param input The creation input parameters
     * @return ML error code (GENIEX_SUCCESS on success, negative on failure)
     */
    virtual int32_t create(const geniex_VlmCreateInput* input) = 0;

    virtual int32_t reset() = 0;

    virtual int32_t apply_chat_template(
        const geniex_VlmApplyChatTemplateInput* input, geniex_VlmApplyChatTemplateOutput* output) = 0;

    virtual int32_t generate(const geniex_VlmGenerateInput* input, geniex_VlmGenerateOutput* output) = 0;

    virtual int32_t get_capabilities(geniex_VlmCapabilities* output) {
        if (!output) return GENIEX_ERROR_COMMON_INVALID_INPUT;
        output->supports_vision = false;
        output->supports_audio  = false;
        return GENIEX_SUCCESS;
    }
};

}  // namespace geniex
