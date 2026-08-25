// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>

#include "chat.h"
#include "htp_session.h"
#include "mtmd.h"
#include "plugin/IVlm.h"
#include "sampling.h"
#include "threadpool.h"

// Forward declarations for llama.cpp types
struct llama_context;
struct llama_model;
struct llama_sampling_context;

namespace geniex {

class LlamaVlm : public IVlm {
    llama_model*    model      = nullptr;
    llama_context*  ctx        = nullptr;
    common_sampler* sampler    = nullptr;
    mtmd_context*   ctx_vision = nullptr;
    Threadpools     pools_;

    // mmproj-reported modality support; both false when no mmproj is loaded.
    bool supports_vision = false;
    bool supports_audio  = false;

    // Append-only KV reuse: skip the common prefix + last generation, feed only
    // the rest. Recurrent models can't roll back, so we never re-feed old tokens.
    int32_t     n_past = 0;
    std::string past_prompt;  // last turn's full prompt
    std::string past_gen;     // last turn's generated text

    // Tracks whether this instance pinned an HTP session; releases on last handoff.
    htp::SessionGuard htp_guard_;

   public:
    ~LlamaVlm() override;

    virtual int32_t create(const geniex_VlmCreateInput* input) override;

    virtual int32_t reset() override;

    virtual int32_t apply_chat_template(
        const geniex_VlmApplyChatTemplateInput* input, geniex_VlmApplyChatTemplateOutput* output) override;

    virtual int32_t generate(const geniex_VlmGenerateInput* input, geniex_VlmGenerateOutput* output) override;

    virtual int32_t get_capabilities(geniex_VlmCapabilities* output) override;

   private:
    void set_sampler(const geniex_SamplerConfig* cfg);
    bool vlm_message_to_common_chat_msg(const geniex_VlmChatMessage* input, common_chat_msg* output);
};

}  // namespace geniex
