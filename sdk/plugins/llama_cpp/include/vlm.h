// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "chat.h"
#include "htp_session.h"
#include "mtmd.h"
#include "plugin/IVlm.h"
#include "sampling.h"
#include "threadpool.h"
#include "vlm_prefix.h"

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

    struct Media {
        VlmPrefixChunk::Kind kind = VlmPrefixChunk::Kind::Text;
        std::string          id;
        std::string          path;
        bool                 explicit_id   = false;
        uint32_t             width         = 0;
        uint32_t             height        = 0;
        size_t               audio_samples = 0;
    };

    struct Pending {
        std::string        prompt;
        std::vector<Media> media;
    };

    int32_t                     n_past = 0;
    std::vector<VlmPrefixChunk> cached_chunks;
    std::vector<Media>          cached_media;
    Pending                     pending;

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
    bool vlm_message_to_common_chat_msg(
        const geniex_VlmChatMessage* input, common_chat_msg* output, size_t media_offset);
};

}  // namespace geniex
