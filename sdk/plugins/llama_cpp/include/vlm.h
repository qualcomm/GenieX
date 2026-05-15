#pragma once

#include "chat.h"
#include "htp_session.h"
#include "mtmd.h"
#include "plugin/IVlm.h"
#include "sampling.h"

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

    // Conversation state tracking
    int32_t n_past              = 0;
    int32_t global_n_past_chars = 0;  // Track character position in prompt text (not tokens)

    // Tracks whether this instance pinned an HTP session; releases on last handoff.
    htp::SessionGuard htp_guard_;

   public:
    ~LlamaVlm() override;

    virtual int32_t create_impl(const geniex_VlmCreateInput* input) override;

    virtual int32_t reset() override;

    virtual int32_t apply_chat_template(
        const geniex_VlmApplyChatTemplateInput* input, geniex_VlmApplyChatTemplateOutput* output) override;

    virtual int32_t generate(const geniex_VlmGenerateInput* input, geniex_VlmGenerateOutput* output) override;

   private:
    void reset_sampler();
    bool vlm_message_to_common_chat_msg(const geniex_VlmChatMessage* input, common_chat_msg* output);
};

}  // namespace geniex
