#pragma once

#include <optional>

#include "htp_session.h"
#include "llama.h"
#include "plugin/ILlm.h"
#include "sampling.h"
#include "threadpool.h"

namespace geniex {

class LlamaLlm : public ILlm {
    llama_model*               model   = nullptr;
    llama_context*             ctx     = nullptr;
    common_sampler*            sampler = nullptr;
    Threadpools                pools_;
    std::optional<std::string> chat_template_str = std::nullopt;

    int                      n_past_global = 0;
    int                      n_past        = 0;   // for context shifting
    std::vector<llama_token> past_prompt_tokens;  // for prefix match

    // Instance-level storage for model loading parameters (thread-safe)
    llama_model_tensor_buft_override tensor_overrides[2];           // MoE override + null terminator
    bool                             allow_special_tokens = false;  // Control special token output

    // Tracks whether this instance pinned an HTP session; releases on last handoff.
    htp::SessionGuard htp_guard_;

   public:
    virtual ~LlamaLlm() override;

    virtual int32_t create_impl(const geniex_LlmCreateInput*) override;

    virtual int32_t reset() override;

    virtual int32_t save_kv_cache(const geniex_KvCacheSaveInput*, geniex_KvCacheSaveOutput*) override;
    virtual int32_t load_kv_cache(const geniex_KvCacheLoadInput*, geniex_KvCacheLoadOutput*) override;

    virtual int32_t apply_chat_template(
        const geniex_LlmApplyChatTemplateInput*, geniex_LlmApplyChatTemplateOutput*) override;

    virtual int32_t generate(const geniex_LlmGenerateInput*, geniex_LlmGenerateOutput*) override;

   private:
    void reset_sampler();
    void set_sampler(const geniex_SamplerConfig* cfg);
};

}  // namespace geniex
