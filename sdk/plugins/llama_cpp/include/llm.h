#pragma once

#include <optional>

#include "llama.h"
#include "plugin/ILlm.h"
#include "sampling.h"

namespace geniex {

class LlamaLlm : public ILlm {
    llama_model*     model                       = nullptr;
    llama_context*   ctx                         = nullptr;
    common_sampler*  sampler                     = nullptr;
    ggml_threadpool* threadpool                  = nullptr;
    ggml_threadpool* threadpool_batch            = nullptr;
    void (*threadpool_free_fn)(ggml_threadpool*) = nullptr;
    std::optional<std::string> chat_template_str = std::nullopt;

    int                      n_past_global = 0;
    int                      n_past        = 0;   // for context shifting
    std::vector<llama_token> past_prompt_tokens;  // for prefix match

    // Instance-level storage for model loading parameters (thread-safe)
    llama_model_tensor_buft_override tensor_overrides[2];           // MoE override + null terminator
    bool                             allow_special_tokens = false;  // Control special token output

   public:
    virtual ~LlamaLlm() override;

    virtual int32_t create_impl(const ml_LlmCreateInput*) override;

    virtual int32_t reset() override;

    virtual int32_t save_kv_cache(const ml_KvCacheSaveInput*, ml_KvCacheSaveOutput*) override;
    virtual int32_t load_kv_cache(const ml_KvCacheLoadInput*, ml_KvCacheLoadOutput*) override;

    virtual int32_t apply_chat_template(const ml_LlmApplyChatTemplateInput*, ml_LlmApplyChatTemplateOutput*) override;

    virtual int32_t generate(const ml_LlmGenerateInput*, ml_LlmGenerateOutput*) override;

   private:
    ml_ModelConfig model_config_default(void);

    void reset_sampler();
    void set_sampler(const ml_SamplerConfig* cfg);
};

}  // namespace geniex
