#pragma once

#include <memory>
#include <string>

#include "pipeline/llm_pipeline.h"
#include "plugin/ILlm.h"

namespace geniex {

class QairtLlm : public ILlm {
    std::unique_ptr<LLMPipeline> pipeline_;
    std::string                  model_name_;
    bool                         enable_thinking_ = false;

   public:
    virtual ~QairtLlm() override;

    virtual int32_t create_impl(const geniex_LlmCreateInput*) override;

    virtual int32_t reset() override;

    virtual int32_t save_kv_cache(const geniex_KvCacheSaveInput*, geniex_KvCacheSaveOutput*) override;
    virtual int32_t load_kv_cache(const geniex_KvCacheLoadInput*, geniex_KvCacheLoadOutput*) override;

    virtual int32_t apply_chat_template(
        const geniex_LlmApplyChatTemplateInput*, geniex_LlmApplyChatTemplateOutput*) override;

    virtual int32_t generate(const geniex_LlmGenerateInput*, geniex_LlmGenerateOutput*) override;
};

}  // namespace geniex
