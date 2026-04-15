#pragma once

#include <memory>
#include <string>

#include "pipeline/llm_pipeline.h"
#include "plugin/ILlm.h"

namespace geniex {

class QairtLlm : public ILlm {
    std::unique_ptr<LLMPipeline> pipeline_;
    std::string model_name_;
    bool enable_thinking_ = false;

   public:
    virtual ~QairtLlm() override;

    virtual int32_t create_impl(const ml_LlmCreateInput*) override;

    virtual int32_t reset() override;

    virtual int32_t save_kv_cache(const ml_KvCacheSaveInput*, ml_KvCacheSaveOutput*) override;
    virtual int32_t load_kv_cache(const ml_KvCacheLoadInput*, ml_KvCacheLoadOutput*) override;

    virtual int32_t apply_chat_template(const ml_LlmApplyChatTemplateInput*, ml_LlmApplyChatTemplateOutput*) override;

    virtual int32_t generate(const ml_LlmGenerateInput*, ml_LlmGenerateOutput*) override;
};

}  // namespace geniex
