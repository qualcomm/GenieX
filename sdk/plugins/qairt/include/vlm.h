#pragma once

#include <memory>
#include <string>

#include "pipeline/vlm_pipeline.h"
#include "plugin/IVlm.h"

namespace geniex {

class QairtVlm : public IVlm {
    std::unique_ptr<VLMPipeline> pipeline_;

    std::string model_name_;
    bool        enable_thinking_ = false;

    // Incremental history tracking.
    // history_size_         — messages already committed to the KV cache (advanced by generate()).
    // pending_history_size_ — messages in the last apply_chat_template call (committed on next generate()).
    size_t history_size_         = 0;
    size_t pending_history_size_ = 0;

   public:
    virtual ~QairtVlm() override;

    virtual int32_t create_impl(const geniex_VlmCreateInput*) override;

    virtual int32_t reset() override;

    virtual int32_t apply_chat_template(
        const geniex_VlmApplyChatTemplateInput*, geniex_VlmApplyChatTemplateOutput*) override;

    virtual int32_t generate(const geniex_VlmGenerateInput*, geniex_VlmGenerateOutput*) override;
};

}  // namespace geniex
