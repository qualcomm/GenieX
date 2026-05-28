#pragma once

#include <memory>

#include "llm/llm_spec_loader.h"  // ParsedSamplerConfig
#include "pipeline/vlm_pipeline.h"
#include "plugin/IVlm.h"

namespace geniex {

class QairtVlm : public IVlm {
    std::unique_ptr<VLMPipeline> pipeline_;

    bool enable_thinking_ = false;
    // True iff a vision encoder shard was located at create time. Audio is not
    // wired into the QAIRT VLM pipeline yet, so always reported as unsupported.
    bool has_vision_encoder_ = false;

    // Bundle's `dialog.sampler` defaults; parsed once at create_impl().
    ParsedSamplerConfig bundle_sampler_;

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

    virtual int32_t get_capabilities(geniex_VlmCapabilities* output) override;
};

}  // namespace geniex
