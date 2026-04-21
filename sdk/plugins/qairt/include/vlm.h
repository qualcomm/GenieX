#pragma once

#include <memory>
#include <string>

#include "external/mm-process-interface.h"
#include "external/tokenizers_cpp.h"
#include "plugin/IVlm.h"
#include "vlm/vlm_model.h"
#include "vlm/vlm_types.h"

namespace geniex {

class QairtVlm : public IVlm {
    std::unique_ptr<VLMModel>              model_;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

    // Processor variant — exactly one is non-null after create.
    std::unique_ptr<mm_process::qwen2_5_omni::Qwen2_5OmniProcessor> omni_processor_;
    std::unique_ptr<mm_process::qwen3vl::Qwen3VLProcessor>          qwen3vl_processor_;

    std::string model_name_;
    std::string tokenizer_path_;
    bool        enable_thinking_ = false;

   public:
    virtual ~QairtVlm() override;

    virtual int32_t create_impl(const geniex_VlmCreateInput*) override;

    virtual int32_t reset() override;

    virtual int32_t apply_chat_template(
        const geniex_VlmApplyChatTemplateInput*, geniex_VlmApplyChatTemplateOutput*) override;

    virtual int32_t generate(const geniex_VlmGenerateInput*, geniex_VlmGenerateOutput*) override;
};

}  // namespace geniex
