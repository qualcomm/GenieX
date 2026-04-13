#pragma once

#include <memory>
#include <string>

#include "ml.h"
#include "plugin/IVlm.h"

namespace geniex {

/**
 * @brief QNN-accelerated Vision Language Model implementation
 *
 * Factory class for creating platform-specific VLM models on Qualcomm NPU.
 */
class QnnVlm : public IVlm {
   public:
    explicit QnnVlm(std::string lib_path = "");
    virtual ~QnnVlm() override;

    virtual int32_t reset() override;
    virtual int32_t apply_chat_template(const ml_VlmApplyChatTemplateInput*, ml_VlmApplyChatTemplateOutput*) override;
    virtual int32_t generate(const ml_VlmGenerateInput*, ml_VlmGenerateOutput*) override;

   protected:
    virtual int32_t create_impl(const ml_VlmCreateInput* input) override;

   private:
    std::unique_ptr<IVlm> m_model_impl;
    std::string           m_model_name;
    std::string           m_lib_path;
};

}  // namespace geniex

// export from qnn-run
extern "C" {
#if defined(__ANDROID__)
// Android supports omni-neural, qwen3vl, and autoneural
geniex::IVlm* create_qnn_omni_neural();
geniex::IVlm* create_qnn_qwen3vl();
geniex::IVlm* create_qnn_autoneural();

#elif defined(_WIN32)
// Windows supports VLM models
geniex::IVlm* create_qnn_omni_neural();
geniex::IVlm* create_qnn_qwen3vl();
geniex::IVlm* create_qnn_autoneural();

#elif defined(__linux__)
// Linux supports autoneural
geniex::IVlm* create_qnn_autoneural();

#endif
}
