#pragma once

#include <memory>
#include <string>

#include "ml.h"
#include "plugin/ILlm.h"

namespace geniex {

/**
 * @brief QNN-accelerated Large Language Model implementation
 *
 * Factory class for creating platform-specific LLM models on Qualcomm NPU.
 */
class QnnLlm : public ILlm {
   public:
    explicit QnnLlm(std::string lib_path = "");
    virtual ~QnnLlm() override;

    virtual int32_t reset() override;
    virtual int32_t save_kv_cache(const ml_KvCacheSaveInput*, ml_KvCacheSaveOutput*) override;
    virtual int32_t load_kv_cache(const ml_KvCacheLoadInput*, ml_KvCacheLoadOutput*) override;
    virtual int32_t apply_chat_template(const ml_LlmApplyChatTemplateInput*, ml_LlmApplyChatTemplateOutput*) override;
    virtual int32_t generate(const ml_LlmGenerateInput*, ml_LlmGenerateOutput*) override;

   protected:
    virtual int32_t create_impl(const ml_LlmCreateInput* input) override;

   private:
    std::unique_ptr<ILlm> m_model_impl;
    std::string           m_lib_path;
};

}  // namespace geniex

// export from qnn-run
extern "C" {
#if defined(__ANDROID__)
// Android only supports liquid-sdk
geniex::ILlm* create_qnn_qwen3_4b();
geniex::ILlm* create_qnn_llama3_3b();
geniex::ILlm* create_qnn_phi4();
geniex::ILlm* create_qnn_phi3_5();
geniex::ILlm* create_qnn_granite4();
geniex::ILlm* create_qnn_liquid();
geniex::ILlm* create_qnn_granite_nano();

#elif defined(_WIN32)
// Windows supports all LLM SDKs
geniex::ILlm* create_qnn_qwen3_4b();
geniex::ILlm* create_qnn_qwen3_8b();
geniex::ILlm* create_qnn_llama3_3b();
geniex::ILlm* create_qnn_phi4();
geniex::ILlm* create_qnn_phi3_5();
geniex::ILlm* create_qnn_granite4();
geniex::ILlm* create_qnn_liquid();
geniex::ILlm* create_qnn_granite_nano();
geniex::ILlm* create_qnn_smolvla();
geniex::ILlm* create_qnn_ministral3_3b();
geniex::ILlm* create_qnn_hymt();

#elif defined(__linux__)
geniex::ILlm* create_qnn_qwen3_4b();
geniex::ILlm* create_qnn_qwen3_8b();
geniex::ILlm* create_qnn_llama3_3b();
geniex::ILlm* create_qnn_phi4();
geniex::ILlm* create_qnn_phi3_5();
geniex::ILlm* create_qnn_granite4();
geniex::ILlm* create_qnn_liquid();
geniex::ILlm* create_qnn_granite_nano();

#endif
}
