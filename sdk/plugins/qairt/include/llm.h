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

    /**
     * @brief Apply chat template to format conversation messages
     *
     * @param input Message array to format
     * @param output Formatted text output
     *
     * @note Memory ownership: output->formatted_text is allocated by this method
     *       using malloc() and must be freed by the caller using ml_free()
     */
    virtual int32_t apply_chat_template(const ml_LlmApplyChatTemplateInput*, ml_LlmApplyChatTemplateOutput*) override;

    /**
     * @brief Generate text from a prompt
     *
     * @param input Generation parameters and prompt
     * @param output Generated text output
     *
     * @note Memory ownership: output->full_text is allocated by this method
     *       using malloc() and must be freed by the caller using ml_free()
     */
    virtual int32_t generate(const ml_LlmGenerateInput*, ml_LlmGenerateOutput*) override;

   protected:
    virtual int32_t create_impl(const ml_LlmCreateInput* input) override;

   private:
    std::unique_ptr<ILlm> m_model_impl;
    std::string           m_lib_path;
};

ML_API ILlm* create_qairt_llm(const char* lib_path);

}  // namespace geniex
