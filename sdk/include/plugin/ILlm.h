#pragma once

#include "IValidatable.h"
#include "ml.h"

namespace geniex {

class ILlm {
   public:
    virtual ~ILlm() = default;

    /**
     * @brief Create the LLM model with optional validation
     * @param input The creation input parameters
     * @return ML error code (ML_SUCCESS on success, negative on failure)
     */
    virtual int32_t create(const ml_LlmCreateInput* input) {
        // Check if this instance implements IValidatable
        auto* validatable = dynamic_cast<IValidatable<ml_LlmCreateInput>*>(this);
        if (validatable) {
            // Check if validation is needed
            if (validatable->is_validation_needed(input)) {
                // Perform validation
                int32_t validation_result = validatable->validate(input);
                if (validation_result != ML_SUCCESS) {
                    return validation_result;
                }
            }
        }

        // Call the actual implementation
        return create_impl(input);
    }

    virtual int32_t reset() = 0;

    virtual int32_t save_kv_cache(const ml_KvCacheSaveInput*, ml_KvCacheSaveOutput*) = 0;
    virtual int32_t load_kv_cache(const ml_KvCacheLoadInput*, ml_KvCacheLoadOutput*) = 0;

    virtual int32_t apply_chat_template(const ml_LlmApplyChatTemplateInput*, ml_LlmApplyChatTemplateOutput*) = 0;

    virtual int32_t generate(const ml_LlmGenerateInput*, ml_LlmGenerateOutput*) = 0;

   protected:
    /**
     * @brief Pure virtual method for actual model creation implementation
     * @param input The creation input parameters
     * @return ML error code (ML_SUCCESS on success, negative on failure)
     */
    virtual int32_t create_impl(const ml_LlmCreateInput* input) = 0;
};

}  // namespace geniex
