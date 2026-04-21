#pragma once

#include "IValidatable.h"
#include "geniex.h"

namespace geniex {

class IVlm {
   public:
    virtual ~IVlm() = default;

    /**
     * @brief Create the VLM model with optional validation
     * @param input The creation input parameters
     * @return ML error code (GENIEX_SUCCESS on success, negative on failure)
     */
    virtual int32_t create(const geniex_VlmCreateInput* input) {
        // Check if this instance implements IValidatable
        auto* validatable = dynamic_cast<IValidatable<geniex_VlmCreateInput>*>(this);
        if (validatable) {
            // Check if validation is needed
            if (validatable->is_validation_needed(input)) {
                // Perform validation
                int32_t validation_result = validatable->validate(input);
                if (validation_result != GENIEX_SUCCESS) {
                    return validation_result;
                }
            }
        }

        // Call the actual implementation
        return create_impl(input);
    }

    virtual int32_t reset() = 0;

    virtual int32_t apply_chat_template(
        const geniex_VlmApplyChatTemplateInput* input, geniex_VlmApplyChatTemplateOutput* output) = 0;

    virtual int32_t generate(const geniex_VlmGenerateInput* input, geniex_VlmGenerateOutput* output) = 0;

   protected:
    /**
     * @brief Pure virtual method for actual model creation implementation
     * @param input The creation input parameters
     * @return ML error code (GENIEX_SUCCESS on success, negative on failure)
     */
    virtual int32_t create_impl(const geniex_VlmCreateInput* input) = 0;
};

}  // namespace geniex
