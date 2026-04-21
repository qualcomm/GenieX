#pragma once

#include "IValidatable.h"
#include "geniex.h"

namespace geniex {

class ITts {
   public:
    virtual ~ITts() = default;

    /**
     * @brief Create the TTS model with optional validation
     * @param input The creation input parameters
     * @return ML error code (GENIEX_SUCCESS on success, negative on failure)
     */
    virtual int32_t create(const geniex_TtsCreateInput* input) {
        // Check if this instance implements IValidatable
        auto* validatable = dynamic_cast<IValidatable<geniex_TtsCreateInput>*>(this);
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

    virtual int32_t synthesize(const geniex_TtsSynthesizeInput*, geniex_TtsSynthesizeOutput*) = 0;

    virtual int32_t list_available_voices(
        const geniex_TtsListAvailableVoicesInput*, geniex_TtsListAvailableVoicesOutput*) = 0;

   protected:
    /**
     * @brief Pure virtual method for actual model creation implementation
     * @param input The creation input parameters
     * @return ML error code (GENIEX_SUCCESS on success, negative on failure)
     */
    virtual int32_t create_impl(const geniex_TtsCreateInput* input) = 0;
};

}  // namespace geniex
