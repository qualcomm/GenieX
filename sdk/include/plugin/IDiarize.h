#pragma once

#include "IValidatable.h"
#include "ml.h"

namespace geniex {

class IDiarize {
   public:
    virtual ~IDiarize() = default;

    /**
     * @brief Create the diarization model with optional validation
     * @param input The creation input parameters
     * @return ML error code (ML_SUCCESS on success, negative on failure)
     */
    virtual int32_t create(const ml_DiarizeCreateInput* input) {
        // Check if this instance implements IValidatable
        auto* validatable = dynamic_cast<IValidatable<ml_DiarizeCreateInput>*>(this);
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

    virtual int32_t infer(const ml_DiarizeInferInput* input, ml_DiarizeInferOutput* output) = 0;

   protected:
    /**
     * @brief Pure virtual method for actual model creation implementation
     * @param input The creation input parameters
     * @return ML error code (ML_SUCCESS on success, negative on failure)
     */
    virtual int32_t create_impl(const ml_DiarizeCreateInput* input) = 0;
};

}  // namespace geniex
