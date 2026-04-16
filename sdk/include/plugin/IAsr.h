#pragma once

#include "IValidatable.h"
#include "ml.h"

namespace geniex {

class IAsr {
   public:
    virtual ~IAsr() = default;

    /**
     * @brief Create the ASR model with optional validation
     * @param input The creation input parameters
     * @return ML error code (ML_SUCCESS on success, negative on failure)
     */
    virtual int32_t create(const ml_AsrCreateInput* input) {
        // Check if this instance implements IValidatable
        auto* validatable = dynamic_cast<IValidatable<ml_AsrCreateInput>*>(this);
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

    virtual int32_t transcribe(const ml_AsrTranscribeInput*, ml_AsrTranscribeOutput*) = 0;

    virtual int32_t list_supported_languages(
        const ml_AsrListSupportedLanguagesInput*, ml_AsrListSupportedLanguagesOutput*) = 0;

    // Streaming ASR interface - default implementations for optional streaming support
    virtual int32_t stream_begin(const ml_AsrStreamBeginInput* input, ml_AsrStreamBeginOutput* output) {
        return ML_ERROR_COMMON_NOT_SUPPORTED;  // Default: streaming not supported
    }

    virtual int32_t stream_push_audio(const ml_AsrStreamPushAudioInput* input) {
        return ML_ERROR_COMMON_NOT_SUPPORTED;  // Default: streaming not supported
    }

    virtual int32_t stream_stop(const ml_AsrStreamStopInput* input) {
        return ML_ERROR_COMMON_NOT_SUPPORTED;  // Default: streaming not supported
    }

   protected:
    /**
     * @brief Pure virtual method for actual model creation implementation
     * @param input The creation input parameters
     * @return ML error code (ML_SUCCESS on success, negative on failure)
     */
    virtual int32_t create_impl(const ml_AsrCreateInput* input) = 0;
};

}  // namespace geniex
