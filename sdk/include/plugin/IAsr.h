#pragma once

#include "IValidatable.h"
#include "geniex.h"

namespace geniex {

class IAsr {
   public:
    virtual ~IAsr() = default;

    /**
     * @brief Create the ASR model with optional validation
     * @param input The creation input parameters
     * @return ML error code (GENIEX_SUCCESS on success, negative on failure)
     */
    virtual int32_t create(const geniex_AsrCreateInput* input) {
        // Check if this instance implements IValidatable
        auto* validatable = dynamic_cast<IValidatable<geniex_AsrCreateInput>*>(this);
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

    virtual int32_t transcribe(const geniex_AsrTranscribeInput*, geniex_AsrTranscribeOutput*) = 0;

    virtual int32_t list_supported_languages(
        const geniex_AsrListSupportedLanguagesInput*, geniex_AsrListSupportedLanguagesOutput*) = 0;

    // Streaming ASR interface - default implementations for optional streaming support
    virtual int32_t stream_begin(const geniex_AsrStreamBeginInput* input, geniex_AsrStreamBeginOutput* output) {
        return GENIEX_ERROR_COMMON_NOT_SUPPORTED;  // Default: streaming not supported
    }

    virtual int32_t stream_push_audio(const geniex_AsrStreamPushAudioInput* input) {
        return GENIEX_ERROR_COMMON_NOT_SUPPORTED;  // Default: streaming not supported
    }

    virtual int32_t stream_stop(const geniex_AsrStreamStopInput* input) {
        return GENIEX_ERROR_COMMON_NOT_SUPPORTED;  // Default: streaming not supported
    }

   protected:
    /**
     * @brief Pure virtual method for actual model creation implementation
     * @param input The creation input parameters
     * @return ML error code (GENIEX_SUCCESS on success, negative on failure)
     */
    virtual int32_t create_impl(const geniex_AsrCreateInput* input) = 0;
};

}  // namespace geniex
