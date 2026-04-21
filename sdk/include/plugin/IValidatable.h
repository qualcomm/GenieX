#pragma once

/**
 * @file IValidatable.h
 * @brief Generic validation interface template for plugin modalities
 *
 * This header contains the IValidatable template interface that provides
 * validation capabilities for any plugin modality. Implementation classes
 * can optionally inherit from this interface to add validation support.
 */

#include "geniex.h"
#include "logging.h"
#ifdef GENIEX_VALIDATION
#include "validation.h"
#endif

namespace geniex {

// =============================================================================
// Validatable Interface Template
// =============================================================================

/**
 * @brief Generic interface for validation logic
 * @tparam CreateInput The type of creation input to validate (e.g., geniex_LlmCreateInput)
 *
 * This interface encapsulates the validation concern and can be composed
 * with any plugin interface to add validation capabilities.
 */
template <typename CreateInput>
class IValidatable {
   public:
    virtual ~IValidatable() = default;

    /**
     * @brief Determine if validation is needed for the given creation input
     * @param input The creation input to check for validation requirements
     * @return true if validation should be performed, false otherwise
     */
    virtual bool is_validation_needed(const CreateInput* input) = 0;

    /**
     * @brief Generic validation logic that can be used by all modalities
     * @param input The creation input to validate
     * @return ML error code (GENIEX_SUCCESS on success, negative on failure)
     */
    int32_t validate(const CreateInput* input) {
        if (!input) {
            GENIEX_LOG_ERROR("Create input is nullptr");
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        // Check if validation is needed (entirely compiled out when validation is disabled)
#ifdef GENIEX_VALIDATION
        if (is_validation_needed(input)) {
            GENIEX_LOG_INFO("Validation required, performing validation checks...");
            auto validation_result = geniex::validation::validate_license("", "");
            return validation_result.result;
        }
#else
        GENIEX_LOG_INFO("Validation skipped (validation disabled)");
#endif

        return GENIEX_SUCCESS;
    }
};

}  // namespace geniex
