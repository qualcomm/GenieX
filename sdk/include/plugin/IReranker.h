#pragma once

#include "../geniex.h"
#include "IValidatable.h"

namespace geniex {

class IReranker {
   public:
    virtual ~IReranker() = default;

    /**
     * @brief Create the Reranker model with optional validation
     * @param input The creation input parameters
     * @return ML error code (GENIEX_SUCCESS on success, negative on failure)
     */
    virtual int32_t create(const geniex_RerankerCreateInput* input) {
        // Check if this instance implements IValidatable
        auto* validatable = dynamic_cast<IValidatable<geniex_RerankerCreateInput>*>(this);
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

    virtual int32_t rerank(const geniex_RerankerRerankInput*, geniex_RerankerRerankOutput*) = 0;

   protected:
    /**
     * @brief Pure virtual method for actual model creation implementation
     * @param input The creation input parameters
     * @return ML error code (GENIEX_SUCCESS on success, negative on failure)
     */
    virtual int32_t create_impl(const geniex_RerankerCreateInput* input) = 0;
};

}  // namespace geniex
