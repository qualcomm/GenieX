#include "embedding.h"

#include <cstring>
#include <memory>
#include <string>
#include <string_view>

#include "common.h"
#include "logging.h"
#include "ml.h"

namespace geniex {

// =============================================================================
// QnnEmbedding Factory Implementation
// =============================================================================

QnnEmbedding::QnnEmbedding(std::string lib_path) : m_model_impl(), m_lib_path(std::move(lib_path)) {}

QnnEmbedding::~QnnEmbedding() {}

/**
 * @brief Create embedding model implementation (embed-gemma and embedneural supported on all platforms)
 *
 * @param input Model creation input with model name and config
 * @return ML_SUCCESS or error code
 */
int32_t QnnEmbedding::create_impl(const ml_EmbedderCreateInput* input) {
    if (!input || !input->model_name) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    GENIEX_LOG_TRACE("Creating specific npu Embedding model for: {}", fmt::to_string(input->model_name));

    std::string_view model_name(input->model_name);

#if defined(__ANDROID__)
    if (model_name == "embed-gemma") {
        m_model_impl = std::unique_ptr<IEmbedding>(create_qnn_embed_gemma());
    } else if (model_name == "embedneural") {
        m_model_impl = std::unique_ptr<IEmbedding>(create_qnn_embed_neural());
    } else {
        GENIEX_LOG_ERROR("Unsupported model name for npu Embedding implementation: {}", model_name);
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

#elif defined(_WIN32)
    // Windows: support both embed-gemma and embedneural
    if (model_name == "embed-gemma") {
        m_model_impl = std::unique_ptr<IEmbedding>(create_qnn_embed_gemma());
    } else if (model_name == "embedneural") {
        m_model_impl = std::unique_ptr<IEmbedding>(create_qnn_embed_neural());
    } else {
        GENIEX_LOG_ERROR("Unsupported model name for npu Embedding implementation: {}", model_name);
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

#elif defined(__linux__)
    // Linux: support both embed-gemma and embedneural
    if (model_name == "embed-gemma") {
        m_model_impl = std::unique_ptr<IEmbedding>(create_qnn_embed_gemma());
    } else if (model_name == "embedneural") {
        m_model_impl = std::unique_ptr<IEmbedding>(create_qnn_embed_neural());
    } else {
        GENIEX_LOG_ERROR("Unsupported model name for Linux npu Embedding implementation: {}", model_name);
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

#endif

    if (!m_model_impl) {
        GENIEX_LOG_ERROR("Failed to create specific model implementation");
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

    // inject qnn path, use class for c_str lifetime management
    QnnFolderPathFiller _(input, m_lib_path);
    return m_model_impl->create(input);
}

/**
 * @brief Generate embeddings for input text
 *
 * @param input Text to embed
 * @param output Embedding vector
 * @return ML_SUCCESS or error code
 */
int32_t QnnEmbedding::embed(const ml_EmbedderEmbedInput* input, ml_EmbedderEmbedOutput* output) {
    if (!m_model_impl || !input || !output) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    return m_model_impl->embed(input, output);
}

/**
 * @brief Get embedding vector dimensionality
 *
 * @param output Embedding dimension value
 * @return ML_SUCCESS or error code
 */
int32_t QnnEmbedding::embedding_dim(ml_EmbedderDimOutput* output) {
    if (!m_model_impl || !output) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    return m_model_impl->embedding_dim(output);
}

}  // namespace geniex
