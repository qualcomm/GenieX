#include "rerank.h"

#include <cstring>
#include <memory>
#include <string>
#include <string_view>

#include "common.h"
#include "logging.h"
#include "ml.h"

namespace geniex {

// =============================================================================
// QnnReranker Factory Implementation
// =============================================================================

QnnReranker::QnnReranker(std::string lib_path) : m_model_impl(), m_lib_path(std::move(lib_path)) {}

QnnReranker::~QnnReranker() {}

/**
 * @brief Create reranker model implementation (jina-rerank supported on all platforms)
 *
 * @param input Model creation input with model name and config
 * @return ML_SUCCESS or error code
 */
int32_t QnnReranker::create_impl(const ml_RerankerCreateInput* input) {
    if (!input || !input->model_name) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    GENIEX_LOG_TRACE("Creating specific npu Reranker model for: {}", fmt::to_string(input->model_name));

    std::string_view model_name(input->model_name);

#if defined(__ANDROID__)
    if (model_name == "jina-rerank") {
        m_model_impl = std::unique_ptr<IReranker>(create_qnn_jina_rerank());
    } else {
        GENIEX_LOG_ERROR("Unsupported model name for npu Reranker implementation: {}", model_name);
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

#elif defined(_WIN32)
    if (model_name == "jina-rerank") {
        m_model_impl = std::unique_ptr<IReranker>(create_qnn_jina_rerank());
    } else {
        GENIEX_LOG_ERROR("Unsupported model name for npu Reranker implementation: {}", model_name);
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

#elif defined(__linux__)
    // Linux: jina-rerank is supported
    if (model_name == "jina-rerank") {
        m_model_impl = std::unique_ptr<IReranker>(create_qnn_jina_rerank());
    } else {
        GENIEX_LOG_ERROR("Unsupported model name for Linux npu Reranker implementation: {}", model_name);
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
 * @brief Rerank documents based on query relevance
 *
 * @param input Query and document list
 * @param output Reranked documents with relevance scores
 * @return ML_SUCCESS or error code
 */
int32_t QnnReranker::rerank(const ml_RerankerRerankInput* input, ml_RerankerRerankOutput* output) {
    if (!m_model_impl || !input || !output) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    return m_model_impl->rerank(input, output);
}

}  // namespace geniex
