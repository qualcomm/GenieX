#pragma once

#include <memory>
#include <string>

#include "plugin/IReranker.h"

namespace geniex {

/**
 * @brief QNN-accelerated document reranking model implementation
 *
 * Factory class for creating reranker models on Qualcomm NPU.
 */
class QnnReranker : public IReranker {
   public:
    explicit QnnReranker(std::string lib_path = "");
    ~QnnReranker() override;

    int32_t rerank(const ml_RerankerRerankInput* input, ml_RerankerRerankOutput* output) override;

   protected:
    virtual int32_t create_impl(const ml_RerankerCreateInput* input) override;

   private:
    std::unique_ptr<IReranker> m_model_impl;
    std::string                m_lib_path;
};

}  // namespace geniex

// export from qnn-run
extern "C" {
#if defined(__ANDROID__)
geniex::IReranker* create_qnn_jina_rerank();

#elif defined(_WIN32)
geniex::IReranker* create_qnn_jina_rerank();

#elif defined(__linux__)
geniex::IReranker* create_qnn_jina_rerank();

#endif
}
