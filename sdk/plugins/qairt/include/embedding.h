#pragma once

#include <memory>
#include <string>

#include "plugin/IEmbedding.h"

namespace geniex {

/**
 * @brief QNN-accelerated text embedding model implementation
 *
 * Factory class for creating embedding models on Qualcomm NPU.
 */
class QnnEmbedding : public IEmbedding {
   public:
    explicit QnnEmbedding(std::string lib_path = "");
    ~QnnEmbedding() override;

    int32_t embed(const ml_EmbedderEmbedInput* input, ml_EmbedderEmbedOutput* output) override;
    int32_t embedding_dim(ml_EmbedderDimOutput* output) override;

   protected:
    virtual int32_t create_impl(const ml_EmbedderCreateInput* input) override;

   private:
    std::unique_ptr<IEmbedding> m_model_impl;
    std::string                 m_lib_path;
};

}  // namespace geniex

// export from qnn-run
extern "C" {
#if defined(__ANDROID__)
geniex::IEmbedding* create_qnn_embed_gemma();
geniex::IEmbedding* create_qnn_embed_neural();
#elif defined(_WIN32)
geniex::IEmbedding* create_qnn_embed_gemma();
geniex::IEmbedding* create_qnn_embed_neural();

#elif defined(__linux__)
geniex::IEmbedding* create_qnn_embed_gemma();
geniex::IEmbedding* create_qnn_embed_neural();

#endif
}
