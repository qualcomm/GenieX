#pragma once

#include <memory>
#include <string>
#include <vector>

#include "llama.h"
#include "plugin/IReranker.h"

namespace geniex {

class LlamaCppReranker : public IReranker {
   public:
    LlamaCppReranker();
    ~LlamaCppReranker() override;

    int32_t create_impl(const geniex_RerankerCreateInput* input) override;
    int32_t rerank(const geniex_RerankerRerankInput* input, geniex_RerankerRerankOutput* output) override;

   private:
    // Model and context
    llama_model*       model_ = nullptr;
    llama_context*     ctx_   = nullptr;
    const llama_vocab* vocab_ = nullptr;

    // Model parameters
    int32_t     n_embd_          = 1024;     // Embedding dimension
    int32_t     n_ctx_           = 2560;     // Context window size
    int32_t     n_batch_         = 2560;     // Batch size for processing
    int32_t     n_ubatch_        = 2560;     // Physical batch size for processing
    uint32_t    n_cls_out_       = 1;        // Number of classification outputs
    const char* rerank_template_ = nullptr;  // Rerank template from model

    // Helper methods
    void batch_add_seq(llama_batch& batch, const std::vector<int32_t>& tokens, llama_seq_id seq_id);
    int  process_batch(llama_batch& batch, float* output, int n_seq);
};

}  // namespace geniex
