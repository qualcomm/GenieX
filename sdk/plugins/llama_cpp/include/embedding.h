#pragma once

#include <memory>
#include <vector>

#include "htp_session.h"
#include "llama.h"
#include "plugin/IEmbedding.h"

namespace geniex {

struct EmbeddingState {
    llama_model*       model       = nullptr;
    llama_context*     ctx         = nullptr;
    const llama_vocab* vocab       = nullptr;
    int32_t            n_embd      = 1024;  // Embedding dimension
    int32_t            n_ctx       = 2560;  // Context window size
    int32_t            n_batch     = 2560;  // Batch size for processing
    int32_t            n_ubatch    = 2560;  // Physical batch size for processing
    bool               initialized = false;
};

class LlamaCppEmbedding : public IEmbedding {
   public:
    LlamaCppEmbedding();
    ~LlamaCppEmbedding() override;

    int32_t create_impl(const geniex_EmbedderCreateInput* input) override;
    int32_t embed(const geniex_EmbedderEmbedInput* input, geniex_EmbedderEmbedOutput* output) override;
    int32_t embedding_dim(geniex_EmbedderDimOutput* output) override;

   private:
    std::unique_ptr<EmbeddingState> state_;

    // Tracks whether this instance pinned an HTP session; releases on last handoff.
    htp::SessionGuard htp_guard_;

    // Helper methods
    void normalize_embeddings(const float* inp, float* out, int n, const char* method);
    void batch_add_seq(llama_batch& batch, const std::vector<int32_t>& tokens, llama_seq_id seq_id, int32_t capacity);
    int  process_batch(llama_batch& batch, float* output, int n_seq, int n_embd, const char* norm_method);
    void cleanup();
};

}  // namespace geniex
