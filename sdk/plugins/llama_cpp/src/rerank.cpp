#include "rerank.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "common.h"
#include "logging.h"
#include "profiler.h"
#include "utils.h"

namespace geniex {

LlamaCppReranker::LlamaCppReranker() { GENIEX_LOG_INFO("Initializing LlamaCppReranker instance"); }

LlamaCppReranker::~LlamaCppReranker() {
    GENIEX_LOG_INFO("Destroying LlamaCppReranker instance");
    GENIEX_LOG_INFO("Starting cleanup of LlamaCppReranker resources");

    // Clean up llama context and model
    if (ctx_) {
        GENIEX_LOG_INFO("Freeing llama context");
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        GENIEX_LOG_INFO("Freeing llama model");
        llama_model_free(model_);
        model_ = nullptr;
    }

    GENIEX_LOG_INFO("Cleanup completed successfully");
}

int32_t LlamaCppReranker::create_impl(const geniex_RerankerCreateInput* input) {
    if (!input || !input->model_path) {
        GENIEX_LOG_ERROR("Invalid input parameters for reranker creation");
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    GENIEX_LOG_INFO("Creating reranker with model: {}", input->model_path);

    // Set up model parameters
    GENIEX_LOG_INFO("Setting up model parameters for reranker");
    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap           = true;
    mparams.use_mlock          = false;
    mparams.n_gpu_layers       = 999;  // Use GPU if available
    GENIEX_LOG_INFO("Model params: use_mmap={}, use_mlock={}, n_gpu_layers={}",
        mparams.use_mmap,
        mparams.use_mlock,
        mparams.n_gpu_layers);
    if (input->device_id) {
        auto device = ggml_backend_dev_by_name(input->device_id);
        if (!device) {
            // Device not found, log warning and continue with default device
            GENIEX_LOG_ERROR("Device '{}' not found", input->device_id);
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        } else {
            // Create a NULL-terminated array with the device
            static ggml_backend_dev_t device_array[2];
            device_array[0] = device;
            device_array[1] = nullptr;  // NULL-terminated
            mparams.devices = device_array;
        }
    }

    // Load the model
    GENIEX_LOG_INFO("Starting model load from file: {}", input->model_path);
    model_ = llama_model_load_from_file(input->model_path, mparams);
    if (!model_) {
        GENIEX_LOG_ERROR("Failed to load model from: {}", input->model_path);
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
    GENIEX_LOG_INFO("Model loaded successfully from: {}", input->model_path);

    // Get model info
    GENIEX_LOG_INFO("Retrieving model information");
    vocab_  = llama_model_get_vocab(model_);
    n_embd_ = llama_model_n_embd(model_);

    // Get number of classification outputs (for multi-label reranking)
    n_cls_out_ = llama_model_n_cls_out(model_);
    GENIEX_LOG_INFO("Model info: embedding dimension={}, classification outputs={}", n_embd_, n_cls_out_);

    // Check if model has rerank template
    rerank_template_ = llama_model_chat_template(model_, "rerank");
    if (rerank_template_) {
        GENIEX_LOG_INFO("Model has rerank template");
    } else {
        GENIEX_LOG_INFO("Model does not have rerank template, will use default formatting");
    }

    // Set up context parameters
    GENIEX_LOG_INFO("Setting up context parameters");
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = n_ctx_;
    cparams.n_batch              = n_batch_;
    cparams.n_ubatch             = n_ubatch_;
    cparams.n_threads            = std::thread::hardware_concurrency();
    cparams.n_threads_batch      = std::thread::hardware_concurrency();
    cparams.embeddings           = true;
    cparams.pooling_type         = LLAMA_POOLING_TYPE_RANK;  // Use RANK pooling for reranking
    cparams.kv_unified           = true;
    cparams.no_perf              = false;
    GENIEX_LOG_INFO("Context params: n_ctx={}, n_batch={}, n_ubatch={}, n_threads={}, pooling=RANK",
        cparams.n_ctx,
        cparams.n_batch,
        cparams.n_ubatch,
        cparams.n_threads);

    // Create context
    GENIEX_LOG_INFO("Creating llama context");
    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        GENIEX_LOG_ERROR("Failed to create llama context");
        llama_model_free(model_);
        model_ = nullptr;
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
    GENIEX_LOG_INFO("Llama context created successfully");

    GENIEX_LOG_INFO("Reranker created successfully");
    return GENIEX_SUCCESS;
}

int32_t LlamaCppReranker::rerank(const geniex_RerankerRerankInput* input, geniex_RerankerRerankOutput* output) {
    if (!input || !output) {
        GENIEX_LOG_ERROR("Invalid input parameters for reranking");
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    if (!input->query || !input->documents || input->documents_count <= 0) {
        GENIEX_LOG_ERROR("Query and documents must be provided");
        return GENIEX_ERROR_RERANK_INPUT;
    }

    if (!model_) {
        GENIEX_LOG_ERROR("Reranker not initialized");
        return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
    }

    common::Profiler profiler;
    profiler.prompt_start();

    GENIEX_LOG_INFO("Reranking {} documents against query", input->documents_count);

    // Get configuration
    int32_t batch_size = (input->config && input->config->batch_size > 0) ? input->config->batch_size : 32;
    batch_size         = std::min(batch_size, n_batch_);
    GENIEX_LOG_INFO("Reranking config: batch_size={}", batch_size);

    // Tokenize query-document pairs
    std::vector<std::vector<int32_t>> token_sequences;
    token_sequences.reserve(input->documents_count);

    // Get separator tokens
    const std::string added_sep_token =
        llama_vocab_get_add_sep(vocab_) ? llama_vocab_get_text(vocab_, llama_vocab_sep(vocab_)) : "";
    const std::string added_eos_token =
        llama_vocab_get_add_eos(vocab_) ? llama_vocab_get_text(vocab_, llama_vocab_eos(vocab_)) : "";

    for (int32_t i = 0; i < input->documents_count; ++i) {
        if (!input->documents[i]) {
            GENIEX_LOG_WARN("Skipping null document at index {}", i);
            continue;
        }

        std::string final_prompt;

        // Use rerank template if available
        if (rerank_template_) {
            final_prompt = rerank_template_;
            // Replace placeholders
            string_replace_all(final_prompt, "{query}", input->query);
            string_replace_all(final_prompt, "{document}", input->documents[i]);
        } else {
            // Default formatting: query + sep/eos + document
            final_prompt = input->query;
            if (!added_eos_token.empty()) {
                final_prompt += added_eos_token;
            }
            if (!added_sep_token.empty()) {
                final_prompt += added_sep_token;
            }
            final_prompt += input->documents[i];
        }

        // Tokenize the combined prompt
        std::vector<llama_token> tokens = common_tokenize(vocab_, final_prompt, true, true);

        if (tokens.empty() || static_cast<int>(tokens.size()) > n_batch_) {
            GENIEX_LOG_ERROR("Invalid token count {} for document {} (max batch: {})", tokens.size(), i, n_batch_);
            return GENIEX_ERROR_RERANK_FAILED;
        }

        token_sequences.emplace_back(tokens.begin(), tokens.end());
    }

    if (token_sequences.empty()) {
        GENIEX_LOG_ERROR("No valid documents to process");
        return GENIEX_ERROR_RERANK_FAILED;
    }

    const size_t n_sequences = token_sequences.size();

    // Allocate output memory for scores
    // Each sequence gets n_cls_out scores (usually 1 for simple reranking)
    const size_t score_count = n_sequences * std::max(1u, n_cls_out_);
    const size_t memory_size = score_count * sizeof(float);
    GENIEX_LOG_INFO("Allocating memory for scores: {} sequences x {} outputs = {} bytes",
        n_sequences,
        std::max(1u, n_cls_out_),
        memory_size);
    float* scores = static_cast<float*>(std::malloc(memory_size));
    if (!scores) {
        GENIEX_LOG_ERROR("Failed to allocate memory for scores: {} bytes", memory_size);
        return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
    }
    GENIEX_LOG_INFO("Memory allocation successful for scores");

    // Process in batches
    size_t  processed    = 0;
    int32_t total_tokens = 0;
    GENIEX_LOG_INFO("Starting batch processing for {} sequences with batch_size={}", n_sequences, batch_size);

    while (processed < n_sequences) {
        llama_batch batch = llama_batch_init(n_batch_, 0, batch_size);
        batch.n_tokens    = 0;

        int seq_in_batch = 0;
        int tok_in_batch = 0;

        // Fill batch with sequences
        for (size_t i = processed; i < n_sequences && seq_in_batch < batch_size; ++i) {
            const auto& tokens = token_sequences[i];
            if (tok_in_batch + static_cast<int>(tokens.size()) > n_batch_) {
                break;
            }
            batch_add_seq(batch, tokens, static_cast<llama_seq_id>(seq_in_batch));
            tok_in_batch += static_cast<int>(tokens.size());
            total_tokens += static_cast<int>(tokens.size());
            ++seq_in_batch;
        }

        // Process batch
        float* out_ptr = scores + processed * std::max(1u, n_cls_out_);
        if (process_batch(batch, out_ptr, seq_in_batch) != 0) {
            GENIEX_LOG_ERROR("Failed to process batch");
            llama_batch_free(batch);
            std::free(scores);
            return GENIEX_ERROR_RERANK_FAILED;
        }

        // Record TTFT on first batch processing
        if (processed == 0) {
            profiler.record_ttft();
        }

        llama_batch_free(batch);
        processed += seq_in_batch;
    }

    // Update profiling data
    profiler.prompt_end();
    profiler.update_prompt_tokens(total_tokens);
    profiler.update_generated_tokens(0);  // Not applicable for reranking
    profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_COMPLETED);

    // Set output
    output->scores      = scores;
    output->score_count = static_cast<int32_t>(score_count);
    profiler.to_profile_data(output->profile_data);

    GENIEX_LOG_INFO("Successfully generated {} scores for {} documents, total tokens processed: {}",
        score_count,
        n_sequences,
        total_tokens);
    return GENIEX_SUCCESS;
}

void LlamaCppReranker::batch_add_seq(llama_batch& batch, const std::vector<int32_t>& tokens, llama_seq_id seq_id) {
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (batch.n_tokens >= n_batch_) break;

        batch.token[batch.n_tokens]     = tokens[i];
        batch.pos[batch.n_tokens]       = static_cast<llama_pos>(i);
        batch.n_seq_id[batch.n_tokens]  = 1;
        batch.seq_id[batch.n_tokens][0] = seq_id;
        batch.logits[batch.n_tokens]    = 1;  // request output for this token

        ++batch.n_tokens;
    }
}

int LlamaCppReranker::process_batch(llama_batch& batch, float* output, int n_seq) {
    // Clear previous kv_cache values (irrelevant for reranking)
    llama_memory_clear(llama_get_memory(ctx_), true);

    if (llama_decode(ctx_, batch) != 0) {
        return -1;
    }

    // For reranking, we use sequence-level embeddings with RANK pooling
    const uint32_t n_outputs = std::max(1u, n_cls_out_);

    for (int s = 0; s < n_seq; ++s) {
        const float* emb = llama_get_embeddings_seq(ctx_, s);
        if (!emb) {
            GENIEX_LOG_ERROR("Failed to get embeddings for sequence {}", s);
            return -1;
        }

        // Copy all classification outputs for this sequence
        for (uint32_t c = 0; c < n_outputs; ++c) {
            output[s * n_outputs + c] = emb[c];
        }
    }

    return 0;
}

}  // namespace geniex
