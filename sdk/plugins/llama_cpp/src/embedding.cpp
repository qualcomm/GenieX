#include "embedding.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "htp_session.h"
#include "logging.h"
#include "profiler.h"

namespace geniex {

LlamaCppEmbedding::LlamaCppEmbedding() : state_(std::make_unique<EmbeddingState>()) {
    GENIEX_LOG_INFO("Initializing LlamaCppEmbedding instance");
}

LlamaCppEmbedding::~LlamaCppEmbedding() {
    GENIEX_LOG_INFO("Destroying LlamaCppEmbedding instance");
    if (!state_) {
        GENIEX_LOG_INFO("Cleanup called but state is null, nothing to clean");
        return;
    }

    GENIEX_LOG_INFO("Starting cleanup of LlamaCppEmbedding resources");

    // Clean up llama context and model
    if (state_->ctx) {
        GENIEX_LOG_INFO("Freeing llama context");
        llama_free(state_->ctx);
        state_->ctx = nullptr;
    }
    if (state_->model) {
        GENIEX_LOG_INFO("Freeing llama model");
        llama_model_free(state_->model);
        state_->model = nullptr;
    }

    state_->initialized = false;
    GENIEX_LOG_INFO("Cleanup completed successfully");
}

int32_t LlamaCppEmbedding::create_impl(const geniex_EmbedderCreateInput* input) {
    if (!input || !input->model_path) {
        GENIEX_LOG_ERROR("Invalid input parameters for embedder creation");
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    GENIEX_LOG_INFO("Creating embedder with model: {}", input->model_path);

    // See llm.cpp for the rationale behind the HTP session release/reacquire
    // dance. Any llama.cpp class that might load onto HTP must participate.
    htp::reacquire_before_load();

    // Set up model parameters
    GENIEX_LOG_INFO("Setting up model parameters for embedder");
    llama_model_params mparams         = llama_model_default_params();
    ggml_backend_dev_t device_array[2] = {nullptr, nullptr};
    mparams.use_mmap                   = true;
    mparams.use_mlock                  = false;
    mparams.n_gpu_layers               = 999;  // Use GPU if available
                                               //
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
            device_array[0] = device;
            mparams.devices = device_array;
        }
    }

    if (htp::devices_include_htp(device_array)) {
        htp_guard_.mark_htp();
    } else if (mparams.n_gpu_layers != 0 && htp::htp_backend_present()) {
        htp_guard_.mark_htp();
    }

    // Load the model
    GENIEX_LOG_INFO("Starting model load from file: {}", input->model_path);
    state_->model = llama_model_load_from_file(input->model_path, mparams);
    if (!state_->model) {
        GENIEX_LOG_ERROR("Failed to load model from: {}", input->model_path);
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
    GENIEX_LOG_INFO("Model loaded successfully from: {}", input->model_path);

    // Get model info
    GENIEX_LOG_INFO("Retrieving model information");
    state_->vocab  = llama_model_get_vocab(state_->model);
    state_->n_embd = llama_model_n_embd(state_->model);
    GENIEX_LOG_INFO("Model info: embedding dimension={}", state_->n_embd);

    // Set up context parameters
    GENIEX_LOG_INFO("Setting up context parameters");
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = state_->n_ctx;
    cparams.n_batch              = state_->n_batch;
    cparams.n_ubatch             = state_->n_ubatch;
    cparams.n_threads            = std::thread::hardware_concurrency();
    cparams.n_threads_batch      = std::thread::hardware_concurrency();
    cparams.embeddings           = true;
    cparams.pooling_type         = LLAMA_POOLING_TYPE_MEAN;
    cparams.kv_unified           = true;
    cparams.no_perf              = false;
    GENIEX_LOG_INFO("Context params: n_ctx={}, n_batch={}, n_ubatch={}, n_threads={}, embeddings={}",
        cparams.n_ctx,
        cparams.n_batch,
        cparams.n_ubatch,
        cparams.n_threads,
        cparams.embeddings);

    // Create context
    GENIEX_LOG_INFO("Creating llama context");
    state_->ctx = llama_init_from_model(state_->model, cparams);
    if (!state_->ctx) {
        GENIEX_LOG_ERROR("Failed to create llama context");
        llama_model_free(state_->model);
        state_->model = nullptr;
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
    GENIEX_LOG_INFO("Llama context created successfully");

    state_->initialized = true;

    GENIEX_LOG_INFO("Embedder created successfully with dimension: {}", state_->n_embd);
    return GENIEX_SUCCESS;
}

int32_t LlamaCppEmbedding::embed(const geniex_EmbedderEmbedInput* input, geniex_EmbedderEmbedOutput* output) {
    if (!input || !output) {
        GENIEX_LOG_ERROR("Invalid input parameters for embedding");
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    // This backend only supports text embeddings, reject image inputs
    const bool has_image = input->image_paths && input->image_count > 0;
    if (has_image) {
        GENIEX_LOG_ERROR("Image embedding requested, but llama_cpp backend only supports text embeddings");
        return GENIEX_ERROR_COMMON_NOT_SUPPORTED;
    }

    // Validate that either texts or input_ids_2d is provided
    bool has_texts     = input->texts && input->text_count > 0;
    bool has_input_ids = input->input_ids_2d && input->input_ids_row_lengths && input->input_ids_row_count > 0;

    if (!has_texts && !has_input_ids) {
        GENIEX_LOG_ERROR("Either texts or input_ids_2d must be provided");
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    if (!state_ || !state_->initialized) {
        GENIEX_LOG_ERROR("Embedder not initialized");
        return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
    }

    common::Profiler profiler;
    profiler.prompt_start();

    GENIEX_LOG_INFO("Generating embeddings for {} texts", input->text_count);

    // Get configuration
    int32_t     batch_size   = (input->config && input->config->batch_size > 0) ? input->config->batch_size : 32;
    bool        do_normalize = input->config ? input->config->normalize : true;
    const char* norm_method =
        (input->config && input->config->normalize_method) ? input->config->normalize_method : "l2";
    batch_size = std::min(batch_size, state_->n_batch);
    GENIEX_LOG_INFO(
        "Embedding config: batch_size={}, do_normalize={}, norm_method={}", batch_size, do_normalize, norm_method);

    // Handle tokenization: use input_ids_2d if provided, otherwise tokenize texts
    std::vector<std::vector<int32_t>> token_sequences;

    if (input->input_ids_2d && input->input_ids_row_lengths && input->input_ids_row_count > 0) {
        // Use pre-tokenized input_ids_2d (ignore texts field)
        GENIEX_LOG_INFO("Using pre-tokenized input_ids_2d with {} sequences", input->input_ids_row_count);

        token_sequences.reserve(input->input_ids_row_count);

        // Process each row in the 2D array
        for (int32_t i = 0; i < input->input_ids_row_count; ++i) {
            const int32_t row_length = input->input_ids_row_lengths[i];

            if (row_length <= 0) {
                GENIEX_LOG_DEBUG("Skipping empty token sequence at index {}", i);
                continue;
            }

            if (row_length > state_->n_batch) {
                GENIEX_LOG_ERROR(
                    "Token sequence {} length {} exceeds max batch size {}", i, row_length, state_->n_batch);
                return GENIEX_ERROR_EMBEDDING_GENERATION;
            }

            const int32_t* row_data = input->input_ids_2d[i];
            if (!row_data) {
                GENIEX_LOG_DEBUG("Skipping null token sequence at index {}", i);
                continue;
            }

            // Convert row to token sequence
            std::vector<int32_t> tokens(row_data, row_data + row_length);
            token_sequences.push_back(std::move(tokens));
        }
    } else {
        // Tokenize texts as before
        GENIEX_LOG_INFO("Tokenizing {} input texts", input->text_count);
        token_sequences.reserve(input->text_count);

        for (int32_t i = 0; i < input->text_count; ++i) {
            if (!input->texts[i]) continue;

            const char* text     = input->texts[i];
            int         n_tokens = -llama_tokenize(state_->vocab, text, std::strlen(text), nullptr, 0, true, true);

            if (n_tokens <= 0 || n_tokens > state_->n_batch) {
                GENIEX_LOG_ERROR("Invalid token count {} for text {} (max batch: {})", n_tokens, i, state_->n_batch);
                return GENIEX_ERROR_EMBEDDING_GENERATION;
            }

            std::vector<llama_token> tokens(n_tokens);
            llama_tokenize(state_->vocab, text, std::strlen(text), tokens.data(), tokens.size(), true, true);
            token_sequences.emplace_back(tokens.begin(), tokens.end());
        }
    }

    if (token_sequences.empty()) {
        GENIEX_LOG_ERROR("No valid texts to process");
        return GENIEX_ERROR_EMBEDDING_GENERATION;
    }

    const size_t n_sequences = token_sequences.size();

    // Allocate output memory
    const size_t memory_size = n_sequences * state_->n_embd * sizeof(float);
    GENIEX_LOG_INFO("Allocating memory for embeddings: {} sequences x {} dimensions = {} bytes",
        n_sequences,
        state_->n_embd,
        memory_size);
    float* embeddings = static_cast<float*>(std::malloc(memory_size));
    if (!embeddings) {
        GENIEX_LOG_ERROR("Failed to allocate memory for embeddings: {} bytes", memory_size);
        return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
    }
    GENIEX_LOG_INFO("Memory allocation successful for embeddings");

    // Process in batches
    size_t  processed    = 0;
    int32_t total_tokens = 0;
    GENIEX_LOG_INFO("Starting batch processing for {} sequences with batch_size={}", n_sequences, batch_size);

    while (processed < n_sequences) {
        llama_batch batch = llama_batch_init(state_->n_batch, 0, batch_size);
        batch.n_tokens    = 0;

        int seq_in_batch = 0;
        int tok_in_batch = 0;

        // Fill batch with sequences
        for (size_t i = processed; i < n_sequences && seq_in_batch < batch_size; ++i) {
            const auto& tokens = token_sequences[i];
            if (tok_in_batch + static_cast<int>(tokens.size()) > state_->n_batch) {
                break;
            }
            batch_add_seq(batch, tokens, static_cast<llama_seq_id>(seq_in_batch), state_->n_batch);
            tok_in_batch += static_cast<int>(tokens.size());
            total_tokens += static_cast<int>(tokens.size());
            ++seq_in_batch;
        }

        // Process batch
        float* out_ptr = embeddings + processed * state_->n_embd;
        if (process_batch(batch, out_ptr, seq_in_batch, state_->n_embd, do_normalize ? norm_method : "none") != 0) {
            GENIEX_LOG_ERROR("Failed to process batch");
            llama_batch_free(batch);
            std::free(embeddings);
            return GENIEX_ERROR_EMBEDDING_GENERATION;
        }
        // Record TTFT on first batch processing (for embedding, this is the first batch completion)
        if (processed == 0) {
            profiler.record_ttft();
        }

        llama_batch_free(batch);
        processed += seq_in_batch;
    }

    // Update profiling data
    profiler.prompt_end();
    profiler.update_prompt_tokens(total_tokens);
    profiler.update_generated_tokens(0);  // Not applicable for embeddings
    profiler.set_stop_reason(common::StopReason::GENIEX_STOP_REASON_COMPLETED);

    // Set output
    output->embeddings      = embeddings;
    output->embedding_count = static_cast<int32_t>(n_sequences);
    profiler.to_profile_data(output->profile_data);

    GENIEX_LOG_INFO("Successfully generated {} embeddings with dimension {}, total tokens processed: {}",
        n_sequences,
        state_->n_embd,
        total_tokens);
    return GENIEX_SUCCESS;
}

int32_t LlamaCppEmbedding::embedding_dim(geniex_EmbedderDimOutput* output) {
    if (!output) {
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    if (!state_ || !state_->initialized) {
        return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
    }

    GENIEX_LOG_TRACE("embedder embedding dim: {}", state_->n_embd);
    output->dimension = state_->n_embd;
    return GENIEX_SUCCESS;
}

void LlamaCppEmbedding::normalize_embeddings(const float* inp, float* out, int n, const char* method) {
    if (!method || std::strcmp(method, "none") == 0) {
        std::memcpy(out, inp, n * sizeof(float));
        return;
    }

    double sum = 0.0;
    if (std::strcmp(method, "l2") == 0) {
        // L2 norm - euclidean norm
        for (int i = 0; i < n; ++i) {
            sum += inp[i] * inp[i];
        }
        sum = std::sqrt(sum);
    } else if (std::strcmp(method, "l1") == 0) {
        // L1 norm - sum of absolute values
        for (int i = 0; i < n; ++i) {
            sum += std::abs(inp[i]);
        }
    } else if (std::strcmp(method, "mean") == 0) {
        // Mean normalization - average absolute value
        for (int i = 0; i < n; ++i) {
            sum += std::abs(inp[i]);
        }
        sum /= n;
    } else {  // fallback - L2
        for (int i = 0; i < n; ++i) {
            sum += inp[i] * inp[i];
        }
        sum = std::sqrt(sum);
    }

    const float norm = sum > 0.0 ? 1.0f / static_cast<float>(sum) : 0.0f;
    for (int i = 0; i < n; ++i) {
        out[i] = inp[i] * norm;
    }
}

void LlamaCppEmbedding::batch_add_seq(
    llama_batch& batch, const std::vector<int32_t>& tokens, llama_seq_id seq_id, int32_t capacity) {
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (batch.n_tokens >= capacity) break;

        batch.token[batch.n_tokens]     = tokens[i];
        batch.pos[batch.n_tokens]       = static_cast<llama_pos>(i);
        batch.n_seq_id[batch.n_tokens]  = 1;
        batch.seq_id[batch.n_tokens][0] = seq_id;
        batch.logits[batch.n_tokens]    = 1;  // request output for this token

        ++batch.n_tokens;
    }
}

int LlamaCppEmbedding::process_batch(
    llama_batch& batch, float* output, int n_seq, int n_embd, const char* norm_method) {
    // Clear previous kv_cache values (irrelevant for embeddings)
    llama_memory_clear(llama_get_memory(state_->ctx), true);

    if (llama_decode(state_->ctx, batch) != 0) {
        return -1;
    }

    const auto pooling = llama_pooling_type(state_->ctx);

    if (pooling == LLAMA_POOLING_TYPE_NONE) {
        // Token-level embeddings – write sequentially
        int out_idx = 0;
        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i]) continue;
            const float* emb = llama_get_embeddings_ith(state_->ctx, i);
            if (!emb) return -1;
            normalize_embeddings(emb, output + out_idx * n_embd, n_embd, norm_method);
            ++out_idx;
        }
    } else {
        // Sequence-level (recommended – ctx set to MEAN)
        for (int s = 0; s < n_seq; ++s) {
            const float* emb = llama_get_embeddings_seq(state_->ctx, s);
            if (!emb) return -1;
            normalize_embeddings(emb, output + s * n_embd, n_embd, norm_method);
        }
    }
    return 0;
}

}  // namespace geniex
