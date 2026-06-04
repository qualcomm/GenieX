#pragma once

#include "ggml-cpu.h"  // ggml_threadpool_params, ggml_threadpool_new/free

struct llama_context;

namespace geniex {

// Owns a main + optional batch threadpool created from the CPU backend and
// attached to a context. Both LlamaLlm and LlamaVlm need identical setup plus
// teardown, so the lifecycle lives here. Destruction order: the owning class
// frees its `ctx` (llama_free) in its destructor body before this member is
// destroyed, matching llama.cpp's free order.
struct Threadpools {
    ggml_threadpool* main_pool        = nullptr;
    ggml_threadpool* batch_pool       = nullptr;
    void (*free_fn)(ggml_threadpool*) = nullptr;

    Threadpools()                              = default;
    Threadpools(const Threadpools&)            = delete;
    Threadpools& operator=(const Threadpools&) = delete;
    ~Threadpools();
};

// Create the CPU-backend threadpools described by `main` / `batch` and attach
// them to `ctx`. When the two configs differ we materialise both pools and
// keep the main pool paused (matches llama.cpp main.cpp). When they match,
// only the main pool is created and serves both prefill and decode. All
// per-(platform, device) tuning has already been baked into the inputs by
// build_threadpool_params; this function is pure lifecycle.
int32_t create_and_attach_threadpools(
    Threadpools& pools, llama_context* ctx, ggml_threadpool_params main, ggml_threadpool_params batch);

}  // namespace geniex
