#pragma once

#include "ggml-cpu.h"  // ggml_threadpool_params, ggml_threadpool_new/free

struct llama_context;

namespace geniex {

// Owns a main + optional batch threadpool attached to a context, shared by
// LlamaLlm and LlamaVlm. The owner must free its `ctx` before this member is
// destroyed, matching llama.cpp's free order.
class Threadpools {
   public:
    Threadpools()                              = default;
    Threadpools(const Threadpools&)            = delete;
    Threadpools& operator=(const Threadpools&) = delete;
    ~Threadpools();

    // Create the CPU-backend threadpools and attach them to `ctx`. A separate
    // batch pool (with main paused, as in llama.cpp main.cpp) is created only
    // when batch tuning differs from main; otherwise main serves both.
    int32_t attach(llama_context* ctx, ggml_threadpool_params main, ggml_threadpool_params batch);

   private:
    ggml_threadpool* main_pool_        = nullptr;
    ggml_threadpool* batch_pool_       = nullptr;
    void (*free_fn_)(ggml_threadpool*) = nullptr;
};

}  // namespace geniex
