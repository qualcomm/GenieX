#include "threadpool.h"

#include <string>

#include "geniex.h"
#include "ggml-backend.h"
#include "llama.h"
#include "logging.h"

namespace geniex {

Threadpools::~Threadpools() {
    if (!free_fn_) return;
    if (main_pool_) free_fn_(main_pool_);
    if (batch_pool_) free_fn_(batch_pool_);
}

int32_t Threadpools::attach(llama_context* ctx, ggml_threadpool_params main, ggml_threadpool_params batch) {
    auto* cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        GENIEX_LOG_ERROR("No CPU backend found");
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }

    auto* reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto* threadpool_new =
        (decltype(ggml_threadpool_new)*)ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto* threadpool_free =
        (decltype(ggml_threadpool_free)*)ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
    if (!threadpool_new || !threadpool_free) {
        GENIEX_LOG_ERROR("Failed to get threadpool functions");
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
    this->free_fn_ = threadpool_free;

    if (batch.n_threads != main.n_threads) {
        this->batch_pool_ = threadpool_new(&batch);
        if (!this->batch_pool_) {
            GENIEX_LOG_ERROR("Batch threadpool create failed: n_threads {}", batch.n_threads);
            return GENIEX_ERROR_COMMON_MODEL_LOAD;
        }
        // Start the non-batch threadpool paused (matches llama.cpp main.cpp).
        main.paused = true;
    }

    this->main_pool_ = threadpool_new(&main);
    if (!this->main_pool_) {
        GENIEX_LOG_ERROR("Threadpool create failed: n_threads {}", main.n_threads);
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }

    llama_attach_threadpool(ctx, this->main_pool_, this->batch_pool_);

    std::string pinned;
    for (int i = 0; i < GGML_MAX_N_THREADS; ++i) {
        if (main.cpumask[i]) {
            if (!pinned.empty()) pinned += ",";
            pinned += std::to_string(i);
        }
    }
    GENIEX_LOG_INFO(
        "[Optimise] threadpool attached: n_threads={}, n_threads_batch={}, strict_cpu={}, poll={}, "
        "pinned_cores=[{}]",
        main.n_threads,
        batch.n_threads,
        main.strict_cpu,
        main.poll,
        pinned.empty() ? "none" : pinned);
    return GENIEX_SUCCESS;
}

}  // namespace geniex
