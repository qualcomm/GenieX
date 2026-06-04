#include "threadpool.h"

#include <string>

#include "geniex.h"
#include "ggml-backend.h"
#include "llama.h"
#include "logging.h"

namespace geniex {

Threadpools::~Threadpools() {
    if (!free_fn) return;
    if (main_pool) free_fn(main_pool);
    if (batch_pool) free_fn(batch_pool);
}

int32_t create_and_attach_threadpools(
    Threadpools& pools, llama_context* ctx, ggml_threadpool_params main, ggml_threadpool_params batch) {
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
    pools.free_fn = threadpool_free;

    // Two pools only when the batch tuning genuinely differs from the main
    // tuning; otherwise main serves both prefill and decode.
    if (batch.n_threads != main.n_threads) {
        pools.batch_pool = threadpool_new(&batch);
        if (!pools.batch_pool) {
            GENIEX_LOG_ERROR("Batch threadpool create failed: n_threads {}", batch.n_threads);
            return GENIEX_ERROR_COMMON_MODEL_LOAD;
        }
        // Start the non-batch threadpool paused (matches llama.cpp main.cpp).
        main.paused = true;
    }

    pools.main_pool = threadpool_new(&main);
    if (!pools.main_pool) {
        GENIEX_LOG_ERROR("Threadpool create failed: n_threads {}", main.n_threads);
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }

    llama_attach_threadpool(ctx, pools.main_pool, pools.batch_pool);

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
