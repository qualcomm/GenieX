#include "threadpool.h"

#include <cstdint>
#include <string>
#include <thread>

#include "geniex.h"
#include "ggml-backend.h"
#include "llama.h"
#include "logging.h"

namespace geniex {

namespace {
constexpr int      kReservedCores  = 2;     // cores 0/1: OS + HTP FastRPC busy-wait thread
constexpr int      kOffloadThreads = 6;     // upstream-fixed `-t 6` (cores 2-7 via 0xfc)
constexpr uint32_t kOffloadPoll    = 1000;  // aggressive polling; matches upstream `--poll 1000`
}  // namespace

int resolve_n_threads(int requested, bool offloading, int cpu_default) {
    if (requested > 0) return requested;
    return offloading ? kOffloadThreads : cpu_default;
}

ggml_threadpool_params optimize_threadpool_params(int n_threads, bool offloading) {
    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    if (!offloading || n_threads < 1) {
        return tpp;
    }

    // Pin only if every worker gets its own core after reserving 0/1; else
    // oversubscription would fight the accelerator cadence, so keep defaults.
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw < static_cast<unsigned>(n_threads) + kReservedCores) {
        return tpp;
    }

    for (int i = 0; i < n_threads; ++i) {
        tpp.cpumask[kReservedCores + i] = true;
    }
    tpp.strict_cpu = true;
    tpp.poll       = kOffloadPoll;
    return tpp;
}

Threadpools::~Threadpools() {
    if (!free_fn) return;
    if (main_pool) free_fn(main_pool);
    if (batch_pool) free_fn(batch_pool);
}

int32_t create_and_attach_threadpools(
    Threadpools& pools, llama_context* ctx, int n_threads, int n_threads_batch, bool offloading) {
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

    ggml_threadpool_params tpp = optimize_threadpool_params(n_threads, offloading);

    // Create batch threadpool only when it differs from the main one.
    if (n_threads_batch != n_threads) {
        ggml_threadpool_params tpp_batch = optimize_threadpool_params(n_threads_batch, offloading);
        pools.batch_pool                 = threadpool_new(&tpp_batch);
        if (!pools.batch_pool) {
            GENIEX_LOG_ERROR("Batch threadpool create failed: n_threads {}", tpp_batch.n_threads);
            return GENIEX_ERROR_COMMON_MODEL_LOAD;
        }
        // Start the non-batch threadpool paused (matches llama.cpp main.cpp).
        tpp.paused = true;
    }

    pools.main_pool = threadpool_new(&tpp);
    if (!pools.main_pool) {
        GENIEX_LOG_ERROR("Threadpool create failed: n_threads {}", tpp.n_threads);
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }

    llama_attach_threadpool(ctx, pools.main_pool, pools.batch_pool);

    std::string pinned;
    for (int i = 0; i < GGML_MAX_N_THREADS; ++i) {
        if (tpp.cpumask[i]) {
            if (!pinned.empty()) pinned += ",";
            pinned += std::to_string(i);
        }
    }
    GENIEX_LOG_INFO(
        "[Optimise] threadpool attached: n_threads={}, n_threads_batch={}, offloading={}, strict_cpu={}, poll={}, "
        "pinned_cores=[{}]",
        n_threads,
        n_threads_batch,
        offloading,
        tpp.strict_cpu,
        tpp.poll,
        pinned.empty() ? "none" : pinned);
    return GENIEX_SUCCESS;
}

}  // namespace geniex
