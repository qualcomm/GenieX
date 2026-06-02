#pragma once

#include "ggml-cpu.h"  // ggml_threadpool_params, ggml_threadpool_new/free

struct llama_context;

namespace geniex {

// CPU threadpool tuning for the llama.cpp plugin, mirroring upstream's
// Snapdragon test scripts. Both the Android adb wrappers
// (llama.cpp/scripts/snapdragon/adb/run-{bench,completion}.sh) and the Linux
// IoT / QDC path (compute_configs.py, "single source of truth for
// per-compute-unit flags across all test types") run offloaded inference with
// the SAME hardcoded flags — `-t 6 --cpu-mask 0xfc --cpu-strict 1 --poll 1000`
// — identically on mobile (SM8750/8650/8850) and desktop X Elite. There is no
// per-platform split upstream, so we don't add one either.

// Resolve the effective thread count. An explicit caller value (`requested`
// > 0, from geniex_ModelConfig.n_threads) always wins. Otherwise offloaded
// inference uses the upstream-fixed count (6); pure CPU uses `cpu_default`
// (the model_config default, typically hardware_concurrency()/2).
int resolve_n_threads(int requested, bool offloading, int cpu_default);

// Build threadpool params for `n_threads`. When `offloading` and the host has
// enough cores, worker threads are pinned to performance cores
// [2, 2 + n_threads) — reserving cores 0/1 for the OS and the HTP FastRPC
// busy-wait thread — with strict placement and aggressive polling, matching
// the upstream `--cpu-mask 0xfc --cpu-strict 1 --poll 1000` (cores 2-7 at the
// default 6 threads). Pure-CPU inference and core-starved hosts keep
// llama.cpp's defaults (shared affinity, poll=50).
ggml_threadpool_params optimize_threadpool_params(int n_threads, bool offloading);

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

// Create platform-tuned threadpools and attach them to `ctx`. `n_threads` /
// `n_threads_batch` are the already-resolved counts (see resolve_n_threads).
// Returns a GENIEX_ERROR_* code on failure, GENIEX_SUCCESS otherwise.
int32_t create_and_attach_threadpools(
    Threadpools& pools, llama_context* ctx, int n_threads, int n_threads_batch, bool offloading);

}  // namespace geniex
