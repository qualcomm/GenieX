#pragma once

#include <optional>
#include <vector>

#include "common.h"        // common_params_sampling
#include "geniex.h"        // geniex_ModelConfig
#include "ggml-backend.h"  // ggml_backend_dev_t
#include "ggml-cpu.h"      // ggml_threadpool_params
#include "llama.h"         // llama_model_params, llama_context_params

namespace geniex {

// Coarse compute target after device alias resolution (sdk/src/device.cpp).
// Drives per-(platform, device) defaults across all the build_* mappers
// below; mirrors test-llama.cpp's compute_configs.py ComputeUnit. HTP covers
// both the pinned `npu` alias and the `hybrid` per-tensor scheduler — they
// share the same upstream tuning.
enum class Device { CPU, GPU, HTP };

// Reverse-classify a resolved device selection. Mirrors the alias table in
// sdk/src/device.cpp:
//   cpu    -> n_gpu_layers == 0                     -> CPU
//   gpu    -> device_id starts with "GPU"           -> GPU
//   npu    -> device_id starts with "HTP"           -> HTP
//   hybrid -> empty device_id, ngl == 999           -> HTP
Device classify_device(const char* device_id, int n_gpu_layers);

// Resolve a caller's config against the plugin defaults: each unset (0) numeric
// field takes its default, and thread counts use the per-(platform, device)
// rule (offloaded inference pins the upstream-fixed count; pure CPU uses
// cores/2). Only n_ctx's default differs between LLM (4096) and VLM (16384).
// The returned config has every numeric field populated, so the build_*_params
// mappers below are pure field copies.
geniex_ModelConfig build_model_config(const geniex_ModelConfig& config, int32_t n_ctx_default, Device device);

// Map an already-resolved config to llama params. Device selection and
// tensor-buffer overrides stay at the call site.
llama_model_params   build_model_params(const geniex_ModelConfig& config);
llama_context_params build_context_params(const geniex_ModelConfig& config, Device device);

// Build threadpool tuning (cpumask / strict / poll) for a given thread count
// on the given target device. Returned struct is what ggml_threadpool_new
// accepts.
ggml_threadpool_params build_threadpool_params(int n_threads, Device device);

// Map a caller's sampler config onto common_params_sampling, each unset (0/0.0)
// field taking the plugin default. A null cfg yields pure defaults. Grammar
// comes from grammar_string when set, otherwise it is read from grammar_path.
common_params_sampling build_sampling_params(const geniex_SamplerConfig* cfg);

// Parse a comma-separated device-id list (e.g. "HTP0,HTP1"), resolving each name against the
// ggml registry. Returns the resolved devices followed by a trailing nullptr, suitable for
// llama_model_params::devices via .data(); keep the result alive until model load returns.
// A null/empty device_id yields an empty vector (caller uses the model default). Returns
// std::nullopt only when device_id names devices but none of them resolve.
std::optional<std::vector<ggml_backend_dev_t>> resolve_devices(const char* device_id);

}  // namespace geniex
