// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

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
enum class Device { CPU, GPU, NPU };

std::optional<std::vector<ggml_backend_dev_t>> resolve_devices(const char* device_id);

// Reverse-classify a resolved device selection. Mirrors the alias table in
// sdk/src/device.cpp:
//   cpu    -> n_gpu_layers == 0                     -> CPU
//   gpu    -> device_id starts with "GPU"           -> GPU
//   npu    -> device_id starts with "HTP"           -> HTP
//   hybrid -> empty device_id, ngl != 0             -> HTP
Device classify_device(const char* device_id, int n_gpu_layers);

// Map a caller's config to llama params, filling each unset (0) field from the
// plugin defaults. build_context_params resolves n_ctx (default differs between
// LLM at 4096 and VLM at 16384) plus n_batch / n_ubatch / n_seq_max, and picks
// thread counts via the per-(platform, device) rule (offloaded inference pins
// the upstream-fixed count; pure CPU uses cores/2). build_model_params only
// reads n_gpu_layers. Device selection and tensor-buffer overrides stay at the
// call site.
//
// spec is the parsed speculative config (nullptr when disabled, and always
// nullptr for the draft context itself): a target context that drafts needs
// n_max recurrent-state snapshots to roll back rejected drafts and one logits
// row per drafted token. Mirrors common_context_params_to_llama +
// server_output_limits.
llama_model_params   build_model_params(const geniex_ModelConfig& config, Device device);
llama_context_params build_context_params(const geniex_ModelConfig& config, int32_t n_ctx_default, Device device,
    const common_params_speculative* spec = nullptr);

// Parse spec_type / spec_n_* into llama.cpp's own speculative params. Returns
// nullopt when speculative decoding is disabled or no type name resolves.
std::optional<common_params_speculative> build_speculative_params(const geniex_ModelConfig& config);

// Build threadpool tuning (cpumask / strict / poll) for a given thread count
// on the given target device. Returned struct is what ggml_threadpool_new
// accepts.
ggml_threadpool_params build_threadpool_params(int n_threads, Device device);
common_params_sampling build_sampling_params(const geniex_SamplerConfig* cfg);

}  // namespace geniex
