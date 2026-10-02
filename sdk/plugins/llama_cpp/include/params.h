// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <optional>
#include <vector>

#include "chat.h"          // common_chat_msg
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

// Pick the mmproj (vision/audio encoder) device given the LM's resolved
// device list. When the LM runs on HTP, sharing its session with the mmproj
// hits a fastrpc_mmap/session-contention failure on some HTP targets (e.g.
// IQ9/QCS9075) as soon as the encoder allocates its own compute buffer, even
// though neither side is short on memory alone. Prefer a distinct HTP
// session (GGML_HEXAGON_DEVICES must expose more than one) for the mmproj;
// fall back to sharing the LM's device when no other HTP session exists.
ggml_backend_dev_t resolve_vision_device(const std::vector<ggml_backend_dev_t>& lm_devices);

// Reverse-classify a resolved device selection. Mirrors the alias table in
// sdk/src/device.cpp:
//   cpu    -> n_gpu_layers == 0                     -> CPU
//   gpu    -> device_id starts with "GPU"           -> GPU
//   npu    -> device_id starts with "HTP"           -> HTP
//   hybrid -> empty device_id, ngl != 0             -> HTP
Device classify_device(const char* device_id, int n_gpu_layers);

// Map a caller's config to llama params, filling each unset (0) field from the
// plugin defaults. build_context_params resolves n_ctx (default differs between
// LLM at 4096 and VLM at 16384) plus n_batch / n_ubatch / n_seq_max, and defaults
// n_threads / n_threads_batch to common_cpu_get_num_math() regardless of device —
// matching upstream llama-bench/llama-cli, which never vary thread count by
// backend. Callers may still override via ModelConfig's n_threads/n_threads_batch.
// build_model_params only reads n_gpu_layers. Device selection and tensor-buffer
// overrides stay at the call site.
//
// spec is the parsed speculative config (nullptr when disabled, and always
// nullptr for the draft context itself): a target context that drafts needs
// n_max recurrent-state snapshots to roll back rejected drafts and one logits
// row per drafted token. Mirrors common_context_params_to_llama +
// server_output_limits.
llama_model_params   build_model_params(const geniex_ModelConfig& config, Device device);
llama_context_params build_context_params(const geniex_ModelConfig& config, int32_t n_ctx_default, Device device,
    const common_params_speculative* spec = nullptr);

// Like build_context_params + llama_init_from_model, but on NPU halves n_ubatch and retries when
// init fails, down to a floor of 128. HTP's fastrpc/CMA pool can fail to satisfy a given ubatch's
// compute buffer as one contiguous block depending on what else is resident (model size, KV cache,
// other sessions), so no single static n_ubatch default is safe for every model/device combo. Skips
// the ladder (single attempt only) when the caller pinned n_ubatch explicitly via config, or on
// CPU/GPU where this failure mode doesn't apply. On success, *out_cpar reflects the n_ubatch that
// actually worked. See qcom-ai-hub/geniex#1683.
llama_context* init_context_with_ubatch_ladder(llama_model* model, const geniex_ModelConfig& config,
    int32_t n_ctx_default, Device device, const common_params_speculative* spec, llama_context_params* out_cpar);

// Parse spec_type / spec_n_* into llama.cpp's own speculative params. Returns
// nullopt when speculative decoding is disabled or no type name resolves.
std::optional<common_params_speculative> build_speculative_params(const geniex_ModelConfig& config);

// Build threadpool tuning (cpumask / strict / poll) for a given thread count
// on the given target device. Returned struct is what ggml_threadpool_new
// accepts.
ggml_threadpool_params build_threadpool_params(int n_threads, Device device);
common_params_sampling build_sampling_params(const geniex_SamplerConfig* cfg);

// Copy the FFI tool-calling fields of a chat message onto a common_chat_msg.
// Templates need tool_calls structurally, not as assistant text: most render a
// "tool" message only from the tool_calls of the assistant turn before it
// (matched by id), so dropping them drops the tool result from the prompt.
void apply_tool_fields(common_chat_msg& msg, const geniex_ToolCall* tool_calls, int32_t tool_call_count,
    const char* tool_call_id, const char* tool_name);

}  // namespace geniex
