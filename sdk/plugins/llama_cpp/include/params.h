#pragma once

#include <optional>
#include <vector>

#include "geniex.h"        // geniex_ModelConfig
#include "ggml-backend.h"  // ggml_backend_dev_t
#include "llama.h"         // llama_model_params, llama_context_params

namespace geniex {

// Resolve a caller's config against the plugin defaults: each unset (0) numeric
// field takes its default, and thread counts use the offload-aware rule
// (offloaded inference pins the upstream-fixed count; pure CPU uses cores/2).
// Only n_ctx's default differs between LLM (4096) and VLM (16384). The returned
// config has every numeric field populated, so the build_*_params mappers below
// are pure field copies.
geniex_ModelConfig build_model_config(const geniex_ModelConfig& config, int32_t n_ctx_default);

// Map an already-resolved config to llama params. Device selection and
// tensor-buffer overrides stay at the call site.
llama_model_params   build_model_params(const geniex_ModelConfig& config);
llama_context_params build_context_params(const geniex_ModelConfig& config);

// Parse a comma-separated device-id list (e.g. "HTP0,HTP1"), resolving each name against the
// ggml registry. Returns the resolved devices followed by a trailing nullptr, suitable for
// llama_model_params::devices via .data(); keep the result alive until model load returns.
// A null/empty device_id yields an empty vector (caller uses the model default). Returns
// std::nullopt only when device_id names devices but none of them resolve.
std::optional<std::vector<ggml_backend_dev_t>> resolve_devices(const char* device_id);

}  // namespace geniex
