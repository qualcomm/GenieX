// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Translates the public C `geniex_SamplerConfig` into qairt-core
// `geniex::GenerationConfig`. Shared by the qairt LLM and VLM plugins.

#pragma once

#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>

#include "geniex.h"
#include "llm/llm_spec_loader.h"  // ParsedSamplerConfig
#include "logging.h"
#include "types.h"  // geniex::GenerationConfig

namespace geniex::qairt {

// Plugin-level fallback when neither the user nor the bundle's `dialog.sampler`
// block specifies a value.
struct DefaultSamplerParams {
    uint32_t seed               = 42;
    int32_t  top_k              = 40;
    float    top_p              = 0.95f;
    float    min_p              = 0.0f;
    float    temperature        = 0.8f;
    float    repetition_penalty = 1.0f;
    float    presence_penalty   = 0.0f;
    float    frequency_penalty  = 0.0f;
};

inline void apply_sampler_config(
    const geniex_SamplerConfig* cfg, GenerationConfig& out, const ParsedSamplerConfig& bundle = {}) {
    constexpr DefaultSamplerParams kDefaults{};

    auto pick_uint = [](uint32_t user_val, const std::optional<uint32_t>& bundle_val, uint32_t fallback) {
        if (user_val != 0) return user_val;
        if (bundle_val) return *bundle_val;
        return fallback;
    };
    auto pick_int = [](int32_t user_val, const std::optional<int32_t>& bundle_val, int32_t fallback) {
        if (user_val != 0) return user_val;
        if (bundle_val) return *bundle_val;
        return fallback;
    };
    auto pick_float = [](float user_val, const std::optional<float>& bundle_val, float fallback) {
        if (user_val != 0.0f) return user_val;
        if (bundle_val) return *bundle_val;
        return fallback;
    };

    out.enable_sampling = true;

    if (!cfg) {
        // No user config → bundle wins where present, otherwise plugin defaults.
        out.seed               = bundle.seed.value_or(kDefaults.seed);
        out.top_k              = bundle.top_k.value_or(kDefaults.top_k);
        out.top_p              = bundle.top_p.value_or(kDefaults.top_p);
        out.min_p              = kDefaults.min_p;  // genie has no min_p; bundle never carries it
        out.temperature        = bundle.temperature.value_or(kDefaults.temperature);
        out.repetition_penalty = bundle.repetition_penalty.value_or(kDefaults.repetition_penalty);
        out.presence_penalty   = bundle.presence_penalty.value_or(kDefaults.presence_penalty);
        out.frequency_penalty  = bundle.frequency_penalty.value_or(kDefaults.frequency_penalty);
        if (bundle.penalty_last_n) out.penalty_last_n = *bundle.penalty_last_n;
        return;
    }

    // Zero-sentinel on cfg: 0 means "defer to lower tiers".
    out.seed  = pick_uint(static_cast<uint32_t>(cfg->seed), bundle.seed, kDefaults.seed);
    out.top_k = pick_int(cfg->top_k, bundle.top_k, kDefaults.top_k);
    out.top_p = pick_float(cfg->top_p, bundle.top_p, kDefaults.top_p);
    out.min_p = (cfg->min_p != 0.0f) ? cfg->min_p : kDefaults.min_p;
    out.repetition_penalty =
        pick_float(cfg->repetition_penalty, bundle.repetition_penalty, kDefaults.repetition_penalty);
    out.presence_penalty  = pick_float(cfg->presence_penalty, bundle.presence_penalty, kDefaults.presence_penalty);
    out.frequency_penalty = pick_float(cfg->frequency_penalty, bundle.frequency_penalty, kDefaults.frequency_penalty);
    if (bundle.penalty_last_n) out.penalty_last_n = *bundle.penalty_last_n;

    // Temperature is special: 0 means "defer", a NEGATIVE value is the
    // greedy/argmax sentinel, positive is the literal temp.
    out.temperature =
        (cfg->temperature == 0.0f) ? bundle.temperature.value_or(kDefaults.temperature) : cfg->temperature;

    // Grammar — string takes priority over path (matches llama_cpp plugin).
    if (cfg->grammar_string && cfg->grammar_string[0] != '\0') {
        out.grammar_str = cfg->grammar_string;
        GENIEX_LOG_DEBUG("Applied grammar string ({} bytes)", out.grammar_str.size());
    } else if (cfg->grammar_path && cfg->grammar_path[0] != '\0') {
        std::ifstream file(cfg->grammar_path);
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            out.grammar_str = buffer.str();
            GENIEX_LOG_DEBUG("Applied grammar from file: {}", cfg->grammar_path);
        } else {
            GENIEX_LOG_ERROR("Failed to read grammar file: {}", cfg->grammar_path);
        }
    }
}

}  // namespace geniex::qairt
