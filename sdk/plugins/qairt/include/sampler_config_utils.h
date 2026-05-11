// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Translates the public C `geniex_SamplerConfig` into qairt-core
// `geniex::GenerationConfig`. Shared by the qairt LLM and VLM plugins.

#pragma once

#include "geniex.h"
#include "geniex-proc/types.h"  // for GENIEX_DEFAULT_SEED
#include "logging.h"
#include "types.h"  // geniex::GenerationConfig

#include <cstdint>
#include <fstream>
#include <sstream>

namespace geniex::qairt {

// QAIRT-recommended sampling defaults.
struct DefaultSamplerParams {
    uint32_t seed               = GENIEX_DEFAULT_SEED;  // random per process
    int32_t  top_k              = 40;
    float    top_p              = 0.95f;
    float    min_p              = 0.05f;
    float    temperature        = 0.8f;
    float    repetition_penalty = 1.0f;
    float    presence_penalty   = 0.0f;
    float    frequency_penalty  = 0.0f;
};

inline void apply_sampler_config(const geniex_SamplerConfig* cfg, GenerationConfig& out) {
    constexpr DefaultSamplerParams kDefaults{};

    out.enable_sampling = true;

    if (!cfg) {
        // No config → apply all defaults wholesale.
        out.seed               = kDefaults.seed;
        out.top_k              = kDefaults.top_k;
        out.top_p              = kDefaults.top_p;
        out.min_p              = kDefaults.min_p;
        out.temperature        = kDefaults.temperature;
        out.repetition_penalty = kDefaults.repetition_penalty;
        out.presence_penalty   = kDefaults.presence_penalty;
        out.frequency_penalty  = kDefaults.frequency_penalty;
        return;
    }

    // Zero-sentinel: every default-initialised (== 0) field maps to the QAIRT default
    out.seed               = (cfg->seed != 0) ? static_cast<uint32_t>(cfg->seed) : kDefaults.seed;
    out.top_k              = (cfg->top_k != 0) ? cfg->top_k : kDefaults.top_k;
    out.top_p              = (cfg->top_p != 0.0f) ? cfg->top_p : kDefaults.top_p;
    out.min_p              = (cfg->min_p != 0.0f) ? cfg->min_p : kDefaults.min_p;
    out.repetition_penalty = (cfg->repetition_penalty != 0.0f) ? cfg->repetition_penalty : kDefaults.repetition_penalty;
    out.presence_penalty   = (cfg->presence_penalty != 0.0f) ? cfg->presence_penalty : kDefaults.presence_penalty;
    out.frequency_penalty  = (cfg->frequency_penalty != 0.0f) ? cfg->frequency_penalty : kDefaults.frequency_penalty;

    // Temperature is special: 0 means "use default", a NEGATIVE value is the
    // greedy/argmax sentinel, positive is the literal temp.
    out.temperature        = (cfg->temperature == 0.0f) ? kDefaults.temperature : cfg->temperature;

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
