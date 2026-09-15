// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Maps the public C `geniex_PowerMode` into qairt-core `geniex::PerfProfile`.
// Shared by the qairt LLM and VLM plugins.

#pragma once

#include "geniex.h"
#include "types.h"  // geniex::ModelConfig, geniex::PerfProfile

namespace geniex::qairt {

inline PerfProfile to_perf_profile(geniex_PowerMode mode) {
    switch (mode) {
        case GENIEX_POWER_MODE_LOW_POWER_SAVER:
            return PerfProfile::LOW_POWER_SAVER;
        case GENIEX_POWER_MODE_POWER_SAVER:
            return PerfProfile::POWER_SAVER;
        case GENIEX_POWER_MODE_HIGH_POWER_SAVER:
            return PerfProfile::HIGH_POWER_SAVER;
        case GENIEX_POWER_MODE_LOW_BALANCED:
            return PerfProfile::LOW_BALANCED;
        case GENIEX_POWER_MODE_BALANCED:
            return PerfProfile::BALANCED;
        case GENIEX_POWER_MODE_HIGH_PERFORMANCE:
            return PerfProfile::HIGH_PERFORMANCE;
        case GENIEX_POWER_MODE_SUSTAINED_HIGH_PERFORMANCE:
            return PerfProfile::SUSTAINED_HIGH_PERFORMANCE;
        case GENIEX_POWER_MODE_BURST:
        default:
            return PerfProfile::BURST;
    }
}

// Sets model_cfg.perf_profile from `mode`. resolveHtpPerfConfig
// (geniex-qairt core/src/llm/llm_spec_loader.cpp) lets this win over whatever
// the bundle's htp_backend_ext_config.json sets.
inline void apply_power_mode(geniex_PowerMode mode, ModelConfig& model_cfg) {
    model_cfg.perf_profile = to_perf_profile(mode);
}

}  // namespace geniex::qairt
