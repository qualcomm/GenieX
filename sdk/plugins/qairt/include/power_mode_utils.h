// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Maps the public C `geniex_PowerMode` into qairt-core `geniex::PerfProfile`,
// and probes a bundle's htp_backend_ext_config.json for a conflicting
// override. Shared by the qairt LLM and VLM plugins.

#pragma once

#include "geniex.h"
#include "llm/llm_spec_loader.h"  // parseHtpConfig
#include "logging.h"
#include "types.h"  // geniex::ModelConfig, geniex::PerfProfile, geniex::HtpPerfConfig

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

// Sets model_cfg.perf_profile from `mode`, then warns if the bundle's
// htp_backend_ext_config.json (when present) sets its own perf_profile:
// Model::initialize (geniex-qairt core/src/model_init.cpp) seeds from
// model_cfg.perf_profile and lets parseHtpConfig overwrite it afterwards, so
// a bundle value silently wins over the caller's --power-mode.
inline void apply_power_mode(geniex_PowerMode mode, ModelConfig& model_cfg) {
    model_cfg.perf_profile = to_perf_profile(mode);

    if (model_cfg.htp_config_path.empty()) return;
    HtpPerfConfig probe{model_cfg.perf_profile};
    parseHtpConfig(model_cfg.htp_config_path, probe);
    if (probe.profile != model_cfg.perf_profile) {
        GENIEX_LOG_WARN(
            "bundle's htp_backend_ext_config.json sets its own perf_profile; it overrides the "
            "requested power_mode for this model");
    }
}

}  // namespace geniex::qairt
