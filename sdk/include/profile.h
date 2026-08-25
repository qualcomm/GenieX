// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "geniex.h"

namespace geniex {

// Ref:
// - TODO: add the current profiling metrics design reference.
// - TODO: add the current ASR and TTS profiling reference.
inline void calculate_profile_data(geniex_ProfileData &pd) {
    pd.prefill_speed =
        pd.prompt_time > 0 ? static_cast<double>(pd.prompt_tokens) / (static_cast<double>(pd.prompt_time) / 1e6) : 0.0;

    pd.decoding_speed = pd.decode_time > 0
                            ? static_cast<double>(pd.generated_tokens) / (static_cast<double>(pd.decode_time) / 1e6)
                            : 0.0;
}

}  // namespace geniex
