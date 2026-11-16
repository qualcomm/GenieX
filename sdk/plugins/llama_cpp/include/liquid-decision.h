// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "chat.h"
#include "decision.h"

namespace geniex {

bool is_liquid_decision(const std::string& type);
void configure_decision_context(
    const llama_model* model, llama_context_params& params, bool default_batch = true, bool default_ubatch = true);
int32_t evaluate_liquid_decision(llama_context* ctx, const DecisionPrepared& prepared, geniex_DecideOutput* output);

void               prepare_liquid_decision(const llama_model* model, common_json target, DecisionPrepared& output);
std::vector<float> liquid_probabilities(const DecisionPrepared& prepared, const std::vector<float>& scores);

}  // namespace geniex
