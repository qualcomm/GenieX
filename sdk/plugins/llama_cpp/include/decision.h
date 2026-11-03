// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>
#include <vector>

#include "geniex.h"
#include "llama.h"

namespace geniex {

struct DecisionPrepared {
    std::string              prompt;
    std::string              reversed_prompt;
    std::vector<std::string> candidates;
    std::vector<size_t>      original_index;
    size_t                   true_index = 0;
    std::string              type;
    std::string              question_type;
    float                    temperature = 1.0f;

    // Liquid protocols carry tokenized inputs: Omni tokenizes budgeted pieces,
    // while d1 reads the best logit from each group of verbalizer variants.
    std::vector<llama_token>              tokens;
    std::vector<std::vector<llama_token>> token_groups;
    std::vector<size_t>                   markers;
    std::vector<float>                    score_bias;
    int32_t                               column = 0;
};

std::string decision_metadata_string(const llama_model* model, const std::string& key);

int32_t prepare_decision(const llama_model* model, const geniex_DecideInput* input, DecisionPrepared* output);

}  // namespace geniex
