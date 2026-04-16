#pragma once

#include <functional>
#include <string>
#include <unordered_map>

#include "llm/llm_model.h"

// Model headers
#include "qwen3.h"
#include "phi4.h"
#include "phi3_5.h"
#include "granite4.h"

namespace geniex {

struct LlmModelEntry {
    std::function<LLMModel()> make_model;
    std::string pipeline_name;  // name used by LLMPipeline for chat template selection
};

inline const std::unordered_map<std::string, LlmModelEntry>& llm_model_registry() {
    static const std::unordered_map<std::string, LlmModelEntry> registry = {
        {"qwen3-4b", {qwen3_4b_instruct_2507_aihub::makeModel, "qwen3"}},
        {"qwen3-4b-aihub", {qwen3_4b_aihub::makeModel, "qwen3"}},
        {"qwen3-4b-base", {qwen3_4b::makeModel, "qwen3"}},
        {"qwen3-8b", {qwen3_8b::makeModel, "qwen3"}},
        {"phi4", {phi4::makeModel, "phi4"}},
        {"phi3.5", {phi3_5::makeModel, "phi3.5"}},
        {"phi3.5-aihub", {phi3_5_aihub::makeModel, "phi3.5"}},
        {"granite4", {granite4_micro::makeModel, "granite4"}},
    };
    return registry;
}

}  // namespace geniex