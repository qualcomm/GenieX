#pragma once

#include <functional>
#include <string>
#include <unordered_map>

#include "pipeline/llm_pipeline.h"
#include "types.h"

// Model headers
#include "qwen3.h"
#include "phi4.h"
#include "phi3_5.h"
#include "granite4.h"

namespace geniex {

struct LlmModelEntry {
    std::function<std::optional<LLMPipeline>(const QnnRuntimeConfig&, const ModelConfig&)> make_pipeline;
};

inline const std::unordered_map<std::string, LlmModelEntry>& llm_model_registry() {
    static const std::unordered_map<std::string, LlmModelEntry> registry = {
        {"qwen3-4b",       {qwen3_4b_instruct_2507_aihub::makePipeline}},
        {"qwen3-4b-aihub", {qwen3_4b_aihub::makePipeline}},
        {"qwen3-4b-base",  {qwen3_4b::makePipeline}},
        {"qwen3-8b",       {qwen3_8b::makePipeline}},
        {"phi4",           {phi4::makePipeline}},
        {"phi3.5",         {phi3_5::makePipeline}},
        {"phi3.5-aihub",   {phi3_5_aihub::makePipeline}},
        {"granite4",       {granite4_micro::makePipeline}},
    };
    return registry;
}

}  // namespace geniex