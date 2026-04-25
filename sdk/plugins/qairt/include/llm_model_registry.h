#pragma once

#include <functional>
#include <string>
#include <unordered_map>

#include "pipeline/llm_pipeline.h"
#include "types.h"

// Model headers. Only include headers that are actually present in the
// pinned third-party/geniex-qairt submodule.
// TODO: re-enable granite4.h / phi4.h once the submodule ships them.
#include "llama3_2.h"
#include "phi3_5.h"
#include "qwen3.h"

namespace geniex {

struct LlmModelEntry {
    std::function<std::optional<LLMPipeline>(const QnnRuntimeConfig&, const ModelConfig&)> make_pipeline;
};

inline const std::unordered_map<std::string, LlmModelEntry>& llm_model_registry() {
    static const std::unordered_map<std::string, LlmModelEntry> registry = {
        // Only namespaces that actually exist in the pinned geniex-qairt
        // submodule are wired in. *_aihub / phi4 / granite4 entries will
        // be restored once the submodule bumps.
        {"qwen3-4b-base", {qwen3_4b::makePipeline}},
        {"qwen3-4b-instruct", {qwen3_4b_instruct_2507::makePipeline}},
        {"qwen3-8b", {qwen3_8b::makePipeline}},
        {"phi3.5", {phi3_5::makePipeline}},
        {"llama3_2-1b", {llama3_2_1b::makePipeline}},
        {"llama3_2-3b", {llama3_2_3b::makePipeline}},
    };
    return registry;
}

}  // namespace geniex