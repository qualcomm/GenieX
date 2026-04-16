#pragma once

#include <algorithm>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "vlm/vlm_model.h"
#include "types.h"

// Model headers
#include "qwen2_omni.h"

namespace geniex {

// Processor type determines which mm-process processor to create and how to
// preprocess multimodal inputs before calling VLMModel::generate().
enum class VlmProcessorType {
    kQwen2_5Omni,  // audio + vision + text
    kQwen3VL,      // vision + text only
};

// Describes how to build all components (LLM shards, vision encoder bins, audio
// encoder bins) from the model directory.  The `make_model` callback receives
// the runtime config plus a model-specific config that the entry constructed
// from file paths found in model_dir.
struct VlmModelEntry {
    // Build and return a fully-initialised VLMModel.
    // `runtime_cfg` has QNN DLL paths already resolved.
    // `model_dir` is the parent directory of the .nexa file.
    std::function<std::unique_ptr<VLMModel>(const QnnRuntimeConfig& runtime_cfg,
                                            const std::filesystem::path& model_dir)> make_model;

    // Which chat-template / preprocessor family this model belongs to.
    VlmProcessorType processor_type;

    // Pipeline name used for chat template selection (passed to LLMPipeline).
    std::string pipeline_name;
};

// ── Factories ────────────────────────────────────────────────────────────────

namespace vlm_factories {

// Generic Qwen2-Omni family factory.  Discovers LLM shards, vision encoder
// bins and audio encoder bins in sub-directories of `model_dir`.
inline std::unique_ptr<VLMModel> make_qwen2_omni(
    const QnnRuntimeConfig& runtime_cfg,
    const std::filesystem::path& model_dir) {

    namespace fs = std::filesystem;
    qwen2_omni::Qwen2OmniConfig config;

    // ── LLM shards ──────────────────────────────────────────────────────
    // Expect model_dir/<something>/  or model_dir/ itself to contain .bin
    // shards.  The main .nexa lives in model_dir; shards are siblings.
    auto collect_bins = [](const fs::path& dir) -> std::vector<std::string> {
        std::vector<std::string> bins;
        if (!fs::exists(dir) || !fs::is_directory(dir)) return bins;
        for (const auto& e : fs::directory_iterator(dir))
            if (e.is_regular_file() && e.path().extension() == ".bin")
                bins.push_back(e.path().string());
        std::sort(bins.begin(), bins.end());
        return bins;
    };

    config.llm_config.model_paths = collect_bins(model_dir);
    if (config.llm_config.model_paths.empty()) {
        // Try "llm" subdirectory — some model packs split by component.
        for (const auto& sub : fs::directory_iterator(model_dir)) {
            if (sub.is_directory()) {
                auto bins = collect_bins(sub.path());
                if (!bins.empty() && config.llm_config.model_paths.empty())
                    config.llm_config.model_paths = std::move(bins);
            }
        }
    }

    // Tokenizer
    auto tok = model_dir / "tokenizer.json";
    if (fs::exists(tok)) config.llm_config.tokenizer_path = tok.string();

    // Embedding table
    auto emb = model_dir / "embed_tokens.npy";
    if (fs::exists(emb)) config.llm_config.embedding_path = emb.string();

    // HTP backend config
    auto htp = model_dir / "htp_backend_ext_config.json";
    if (fs::exists(htp)) config.llm_config.htp_config_path = htp.string();

    // ── Vision encoder ──────────────────────────────────────────────────
    auto vit_dir = model_dir / "vit";
    if (fs::exists(vit_dir))
        config.vision_config.model_paths = collect_bins(vit_dir);

    // ── Audio encoder ───────────────────────────────────────────────────
    auto audio_dir = model_dir / "audio_encoder";
    if (fs::exists(audio_dir))
        config.audio_config.model_paths = collect_bins(audio_dir);

    return qwen2_omni::makeModel(runtime_cfg, config);
}

}  // namespace vlm_factories

// ── Registry ─────────────────────────────────────────────────────────────────

inline const std::unordered_map<std::string, VlmModelEntry>& vlm_model_registry() {
    static const std::unordered_map<std::string, VlmModelEntry> registry = {
        {"omni-neural",
            {vlm_factories::make_qwen2_omni, VlmProcessorType::kQwen2_5Omni, "qwen2-omni"}},
        {"auto-neural",
            {vlm_factories::make_qwen2_omni, VlmProcessorType::kQwen2_5Omni, "qwen2-omni"}},
        {"qwen3vl",
            {vlm_factories::make_qwen2_omni, VlmProcessorType::kQwen3VL, "qwen3vl"}},
    };
    return registry;
}

}  // namespace geniex
