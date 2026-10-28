// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
#include <cstring>
#include <filesystem>
#include <memory>

#include "phonon/phonon_model.h"
#include "plugin/ITts.h"
#include "plugin/Plugin.h"
#include "qnn_runtime_utils.h"

namespace {
class QairtTts final : public geniex::ITts {
    std::unique_ptr<geniex::TTSPipeline> pipeline_;

   public:
    int32_t create(const geniex_TtsCreateInput* input) override {
        if (std::strcmp(input->device, "npu") != 0 && std::strcmp(input->device, "NPU") != 0)
            return GENIEX_ERROR_COMMON_INVALID_DEVICE;
        if (!input->text_library || !*input->text_library || !input->soc_model || !*input->soc_model)
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        const auto root    = std::filesystem::u8path(input->model_path);
        auto       runtime = geniex::qairt::runtime::make_qnn_runtime_config(root);
        if (input->runtime_path && *input->runtime_path) {
            const auto asserted =
                geniex::qairt::paths::locate_qnn_host_lib_dir(std::filesystem::u8path(input->runtime_path));
            std::error_code error;
            const auto current = runtime.backend_path ? std::filesystem::u8path(*runtime.backend_path).parent_path()
                                                      : std::filesystem::path{};
            if (asserted.empty() || current.empty() || !std::filesystem::equivalent(asserted, current, error) || error)
                return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
        geniex::PhononOptions options;
        options.bundle_dir   = input->model_path;
        options.soc_model    = input->soc_model;
        options.language     = input->language ? input->language : "";
        options.text_library = input->text_library;
        options.verify_npu   = (input->flags & GENIEX_TTS_VERIFY_NPU) != 0;
        auto model           = std::make_unique<geniex::PhononModel>(options);
        if (!model->load(runtime)) return GENIEX_ERROR_COMMON_MODEL_LOAD;
        pipeline_ = std::make_unique<geniex::TTSPipeline>(std::move(model));
        return GENIEX_SUCCESS;
    }
    int32_t cancel() override {
        if (!pipeline_) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        pipeline_->cancel();
        return GENIEX_SUCCESS;
    }
    int32_t synthesize(const geniex_TtsSynthesizeInput* input, geniex_TtsSynthesizeOutput* output) override {
        if (!pipeline_ || !pipeline_->isReady()) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        geniex::SpeechConfig config;
        config.voice         = input->voice ? input->voice : "";
        config.collect_audio = false;
        const auto result   = pipeline_->synthesize(input->text, config, [&](const float* pcm, size_t count, int rate) {
            return input->on_audio(pcm, static_cast<uint32_t>(count), static_cast<uint32_t>(rate), input->user_data);
        });
        output->sample_rate = result.sample_rate;
        output->samples     = result.samples;
        output->first_audio_ms = result.first_audio_ms;
        output->generation_ms  = result.total_ms;
        output->prefill_calls  = static_cast<uint32_t>(result.graphs.at("prefill").calls);
        output->step_calls     = static_cast<uint32_t>(result.graphs.at("step").calls);
        output->prefill_npu_ms = result.graphs.at("prefill").accelerator_ms;
        output->step_npu_ms    = result.graphs.at("step").accelerator_ms;
        return result.cancelled ? GENIEX_ERROR_COMMON_CANCELLED : GENIEX_SUCCESS;
    }
};
}  // namespace

extern "C" PLUGIN_API geniex::ITts* geniex_create_tts_v1() { return new QairtTts(); }
