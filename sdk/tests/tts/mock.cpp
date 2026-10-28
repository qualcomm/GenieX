// SPDX-License-Identifier: BSD-3-Clause
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>

#include "plugin/ITts.h"
#include "plugin/Plugin.h"

class MockTts : public geniex::ITts {
    std::atomic<bool> stop_{false};

   public:
    int32_t create(const geniex_TtsCreateInput*) override { return GENIEX_SUCCESS; }
    int32_t cancel() override {
        stop_ = true;
        return GENIEX_SUCCESS;
    }
    int32_t synthesize(const geniex_TtsSynthesizeInput* in, geniex_TtsSynthesizeOutput* out) override {
        if (std::string(in->text) == "throw") throw std::runtime_error("test exception");
        stop_            = false;
        out->sample_rate = 24000;
        const float pcm[]{0.0f, 0.5f, -0.5f, 0.25f};
        for (int i = 0; i < 10; ++i) {
            if (stop_) return GENIEX_ERROR_COMMON_CANCELLED;
            out->samples += 4;
            if (!in->on_audio(pcm, 4, out->sample_rate, in->user_data)) return GENIEX_ERROR_COMMON_CANCELLED;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return stop_ ? GENIEX_ERROR_COMMON_CANCELLED : GENIEX_SUCCESS;
    }
};
class MockPlugin : public geniex::Plugin {};
extern "C" PLUGIN_API geniex_PluginId plugin_id() {
#ifdef LEGACY_PLUGIN
    return "legacy";
#else
    return "mock_tts";
#endif
}
extern "C" PLUGIN_API geniex::Plugin* create_plugin() { return new MockPlugin(); }
#ifndef LEGACY_PLUGIN
extern "C" PLUGIN_API geniex::ITts* geniex_create_tts_v1() { return new MockTts(); }
#endif
