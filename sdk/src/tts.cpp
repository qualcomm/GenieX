// SPDX-License-Identifier: BSD-3-Clause
#include <cstring>
#include <memory>

#include "geniex_tts.h"
#include "logging.h"
#include "registry.h"

namespace {
bool present(const char* value) { return value && *value; }
}  // namespace

int32_t geniex_tts_create(const geniex_TtsCreateInput* input, geniex_TTS** out_handle) {
    if (!out_handle) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    *out_handle = nullptr;
    if (!input || input->struct_size != sizeof(*input) || !present(input->plugin_id) || !present(input->model_path) ||
        !present(input->device))
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    if (input->flags & ~GENIEX_TTS_VERIFY_NPU) return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    try {
        std::unique_ptr<geniex::ITts> backend(geniex::Registry::instance().get<geniex::ITts>(input->plugin_id));
        if (!backend) return GENIEX_ERROR_COMMON_NOT_SUPPORTED;
        auto result = backend->create(input);
        if (result == GENIEX_SUCCESS) *out_handle = reinterpret_cast<geniex_TTS*>(backend.release());
        return result;
    } catch (const geniex::PluginNotFoundException&) {
        return GENIEX_ERROR_COMMON_PLUGIN_INVALID;
    } catch (const geniex::PluginLoadException&) {
        return GENIEX_ERROR_COMMON_PLUGIN_LOAD;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("TTS create: {}", e.what());
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
}

int32_t geniex_tts_destroy(geniex_TTS* handle) {
    if (!handle) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    delete reinterpret_cast<geniex::ITts*>(handle);
    return GENIEX_SUCCESS;
}

int32_t geniex_tts_synthesize(
    geniex_TTS* handle, const geniex_TtsSynthesizeInput* input, geniex_TtsSynthesizeOutput* output) {
    if (!output || output->struct_size != sizeof(*output)) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    *output                = {};
    output->struct_size    = sizeof(*output);
    output->first_audio_ms = -1;
    if (!handle || !input || input->struct_size != sizeof(*input) || !present(input->text) || !input->on_audio)
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    try {
        return reinterpret_cast<geniex::ITts*>(handle)->synthesize(input, output);
    } catch (const std::invalid_argument& e) {
        GENIEX_LOG_ERROR("TTS synthesis input: {}", e.what());
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("TTS synthesis: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_tts_cancel(geniex_TTS* handle) {
    if (!handle) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    try {
        return reinterpret_cast<geniex::ITts*>(handle)->cancel();
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("TTS cancel: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}
