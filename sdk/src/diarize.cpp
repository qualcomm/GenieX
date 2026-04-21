#include <cstdlib>

#include "geniex.h"
#include "logging.h"
#include "plugin/IDiarize.h"
#include "profile.h"
#include "registry.h"

using namespace geniex;

int32_t geniex_diarize_create(const geniex_DiarizeCreateInput* input, geniex_Diarize** out_handle) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto&     registry       = Registry::instance();
        IDiarize* diarize_plugin = registry.get<IDiarize>(input->plugin_id);
        if (!diarize_plugin) return GENIEX_ERROR_COMMON_NOT_SUPPORTED;

        int32_t result = diarize_plugin->create(input);
        if (result != GENIEX_SUCCESS) {
            return result;
        } else {
            *out_handle = reinterpret_cast<geniex_Diarize*>(diarize_plugin);
        }

        return GENIEX_SUCCESS;
    } catch (const PluginNotFoundException& e) {
        GENIEX_LOG_ERROR("plugin not found");
        return GENIEX_ERROR_COMMON_PLUGIN_INVALID;
    } catch (const PluginLoadException& e) {
        GENIEX_LOG_ERROR("plugin load error");
        return GENIEX_ERROR_COMMON_PLUGIN_LOAD;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to create Diarization model: {}", e.what());
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
}

int32_t geniex_diarize_destroy(geniex_Diarize* handle) {
    GENIEX_LOG_TRACE("destroying Diarization model");

    try {
        auto backend = reinterpret_cast<IDiarize*>(handle);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        delete backend;
        return GENIEX_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to destroy Diarization model: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_diarize_infer(
    geniex_Diarize* handle, const geniex_DiarizeInferInput* input, geniex_DiarizeInferOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<IDiarize*>(handle);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;

        int32_t result = backend->infer(input, output);
        calculate_profile_data(output->profile_data);
        GENIEX_LOG_TRACE("{}: {}", static_cast<geniex_ErrorCode>(result), output);

        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to perform Diarization inference: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}
