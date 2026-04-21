#include <cstdlib>

#include "geniex.h"
#include "logging.h"
#include "registry.h"

using namespace geniex;

int32_t geniex_cv_create(const geniex_CVCreateInput* input, geniex_CV** out_handle) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto& registry  = Registry::instance();
        ICv*  cv_plugin = registry.get<ICv>(input->plugin_id);
        if (!cv_plugin) return GENIEX_ERROR_COMMON_NOT_SUPPORTED;

        int32_t result = cv_plugin->create(input);
        if (result != GENIEX_SUCCESS) {
            return result;
        } else {
            *out_handle = reinterpret_cast<geniex_CV*>(cv_plugin);
        }

        return GENIEX_SUCCESS;
    } catch (const PluginNotFoundException& e) {
        GENIEX_LOG_ERROR("plugin not found");
        return GENIEX_ERROR_COMMON_PLUGIN_INVALID;
    } catch (const PluginLoadException& e) {
        GENIEX_LOG_ERROR("plugin load error");
        return GENIEX_ERROR_COMMON_PLUGIN_LOAD;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to create CV model: {}", e.what());
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
}

int32_t geniex_cv_destroy(geniex_CV* handle) {
    GENIEX_LOG_TRACE("destroying CV model");

    try {
        // TODO: this behavior is not same as other
        auto backend = reinterpret_cast<ICv*>(handle);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        delete backend;
        return GENIEX_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to destroy CV model: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_cv_infer(const geniex_CV* handle, const geniex_CVInferInput* input, geniex_CVInferOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<ICv*>(const_cast<geniex_CV*>(handle));
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;

        int32_t result = backend->infer(input, output);
        // TODO: geniex::calculate_profile_data(output->profile_data);
        GENIEX_LOG_TRACE("{}: {}", static_cast<geniex_ErrorCode>(result), output);

        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to perform CV inference: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}
