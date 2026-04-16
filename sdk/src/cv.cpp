#include <cstdlib>

#include "logging.h"
#include "ml.h"
#include "registry.h"

using namespace geniex;

int32_t ml_cv_create(const ml_CVCreateInput* input, ml_CV** out_handle) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto& registry  = Registry::instance();
        ICv*  cv_plugin = registry.get<ICv>(input->plugin_id);
        if (!cv_plugin) return ML_ERROR_COMMON_NOT_SUPPORTED;

        int32_t result = cv_plugin->create(input);
        if (result != ML_SUCCESS) {
            return result;
        } else {
            *out_handle = reinterpret_cast<ml_CV*>(cv_plugin);
        }

        return ML_SUCCESS;
    } catch (const PluginNotFoundException& e) {
        GENIEX_LOG_ERROR("plugin not found");
        return ML_ERROR_COMMON_PLUGIN_INVALID;
    } catch (const PluginLoadException& e) {
        GENIEX_LOG_ERROR("plugin load error");
        return ML_ERROR_COMMON_PLUGIN_LOAD;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to create CV model: {}", e.what());
        return ML_ERROR_COMMON_MODEL_LOAD;
    }
}

int32_t ml_cv_destroy(ml_CV* handle) {
    GENIEX_LOG_TRACE("destroying CV model");

    try {
        // TODO: this behavior is not same as other
        auto backend = reinterpret_cast<ICv*>(handle);
        if (!backend) return ML_ERROR_COMMON_NOT_INITIALIZED;
        delete backend;
        return ML_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to destroy CV model: {}", e.what());
        return ML_ERROR_COMMON_UNKNOWN;
    }
}

int32_t ml_cv_infer(const ml_CV* handle, const ml_CVInferInput* input, ml_CVInferOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<ICv*>(const_cast<ml_CV*>(handle));
        if (!backend) return ML_ERROR_COMMON_NOT_INITIALIZED;

        int32_t result = backend->infer(input, output);
        // TODO: geniex::calculate_profile_data(output->profile_data);
        GENIEX_LOG_TRACE("{}: {}", static_cast<ml_ErrorCode>(result), output);

        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to perform CV inference: {}", e.what());
        return ML_ERROR_COMMON_UNKNOWN;
    }
}
