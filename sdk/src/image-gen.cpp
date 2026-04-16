#include <cstdlib>

#include "logging.h"
#include "ml.h"
#include "plugin/IImageGen.h"
#include "registry.h"

using namespace geniex;

int32_t ml_imagegen_create(const ml_ImageGenCreateInput* input, ml_ImageGen** out_handle) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = Registry::instance().get<IImageGen>(input->plugin_id);
        if (!backend) return ML_ERROR_COMMON_NOT_SUPPORTED;
        int32_t result = backend->create(input);
        if (result != ML_SUCCESS) {
            delete backend;
        } else {
            *out_handle = reinterpret_cast<ml_ImageGen*>(backend);
        }
        return result;
    } catch (const PluginNotFoundException& e) {
        GENIEX_LOG_ERROR("plugin not found");
        return ML_ERROR_COMMON_PLUGIN_INVALID;
    } catch (const PluginLoadException& e) {
        GENIEX_LOG_ERROR("plugin load error");
        return ML_ERROR_COMMON_PLUGIN_LOAD;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to create image gen: {}", e.what());
        return ML_ERROR_COMMON_MODEL_LOAD;
    }
}

int32_t ml_imagegen_destroy(ml_ImageGen* handle) {
    GENIEX_LOG_TRACE("destroying image gen");

    try {
        auto backend = reinterpret_cast<IImageGen*>(handle);
        if (!backend) return ML_ERROR_COMMON_NOT_INITIALIZED;
        delete backend;
        return ML_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("destroy image gen error: {}", e.what());
        return ML_ERROR_COMMON_UNKNOWN;
    }
}

int32_t ml_imagegen_txt2img(ml_ImageGen* handle, const ml_ImageGenTxt2ImgInput* input, ml_ImageGenOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<IImageGen*>(handle);
        if (!backend) return ML_ERROR_COMMON_NOT_INITIALIZED;

        int32_t result = backend->txt2img(input, output);
        // TODO: add profile data
        // calculate_profile_data(output->profile_data);
        GENIEX_LOG_TRACE("{}: {}", static_cast<ml_ErrorCode>(result), output);
        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("image gen txt2img error: {}", e.what());
        return ML_ERROR_COMMON_UNKNOWN;
    }
}

int32_t ml_imagegen_img2img(ml_ImageGen* handle, const ml_ImageGenImg2ImgInput* input, ml_ImageGenOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<IImageGen*>(handle);
        if (!backend) return ML_ERROR_COMMON_NOT_INITIALIZED;

        int32_t result = backend->img2img(input, output);
        // TODO: add profile data
        // calculate_profile_data(output->profile_data);
        GENIEX_LOG_TRACE("{}: {}", static_cast<ml_ErrorCode>(result), output);
        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("image gen img2img error: {}", e.what());
        return ML_ERROR_COMMON_UNKNOWN;
    }
}
