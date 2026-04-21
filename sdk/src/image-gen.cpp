#include <cstdlib>

#include "geniex.h"
#include "logging.h"
#include "plugin/IImageGen.h"
#include "registry.h"

using namespace geniex;

int32_t geniex_imagegen_create(const geniex_ImageGenCreateInput* input, geniex_ImageGen** out_handle) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = Registry::instance().get<IImageGen>(input->plugin_id);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_SUPPORTED;
        int32_t result = backend->create(input);
        if (result != GENIEX_SUCCESS) {
            delete backend;
        } else {
            *out_handle = reinterpret_cast<geniex_ImageGen*>(backend);
        }
        return result;
    } catch (const PluginNotFoundException& e) {
        GENIEX_LOG_ERROR("plugin not found");
        return GENIEX_ERROR_COMMON_PLUGIN_INVALID;
    } catch (const PluginLoadException& e) {
        GENIEX_LOG_ERROR("plugin load error");
        return GENIEX_ERROR_COMMON_PLUGIN_LOAD;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to create image gen: {}", e.what());
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
}

int32_t geniex_imagegen_destroy(geniex_ImageGen* handle) {
    GENIEX_LOG_TRACE("destroying image gen");

    try {
        auto backend = reinterpret_cast<IImageGen*>(handle);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        delete backend;
        return GENIEX_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("destroy image gen error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_imagegen_txt2img(
    geniex_ImageGen* handle, const geniex_ImageGenTxt2ImgInput* input, geniex_ImageGenOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<IImageGen*>(handle);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;

        int32_t result = backend->txt2img(input, output);
        // TODO: add profile data
        // calculate_profile_data(output->profile_data);
        GENIEX_LOG_TRACE("{}: {}", static_cast<geniex_ErrorCode>(result), output);
        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("image gen txt2img error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_imagegen_img2img(
    geniex_ImageGen* handle, const geniex_ImageGenImg2ImgInput* input, geniex_ImageGenOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<IImageGen*>(handle);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;

        int32_t result = backend->img2img(input, output);
        // TODO: add profile data
        // calculate_profile_data(output->profile_data);
        GENIEX_LOG_TRACE("{}: {}", static_cast<geniex_ErrorCode>(result), output);
        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("image gen img2img error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}
