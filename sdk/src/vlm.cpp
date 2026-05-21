#include <cstring>
#include <memory>

#include "geniex.h"
#include "logging.h"
#include "plugin/IVlm.h"
#include "profile.h"
#include "registry.h"
#include "utils.h"

using namespace geniex;

int32_t geniex_vlm_create(const geniex_VlmCreateInput* input, geniex_VLM** out_handle) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = geniex::Registry::instance().get<geniex::IVlm>(input->plugin_id);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_SUPPORTED;

        int32_t result = backend->create(input);
        if (result != GENIEX_SUCCESS) {
            delete backend;
        } else {
            *out_handle = reinterpret_cast<geniex_VLM*>(backend);
        }
        return result;
    } catch (const PluginNotFoundException& e) {
        GENIEX_LOG_ERROR("plugin not found");
        return GENIEX_ERROR_COMMON_PLUGIN_INVALID;
    } catch (const PluginLoadException& e) {
        GENIEX_LOG_ERROR("plugin load error");
        return GENIEX_ERROR_COMMON_PLUGIN_LOAD;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to create VLM: {}", e.what());
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
}

int32_t geniex_vlm_destroy(geniex_VLM* handle) {
    GENIEX_LOG_TRACE("destroying vlm");

    try {
        auto backend = reinterpret_cast<IVlm*>(handle);
        if (!backend) return GENIEX_ERROR_COMMON_INVALID_INPUT;
        delete backend;
        return GENIEX_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to destroy VLM: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_vlm_reset(geniex_VLM* handle) {
    GENIEX_LOG_TRACE("resetting vlm");

    try {
        auto backend = reinterpret_cast<IVlm*>(handle);
        if (!backend) return GENIEX_ERROR_COMMON_INVALID_INPUT;
        return backend->reset();
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to reset VLM: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_vlm_apply_chat_template(
    geniex_VLM* handle, const geniex_VlmApplyChatTemplateInput* input, geniex_VlmApplyChatTemplateOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<IVlm*>(handle);
        if (!backend) return GENIEX_ERROR_COMMON_INVALID_INPUT;
        return backend->apply_chat_template(input, output);
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to apply chat template to VLM: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_vlm_get_capabilities(geniex_VLM* handle, geniex_VlmCapabilities* output) {
    GENIEX_LOG_TRACE("querying vlm capabilities");

    try {
        auto backend = reinterpret_cast<IVlm*>(handle);
        if (!backend || !output) return GENIEX_ERROR_COMMON_INVALID_INPUT;
        return backend->get_capabilities(output);
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to query VLM capabilities: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_vlm_generate(
    geniex_VLM* handle, const geniex_VlmGenerateInput* input, geniex_VlmGenerateOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<IVlm*>(handle);
        if (!backend) return GENIEX_ERROR_COMMON_INVALID_INPUT;

        // Wrap the user's callback with UTF-8 validation if a callback is provided
        std::unique_ptr<Utf8CallbackWrapper> wrapper;
        geniex_VlmGenerateInput              modified_input;

        if (input->on_token) {
            // Create wrapper that accumulates incomplete UTF-8 sequences
            wrapper                     = std::make_unique<Utf8CallbackWrapper>();
            wrapper->original_callback  = input->on_token;
            wrapper->original_user_data = input->user_data;

            // Create a modified input with our wrapped callback
            modified_input           = *input;
            modified_input.on_token  = get_utf8_callback_wrapper();
            modified_input.user_data = wrapper.get();

            int32_t result = backend->generate(&modified_input, output);
            calculate_profile_data(output->profile_data);
            GENIEX_LOG_TRACE("{}: {}", static_cast<geniex_ErrorCode>(result), output);

            // Flush any remaining incomplete UTF-8 sequence after generation completes
            wrapper->flush();

            return result;
        } else {
            // No callback, pass through directly
            int32_t result = backend->generate(input, output);
            calculate_profile_data(output->profile_data);
            GENIEX_LOG_TRACE("{}: {}", static_cast<geniex_ErrorCode>(result), output);
            return result;
        }
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to generate VLM: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}
