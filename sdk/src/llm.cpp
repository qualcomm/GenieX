#include <cstring>
#include <memory>

#include "geniex.h"
#include "logging.h"
#include "plugin/ILlm.h"
#include "profile.h"
#include "registry.h"
#include "utils.h"

using namespace geniex;

int32_t geniex_llm_create(const geniex_LlmCreateInput* input, geniex_LLM** out_handle) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = geniex::Registry::instance().get<geniex::ILlm>(input->plugin_id);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_SUPPORTED;
        int32_t res = backend->create(input);
        if (res != GENIEX_SUCCESS) {
            delete backend;
        } else {
            *out_handle = reinterpret_cast<geniex_LLM*>(backend);
        }
        return res;
    } catch (const PluginNotFoundException& e) {
        GENIEX_LOG_ERROR("plugin not found");
        return GENIEX_ERROR_COMMON_PLUGIN_INVALID;
    } catch (const PluginLoadException& e) {
        GENIEX_LOG_ERROR("plugin load error");
        return GENIEX_ERROR_COMMON_PLUGIN_LOAD;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("creating llm error: {}", e.what());
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
}

int32_t geniex_llm_destroy(geniex_LLM* h) {
    GENIEX_LOG_TRACE("llm destroy");

    try {
        auto backend = reinterpret_cast<ILlm*>(h);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        delete backend;
        return GENIEX_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("destroy llm error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_llm_reset(geniex_LLM* h) {
    GENIEX_LOG_TRACE("llm reset");

    try {
        auto backend = reinterpret_cast<ILlm*>(h);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        return backend->reset();
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("reset llm error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_llm_save_kv_cache(
    geniex_LLM* h, const geniex_KvCacheSaveInput* input, geniex_KvCacheSaveOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<ILlm*>(h);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        auto result = backend->save_kv_cache(input, output);
        GENIEX_LOG_TRACE("{}: {}", static_cast<geniex_ErrorCode>(result), output);
        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("llm save kv cache error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_llm_load_kv_cache(
    geniex_LLM* h, const geniex_KvCacheLoadInput* input, geniex_KvCacheLoadOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<ILlm*>(h);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        auto result = backend->load_kv_cache(input, output);
        GENIEX_LOG_TRACE("{}: {}", static_cast<geniex_ErrorCode>(result), output);
        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("llm load kv cache error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_llm_apply_chat_template(
    geniex_LLM* h, const geniex_LlmApplyChatTemplateInput* input, geniex_LlmApplyChatTemplateOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<ILlm*>(h);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        auto result = backend->apply_chat_template(input, output);
        GENIEX_LOG_TRACE("{}: {}", static_cast<geniex_ErrorCode>(result), output);
        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("llm apply chat template error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_llm_generate(geniex_LLM* h, const geniex_LlmGenerateInput* input, geniex_LlmGenerateOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        auto backend = reinterpret_cast<ILlm*>(h);
        if (!backend) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;

        // Wrap the user's callback with UTF-8 validation if a callback is provided
        std::unique_ptr<Utf8CallbackWrapper> wrapper;
        geniex_LlmGenerateInput              modified_input;

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
        GENIEX_LOG_ERROR("llm generate error: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}
