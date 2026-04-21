#include "geniex.h"
#include "logging.h"
#include "plugin/IReranker.h"
#include "profile.h"
#include "registry.h"

using namespace geniex;

int32_t geniex_reranker_create(const geniex_RerankerCreateInput* input, geniex_Reranker** out_handle) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        if (!input || !input->plugin_id) {
            GENIEX_LOG_ERROR("Invalid input parameters for reranker creation");
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        GENIEX_LOG_INFO("Retrieving backend for plugin: {}", input->plugin_id);
        auto backend = geniex::Registry::instance().get<geniex::IReranker>(input->plugin_id);
        if (!backend) {
            GENIEX_LOG_ERROR("Failed to get backend for plugin: {}", input->plugin_id);
            return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        }

        GENIEX_LOG_INFO("Backend retrieved successfully, calling create implementation");
        int32_t res = backend->create(input);
        if (res != GENIEX_SUCCESS) {
            delete backend;
        } else {
            GENIEX_LOG_INFO("Reranker created successfully");
            *out_handle = reinterpret_cast<geniex_Reranker*>(backend);
        }
        return res;
    } catch (const PluginNotFoundException& e) {
        GENIEX_LOG_ERROR("plugin not found");
        return GENIEX_ERROR_COMMON_PLUGIN_INVALID;
    } catch (const PluginLoadException& e) {
        GENIEX_LOG_ERROR("plugin load error");
        return GENIEX_ERROR_COMMON_PLUGIN_LOAD;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("Exception creating reranker: {}", e.what());
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
}

int32_t geniex_reranker_destroy(geniex_Reranker* handle) {
    GENIEX_LOG_INFO("Destroying reranker instance");

    try {
        auto backend = reinterpret_cast<geniex::IReranker*>(handle);
        if (!backend) {
            GENIEX_LOG_ERROR("Attempted to destroy null reranker handle");
            return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        }

        GENIEX_LOG_INFO("Deleting reranker backend instance");
        delete backend;
        GENIEX_LOG_INFO("Reranker destroyed successfully");
        return GENIEX_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("Exception destroying reranker: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_reranker_rerank(
    geniex_Reranker* h, const geniex_RerankerRerankInput* input, geniex_RerankerRerankOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        if (!input || !output) {
            GENIEX_LOG_ERROR("Invalid input or output parameters for reranking");
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        auto backend = reinterpret_cast<geniex::IReranker*>(h);
        if (!backend) {
            GENIEX_LOG_ERROR("Reranker backend is null");
            return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        }

        GENIEX_LOG_INFO("Calling backend rerank implementation");
        int32_t result = backend->rerank(input, output);

        if (result == GENIEX_SUCCESS) {
            GENIEX_LOG_INFO("Reranking operation completed successfully, generated {} scores", output->score_count);
        } else {
            GENIEX_LOG_ERROR("Reranking operation failed with error code: {}", result);
        }

        calculate_profile_data(output->profile_data);
        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("Exception in reranker rerank: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}
