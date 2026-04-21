#include "geniex.h"
#include "logging.h"
#include "plugin/IEmbedding.h"
#include "profile.h"
#include "registry.h"

using namespace geniex;

int32_t geniex_embedder_create(const geniex_EmbedderCreateInput* input, geniex_Embedder** out_handle) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        if (!input || !input->plugin_id) {
            GENIEX_LOG_ERROR("Invalid input parameters for embedder creation");
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        GENIEX_LOG_INFO("Retrieving backend for plugin: {}", input->plugin_id);
        auto backend = geniex::Registry::instance().get<geniex::IEmbedding>(input->plugin_id);
        if (!backend) {
            GENIEX_LOG_ERROR("Failed to get backend for plugin: {}", input->plugin_id);
            return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        }

        GENIEX_LOG_INFO("Backend retrieved successfully, calling create implementation");
        int32_t res = backend->create(input);
        if (res != GENIEX_SUCCESS) {
            GENIEX_LOG_ERROR("Backend create failed with error code: {}", res);
            delete backend;
        } else {
            GENIEX_LOG_INFO("Embedder created successfully");
            *out_handle = reinterpret_cast<geniex_Embedder*>(backend);
        }
        return res;
    } catch (const PluginNotFoundException& e) {
        GENIEX_LOG_ERROR("plugin not found");
        return GENIEX_ERROR_COMMON_PLUGIN_INVALID;
    } catch (const PluginLoadException& e) {
        GENIEX_LOG_ERROR("plugin load error");
        return GENIEX_ERROR_COMMON_PLUGIN_LOAD;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("Exception creating embedder: {}", e.what());
        return GENIEX_ERROR_COMMON_MODEL_LOAD;
    }
}

int32_t geniex_embedder_destroy(geniex_Embedder* h) {
    GENIEX_LOG_INFO("Destroying embedder instance");

    try {
        auto backend = reinterpret_cast<IEmbedding*>(h);
        if (!backend) {
            GENIEX_LOG_ERROR("Attempted to destroy null embedder handle");
            return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        }

        GENIEX_LOG_INFO("Deleting embedder backend instance");
        delete backend;
        GENIEX_LOG_INFO("Embedder destroyed successfully");
        return GENIEX_SUCCESS;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("Exception destroying embedder: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_embedder_embed(
    geniex_Embedder* h, const geniex_EmbedderEmbedInput* input, geniex_EmbedderEmbedOutput* output) {
    GENIEX_LOG_TRACE("{}", input);

    try {
        if (!input || !output) {
            GENIEX_LOG_ERROR("Invalid input or output parameters for embedding");
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        // --- API-level validation: which modalities are present? ---
        const bool has_text =
            (input->texts && input->text_count > 0) || (input->input_ids_2d && input->input_ids_row_count > 0);
        const bool has_image = (input->image_paths && input->image_count > 0);
        const bool has_video = (input->video_paths && input->video_count > 0);

        if (!has_text && !has_image && !has_video) {
            GENIEX_LOG_ERROR("Embedding input must provide at least one text/token, image, or video");
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        auto backend = reinterpret_cast<IEmbedding*>(h);
        if (!backend) {
            GENIEX_LOG_ERROR("Embedder backend is null");
            return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        }

        GENIEX_LOG_INFO("Calling backend embed implementation");
        int32_t result = backend->embed(input, output);

        if (result == GENIEX_SUCCESS) {
            GENIEX_LOG_INFO(
                "Embedding operation completed successfully, generated {} embeddings", output->embedding_count);
        } else {
            GENIEX_LOG_ERROR("Embedding operation failed with error code: {}", result);
        }

        calculate_profile_data(output->profile_data);
        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("Exception in embedder embed: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}

int32_t geniex_embedder_embedding_dim(const geniex_Embedder* h, geniex_EmbedderDimOutput* output) {
    GENIEX_LOG_INFO("Getting embedding dimension");

    try {
        if (!output) {
            GENIEX_LOG_ERROR("Invalid output parameter for embedding dimension");
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        auto backend = reinterpret_cast<IEmbedding*>(const_cast<geniex_Embedder*>(h));
        if (!backend) {
            GENIEX_LOG_ERROR("Embedder backend is null");
            return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
        }

        int32_t result = backend->embedding_dim(output);

        if (result == GENIEX_SUCCESS) {
            GENIEX_LOG_INFO("Embedding dimension retrieved successfully: {}", output->dimension);
        } else {
            GENIEX_LOG_ERROR("Failed to get embedding dimension, error code: {}", result);
        }

        return result;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("Exception getting embedding dimension: {}", e.what());
        return GENIEX_ERROR_COMMON_UNKNOWN;
    }
}
