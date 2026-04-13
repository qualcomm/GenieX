#include "llm.h"

#include <cstring>
#include <memory>
#include <string>
#include <string_view>

#include "common.h"
#include "logging.h"
#include "ml.h"

namespace geniex {

// =============================================================================
// QnnLlm Factory Implementation
// =============================================================================

QnnLlm::QnnLlm(std::string lib_path) : m_model_impl(), m_lib_path(std::move(lib_path)) {}

QnnLlm::~QnnLlm() { GENIEX_LOG_TRACE("QnnLlm destructor"); }

/**
 * @brief Create platform-specific LLM model implementation
 *
 * Android: qwen3-4b, phi3.5, phi4, llama3-3b, granite4, granite4-nano, liquid-v2.
 * Windows: qwen3, llama3, phi, granite, liquid, smolvla, hy-mt.
 * Linux: liquid-v2, granite4-nano.
 *
 * @param input Model creation input with model name and config
 * @return ML_SUCCESS or error code
 */
int32_t QnnLlm::create_impl(const ml_LlmCreateInput* input) {
    if (!input || !input->model_name) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    GENIEX_LOG_TRACE("Creating specific npu LLM model for: {}", fmt::to_string(input->model_name));

    std::string_view model_name(input->model_name);

#if defined(__ANDROID__)
    // Android: supported LLM models
    if (model_name == "qwen3-4b") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_qwen3_4b());
    } else if (model_name == "llama3-3b") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_llama3_3b());
    } else if (model_name == "phi4") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_phi4());
    } else if (model_name == "phi3.5") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_phi3_5());
    } else if (model_name == "granite4") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_granite4());
    } else if (model_name == "granite4-nano") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_granite_nano());
    } else if (model_name == "liquid-v2") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_liquid());
    } else {
        GENIEX_LOG_ERROR("Unsupported model name for Android npu LLM implementation: {}", model_name);
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

#elif defined(_WIN32)
    // Windows: all models supported
    if (model_name == "qwen3-4b") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_qwen3_4b());
    } else if (model_name == "qwen3-8b") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_qwen3_8b());
    } else if (model_name == "llama3-3b") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_llama3_3b());
    } else if (model_name == "phi4") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_phi4());
    } else if (model_name == "phi3.5") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_phi3_5());
    } else if (model_name == "granite4") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_granite4());
    } else if (model_name == "granite4-nano") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_granite_nano());
    } else if (model_name == "liquid-v2") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_liquid());
    } else if (model_name == "smolvla") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_smolvla());
    } else if (model_name == "ministral3-3b") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_ministral3_3b());
    } else if (model_name == "hy-mt") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_hymt());
    } else {
        GENIEX_LOG_ERROR("Unsupported model name for npu LLM implementation: {}", model_name);
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

#elif defined(__linux__)
    // Linux: only liquid-v2 and granite-nano are available in libs
    if (model_name == "liquid-v2") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_liquid());
    } else if (model_name == "granite4-nano") {
        m_model_impl = std::unique_ptr<ILlm>(create_qnn_granite_nano());
    } else {
        GENIEX_LOG_ERROR(
            "Unsupported model name for Linux npu LLM implementation: {}. Available models: liquid-v2, granite-nano",
            model_name);
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

#endif

    if (!m_model_impl) {
        GENIEX_LOG_ERROR("Failed to create specific model implementation");
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

    // inject qnn path, use class for c_str lifetime management
    QnnFolderPathFiller _(input, m_lib_path);
    return m_model_impl->create(input);
}

/**
 * @brief Reset LLM model state and clear KV cache
 *
 * @return ML_SUCCESS or error code
 */
int32_t QnnLlm::reset() {
    if (!m_model_impl) {
        return ML_ERROR_COMMON_NOT_INITIALIZED;
    }

    return m_model_impl->reset();
}

/**
 * @brief Save KV cache state to file
 *
 * @param input Cache save configuration
 * @param output Save operation result
 * @return ML_SUCCESS or error code
 */
int32_t QnnLlm::save_kv_cache(const ml_KvCacheSaveInput* input, ml_KvCacheSaveOutput* output) {
    if (!m_model_impl) {
        return ML_ERROR_COMMON_NOT_INITIALIZED;
    }

    return m_model_impl->save_kv_cache(input, output);
}

/**
 * @brief Load KV cache state from file
 *
 * @param input Cache load configuration
 * @param output Load operation result
 * @return ML_SUCCESS or error code
 */
int32_t QnnLlm::load_kv_cache(const ml_KvCacheLoadInput* input, ml_KvCacheLoadOutput* output) {
    if (!m_model_impl) {
        return ML_ERROR_COMMON_NOT_INITIALIZED;
    }

    return m_model_impl->load_kv_cache(input, output);
}

/**
 * @brief Extract last message content (minimal template formatting)
 *
 * @param input Chat messages
 * @param output Last message content
 * @return ML_SUCCESS or error code
 */
int32_t QnnLlm::apply_chat_template(const ml_LlmApplyChatTemplateInput* input, ml_LlmApplyChatTemplateOutput* output) {
    if (!m_model_impl) {
        return ML_ERROR_COMMON_NOT_INITIALIZED;
    }

    return m_model_impl->apply_chat_template(input, output);
}

/**
 * @brief Generate text completion from LLM
 *
 * @param input Generation prompt and configuration
 * @param output Generated text response
 * @return ML_SUCCESS or error code
 */
int32_t QnnLlm::generate(const ml_LlmGenerateInput* input, ml_LlmGenerateOutput* output) {
    if (!m_model_impl) {
        return ML_ERROR_COMMON_NOT_INITIALIZED;
    }

    return m_model_impl->generate(input, output);
}

}  // namespace geniex
