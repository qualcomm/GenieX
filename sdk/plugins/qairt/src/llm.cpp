#include "llm.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ml.h"
#include "pipeline/llm_pipeline.h"

#if defined(_WIN32)
#include "granite4.h"
#include "phi4.h"
#include "qwen3.h"
#endif

namespace {

char* DupCString(const std::string& value) {
    auto* buffer = static_cast<char*>(std::malloc(value.size() + 1));
    if (!buffer) {
        return nullptr;
    }
    std::memcpy(buffer, value.c_str(), value.size() + 1);
    return buffer;
}

struct ExtractedMessage {
    std::string system_prompt;
    std::string last_user_message;
    bool has_system = false;
};

ExtractedMessage extract_last_user_message(const ml_LlmApplyChatTemplateInput* input) {
    ExtractedMessage result;

    for (int32_t i = 0; i < input->message_count; ++i) {
        const char* role = input->messages[i].role;
        const char* content = input->messages[i].content;

        if (role && std::strcmp(role, "system") == 0 && !result.has_system) {
            result.system_prompt = content ? content : "";
            result.has_system = true;
        } else if (role && std::strcmp(role, "user") == 0) {
            result.last_user_message = content ? content : "";
        }
    }

    return result;
}

class LlmFolderPathFiller {
   public:
    LlmFolderPathFiller(const ml_LlmCreateInput* value, const std::string& preset_lib_path = "") {
        auto* mutable_value = const_cast<ml_LlmCreateInput*>(value);

        if (!mutable_value->config.qnn_model_folder_path && mutable_value->model_path) {
            model_path_ = std::filesystem::path(mutable_value->model_path).parent_path().string();
            mutable_value->config.qnn_model_folder_path = model_path_.c_str();
        }

        if (!mutable_value->config.qnn_lib_folder_path && !preset_lib_path.empty()) {
            lib_path_ = preset_lib_path;
            mutable_value->config.qnn_lib_folder_path = lib_path_.c_str();
        }
    }

   private:
    std::string model_path_;
    std::string lib_path_;
};

}  // namespace

namespace geniex {

class PipelineLlm final : public ILlm {
   public:
    using ModelFactory = std::function<LLMModel()>;

    PipelineLlm(std::string family_name,
                std::string model_subdir,
                ModelFactory model_factory,
                std::vector<std::string> model_files,
                bool needs_embedding)
        : family_name_(std::move(family_name)),
          model_subdir_(std::move(model_subdir)),
          model_factory_(std::move(model_factory)),
          model_files_(std::move(model_files)),
          needs_embedding_(needs_embedding) {}

    int32_t reset() override {
        if (!ready_) {
            return ML_ERROR_COMMON_NOT_INITIALIZED;
        }
        pipeline_.reset();
        return ML_SUCCESS;
    }

    int32_t save_kv_cache(const ml_KvCacheSaveInput* input, ml_KvCacheSaveOutput*) override {
        if (!ready_) {
            return ML_ERROR_COMMON_NOT_INITIALIZED;
        }
        if (!input || !input->path) {
            return ML_ERROR_COMMON_INVALID_INPUT;
        }

        try {
            pipeline_.saveKVCache(input->path);
            return ML_SUCCESS;
        } catch (const std::exception&) {
            return ML_ERROR_COMMON_UNKNOWN;
        }
    }

    int32_t load_kv_cache(const ml_KvCacheLoadInput* input, ml_KvCacheLoadOutput*) override {
        if (!ready_) {
            return ML_ERROR_COMMON_NOT_INITIALIZED;
        }
        if (!input || !input->path) {
            return ML_ERROR_COMMON_INVALID_INPUT;
        }

        try {
            pipeline_.loadKVCache(input->path);
            return ML_SUCCESS;
        } catch (const std::exception&) {
            return ML_ERROR_COMMON_UNKNOWN;
        }
    }

    int32_t apply_chat_template(const ml_LlmApplyChatTemplateInput* input, ml_LlmApplyChatTemplateOutput* output) override {
        if (!ready_) {
            return ML_ERROR_COMMON_NOT_INITIALIZED;
        }
        if (!input || !output || !input->messages || input->message_count <= 0) {
            return ML_ERROR_COMMON_INVALID_INPUT;
        }

        auto extracted = extract_last_user_message(input);

        if (extracted.last_user_message.empty()) {
            return ML_ERROR_COMMON_INVALID_INPUT;
        }

        if (extracted.has_system && pipeline_.nPast() == 0) {
            pipeline_.setSystemPrompt(extracted.system_prompt);
        }

        std::string formatted_prompt = pipeline_.applyChatTemplate(
            extracted.last_user_message,
            input->enable_thinking
        );

        output->formatted_text = DupCString(formatted_prompt);
        if (!output->formatted_text && !formatted_prompt.empty()) {
            return ML_ERROR_COMMON_UNKNOWN;
        }

        return ML_SUCCESS;
    }

    int32_t generate(const ml_LlmGenerateInput* input, ml_LlmGenerateOutput* output) override {
        if (!ready_) {
            return ML_ERROR_COMMON_NOT_INITIALIZED;
        }
        if (!input || !output || !input->prompt_utf8) {
            return ML_ERROR_COMMON_INVALID_INPUT;
        }
        if (input->input_ids && input->input_ids_count > 0) {
            return ML_ERROR_COMMON_INVALID_INPUT;
        }

        GenerationConfig generation_config;
        if (input->config) {
            generation_config.max_tokens = input->config->max_tokens > 0 ? input->config->max_tokens : generation_config.max_tokens;
            if (input->config->sampler_config) {
                generation_config.temperature = input->config->sampler_config->temperature > 0.0f
                                                    ? input->config->sampler_config->temperature
                                                    : generation_config.temperature;
                generation_config.top_p = input->config->sampler_config->top_p > 0.0f
                                              ? input->config->sampler_config->top_p
                                              : generation_config.top_p;
            }
        }
        generation_config.thinking_mode = enable_thinking_;

        try {
            auto result = pipeline_.generate(
                input->prompt_utf8,
                generation_config,
                [input](const char* piece) {
                    return !input->on_token || input->on_token(piece, input->user_data);
                });
            output->full_text = DupCString(result.full_text);
            if (!output->full_text && !result.full_text.empty()) {
                return ML_ERROR_COMMON_UNKNOWN;
            }
            return ML_SUCCESS;
        } catch (const std::exception&) {
            return ML_ERROR_COMMON_UNKNOWN;
        }
    }

   protected:
    int32_t create_impl(const ml_LlmCreateInput* input) override {
        if (!input || !input->config.qnn_model_folder_path || !input->config.qnn_lib_folder_path) {
            return ML_ERROR_COMMON_INVALID_INPUT;
        }

        const std::filesystem::path model_root(input->config.qnn_model_folder_path);
        const std::filesystem::path model_dir = std::filesystem::exists(model_root / model_subdir_)
                                                    ? (model_root / model_subdir_)
                                                    : model_root;
        const std::filesystem::path lib_dir(input->config.qnn_lib_folder_path);

        QnnRuntimeConfig runtime_config;
        runtime_config.backend_path = (lib_dir / "QnnHtp.dll").string();
        runtime_config.system_lib_path = (lib_dir / "QnnSystem.dll").string();
        runtime_config.extensions_path = (lib_dir / "QnnHtpNetRunExtensions.dll").string();

        ModelConfig model_config;
        for (const std::string& file_name : model_files_) {
            model_config.model_paths.push_back((model_dir / file_name).string());
        }
        model_config.tokenizer_path = input->tokenizer_path ? input->tokenizer_path : (model_dir / "tokenizer.json").string();
        model_config.htp_config_path = (model_dir / "htp_backend_ext_config.json").string();
        if (needs_embedding_) {
            model_config.embedding_path = (model_dir / "embed_tokens.npy").string();
        }

        enable_thinking_ = input->config.enable_thinking;
        if (!pipeline_.create(family_name_, model_factory_(), runtime_config, model_config)) {
            return ML_ERROR_COMMON_MODEL_LOAD;
        }
        if (input->config.system_prompt) {
            pipeline_.setSystemPrompt(input->config.system_prompt);
        }
        ready_ = true;
        return ML_SUCCESS;
    }

   private:
    std::string              family_name_;
    std::string              model_subdir_;
    ModelFactory             model_factory_;
    std::vector<std::string> model_files_;
    bool                     needs_embedding_ = false;
    bool                     enable_thinking_ = false;
    bool                     ready_ = false;
    LLMPipeline              pipeline_;
};

QnnLlm::QnnLlm(std::string lib_path) : m_model_impl(), m_lib_path(std::move(lib_path)) {}

QnnLlm::~QnnLlm() = default;

int32_t QnnLlm::create_impl(const ml_LlmCreateInput* input) {
    if (!input || !input->model_name) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    std::string_view model_name(input->model_name);

#if defined(_WIN32)
    if (model_name == "qwen3-4b") {
        m_model_impl = std::make_unique<PipelineLlm>(
            "qwen3",
            "qwen3_4b_instruct_2507_aihub",
            []() { return qwen3_4b_instruct_2507_aihub::makeModel(); },
            std::vector<std::string>{
                "qwen3_4b_instruct_2507_part_1_of_4.bin",
                "qwen3_4b_instruct_2507_part_2_of_4.bin",
                "qwen3_4b_instruct_2507_part_3_of_4.bin",
                "qwen3_4b_instruct_2507_part_4_of_4.bin",
            },
            false);
    } else if (model_name == "qwen3-8b") {
        m_model_impl = std::make_unique<PipelineLlm>(
            "qwen3",
            "qwen3_8b",
            []() { return qwen3_8b::makeModel(); },
            std::vector<std::string>{
                "weight_sharing_model_1_of_4.serialized.bin",
                "weight_sharing_model_2_of_4.serialized.bin",
                "weight_sharing_model_3_of_4.serialized.bin",
                "weight_sharing_model_4_of_4.serialized.bin",
            },
            true);
    } else if (model_name == "phi4") {
        m_model_impl = std::make_unique<PipelineLlm>(
            "phi4",
            "phi4",
            []() { return phi4::makeModel(); },
            std::vector<std::string>{
                "weight_sharing_model_1_of_2.serialized.bin",
                "weight_sharing_model_2_of_2.serialized.bin",
            },
            true);
    } else if (model_name == "granite4") {
        m_model_impl = std::make_unique<PipelineLlm>(
            "granite4",
            "granite4_micro",
            []() { return granite4_micro::makeModel(); },
            std::vector<std::string>{
                "weight_sharing_model_1_of_2.serialized.bin",
                "weight_sharing_model_2_of_2.serialized.bin",
            },
            true);
    } else {
        return ML_ERROR_COMMON_MODEL_LOAD;
    }
#else
    return ML_ERROR_COMMON_MODEL_LOAD;
#endif

    LlmFolderPathFiller _(input, m_lib_path);
    return m_model_impl->create(input);
}

int32_t QnnLlm::reset() {
    if (!m_model_impl) {
        return ML_ERROR_COMMON_NOT_INITIALIZED;
    }
    return m_model_impl->reset();
}

int32_t QnnLlm::save_kv_cache(const ml_KvCacheSaveInput* input, ml_KvCacheSaveOutput* output) {
    if (!m_model_impl) {
        return ML_ERROR_COMMON_NOT_INITIALIZED;
    }
    return m_model_impl->save_kv_cache(input, output);
}

int32_t QnnLlm::load_kv_cache(const ml_KvCacheLoadInput* input, ml_KvCacheLoadOutput* output) {
    if (!m_model_impl) {
        return ML_ERROR_COMMON_NOT_INITIALIZED;
    }
    return m_model_impl->load_kv_cache(input, output);
}

int32_t QnnLlm::apply_chat_template(const ml_LlmApplyChatTemplateInput* input, ml_LlmApplyChatTemplateOutput* output) {
    if (!m_model_impl) {
        return ML_ERROR_COMMON_NOT_INITIALIZED;
    }
    return m_model_impl->apply_chat_template(input, output);
}

int32_t QnnLlm::generate(const ml_LlmGenerateInput* input, ml_LlmGenerateOutput* output) {
    if (!m_model_impl) {
        return ML_ERROR_COMMON_NOT_INITIALIZED;
    }
    return m_model_impl->generate(input, output);
}

ILlm* create_qairt_llm(const char* lib_path) {
    return new QnnLlm(lib_path ? lib_path : "");
}

}  // namespace geniex
