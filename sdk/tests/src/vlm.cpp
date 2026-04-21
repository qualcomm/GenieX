#include <cstdlib>  // For std::getenv
#include <cstring>
#include <iostream>
#include <optional>
#include <tuple>

#include "doctest.h"
#include "external/json.hpp"
#include "geniex.h"
#include "logging.h"
#include "utf8.h"  // IWYU pragma: export
#include "util.h"

namespace {

#define PLUGINS(M) M(llama_cpp) M(qairt)
using Param = std::tuple<std::string, std::string, std::string, std::optional<std::string>, std::optional<std::string>>;

Setup<Param, geniex_VLM> setup_guard(
    SetupMap<Param>{
        {llama_cpp::value,
            {
                {"SmolVLM-256M-Instruct-Q8",
                    "SmolVLM-256M-Q8",
                    "modelfiles/llama_cpp/SmolVLM-256M-Instruct-Q8_0.gguf",
                    "modelfiles/llama_cpp/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
                    std::nullopt},
                // Add more llama_cpp models here as needed
            }},
        {qairt::value,
            {
#if defined(__ANDROID__)
                {"OmniNeural-4B",
                    "omni-neural",
                    "/data/local/tmp/geniex/modelfiles/OmniNeural-4B/weights-1-8.nexa",
                    std::nullopt,
                    std::nullopt},
                {"AutoNeural",
                    "autoneural",
                    "/data/local/tmp/geniex/modelfiles/AutoNeural/weights-1-3.nexa",
                    std::nullopt,
                    std::nullopt},
                {"Qwen3-VL-4B-Instruct-NPU",
                    "qwen3vl",
                    "/data/local/tmp/geniex/modelfiles/Qwen3-VL-4B-Instruct-NPU/weights-1-4.nexa",
                    std::nullopt,
                    std::nullopt},
#elif defined(_WIN32)
                {"OmniNeural-4B",
                    "omni-neural",
                    "modelfiles/qairt/OmniNeural-4B/weights-1-8.nexa",
                    std::nullopt,
                    std::nullopt},
                {"Qwen3-VL-4B-Instruct-NPU",
                    "qwen3vl",
                    "modelfiles/qairt/Qwen3-VL-4B-Instruct-NPU/weights-1-4.nexa",
                    std::nullopt,
                    std::nullopt},
                {"AutoNeural",
                    "auto-neural",
                    "modelfiles/qairt/AutoNeural/weights-1-3.nexa",
                    std::nullopt,
                    std::nullopt},
#elif defined(__linux__)
                {"AutoNeural",
                    "auto-neural",
                    "modelfiles/qairt/AutoNeural/weights-1-3.nexa",
                    std::nullopt,
                    std::nullopt},
#endif
            }},
    },
    [](geniex_PluginId plugin, Param param) {
        geniex_VLM* vlm = nullptr;

        auto [test_id, name, model, mmproj, tokenizer] = std::move(param);

        // Check if model file exists
        if (!std::filesystem::exists(model)) {
            GENIEX_LOG_WARN("Model file not found: {}", model);
            GENIEX_LOG_WARN("Skipping tests for model: {}", name);
            g_test_summary.add_skipped_model(name, model);
            return static_cast<geniex_VLM*>(nullptr);  // Return nullptr to indicate skip
        }

        geniex_VlmCreateInput input{};
        input.model_name       = name.c_str();
        input.model_path       = model.c_str();
        input.mmproj_path      = mmproj.has_value() ? mmproj.value().c_str() : nullptr;
        input.tokenizer_path   = tokenizer.has_value() ? tokenizer.value().c_str() : nullptr;
        input.config.n_seq_max = 64;
        input.plugin_id        = plugin;
        input.config.n_ctx     = 512;

        int32_t res = geniex_vlm_create(&input, &vlm);
        CHECK_ML_ERROR(res);
        REQUIRE(vlm != nullptr);

        return vlm;
    },
    geniex_vlm_reset, geniex_vlm_destroy);

std::string test_prompt = " 🥳 🎂 Once upon a time";

// Global state to track UTF-8 validation
static size_t g_total_tokens_checked        = 0;
static size_t g_invalid_tokens_count        = 0;
static bool   g_last_token_was_invalid_utf8 = false;

// Stream callback function
bool stream_callback(const char* token, void* _) {
    std::cout << token << std::flush;  // Print tokens on the same line

    // Check UTF-8 validity and record state
    bool is_valid = utf8::is_valid(token, token + std::strlen(token));

    if (!is_valid) {
        g_invalid_tokens_count++;
        g_last_token_was_invalid_utf8 = true;
    } else {
        g_last_token_was_invalid_utf8 = false;
    }

    g_total_tokens_checked++;

    return true;  // Continue streaming
}

// Helper function to validate and reset UTF-8 state
//
// UTF-8 Validation Mechanism:
// The stream_callback tracks UTF-8 validity for each token as it's streamed. This check ensures
// that the model outputs valid UTF-8 text, which is critical for proper text handling in applications.
//
// Why allow invalid UTF-8 in the last token:
// When generation stops (for any reason: max_tokens, EOS, stop sequences, etc.), the streaming buffer
// may flush an incomplete multi-byte UTF-8 sequence as the final token. This is expected behavior,
// as UTF-8 characters can span 1-4 bytes, and truncation at an arbitrary point may split a character.
//
// Validation logic:
// - PASS: All tokens are valid UTF-8
// - PASS: Only the last token is invalid UTF-8 (incomplete sequence at generation end)
// - FAIL: Multiple tokens are invalid, or invalid tokens appear in non-final positions
void validate_and_reset_utf8_state(const geniex_VlmGenerateOutput* output = nullptr) {
    CHECK(g_total_tokens_checked > 0);  // Ensure we actually checked some tokens

    if (g_invalid_tokens_count > 0) {
        // Allow invalid UTF-8 only if it's just the last token
        // (handles incomplete multi-byte sequences when generation stops)
        bool acceptable_invalid_utf8 = (g_invalid_tokens_count == 1) && g_last_token_was_invalid_utf8;

        if (!acceptable_invalid_utf8) {
            GENIEX_LOG_ERROR("Found {} invalid UTF-8 token(s) out of {} total tokens",
                g_invalid_tokens_count,
                g_total_tokens_checked);
            GENIEX_LOG_ERROR("Invalid UTF-8 found in non-final tokens - this indicates a problem");
            CHECK(false);
        } else {
            GENIEX_LOG_INFO("Allowing invalid UTF-8 in last token (incomplete sequence at generation end)");
        }
    }

    // Reset state for next test
    g_total_tokens_checked        = 0;
    g_invalid_tokens_count        = 0;
    g_last_token_was_invalid_utf8 = false;
}

// Test function definitions
void test_generate_basic(geniex_VLM* vlm, const std::string& model_name) {
    geniex_GenerationConfig cfg{};
    cfg.max_tokens  = 32;
    cfg.image_paths = nullptr;
    cfg.image_count = 0;
    cfg.audio_paths = nullptr;
    cfg.audio_count = 0;

    geniex_VlmGenerateInput input{};
    input.prompt_utf8 = test_prompt.c_str();
    input.config      = &cfg;
    input.on_token    = nullptr;
    input.user_data   = nullptr;

    geniex_VlmGenerateOutput output{};

    int32_t res = geniex_vlm_generate(vlm, &input, &output);
    CHECK_ML_ERROR(res);
    GENIEX_LOG_INFO("Output: {}", output);
    if (output.full_text) {
        free(output.full_text);
    }
}

void test_apply_chat_template(geniex_VLM* vlm, const std::string& model_name) {
    auto* contents = new geniex_VlmContent[2]{
        {"text", strdup(test_prompt.c_str())}, {"image", strdup("modelfiles/assets/test_image.png")}};

    geniex_VlmChatMessage message{};
    message.contents      = contents;
    message.content_count = 2;
    message.role          = "user";

    geniex_VlmApplyChatTemplateInput  apply_input{};
    geniex_VlmApplyChatTemplateOutput apply_output{};
    apply_input.messages      = &message;
    apply_input.message_count = 1;
    apply_input.tools         = nullptr;

    int32_t res = geniex_vlm_apply_chat_template(vlm, &apply_input, &apply_output);
    CHECK_ML_ERROR(res);

    // Cleanup
    free(const_cast<char*>(contents[0].text));
    free(const_cast<char*>(contents[1].text));
    delete[] contents;
}

void test_generate_multi_round(geniex_VLM* vlm, const std::string& model_name) {
    std::string user_prompt;
    if (model_name.find("DeepSeek-OCR") != std::string::npos) {
        user_prompt = "Free OCR.";
    } else {
        user_prompt = "Describe the image.";
    }
    std::vector<std::vector<geniex_VlmContent>> user_rounds = {
        {{"text", user_prompt.c_str()}, {"image", "modelfiles/assets/test_image.png"}},
    };

    std::vector<geniex_VlmChatMessage> history;
    geniex_GenerationConfig            cfg{};
    cfg.max_tokens                              = 512;
    cfg.audio_paths                             = nullptr;
    cfg.audio_count                             = 0;
    static std::vector<geniex_Path> image_paths = {"modelfiles/assets/test_image.png"};
    cfg.image_paths                             = image_paths.data();
    cfg.image_count                             = image_paths.size();
    cfg.image_max_length                        = 768;

    for (size_t i = 0; i < user_rounds.size(); ++i) {
        history.push_back({"user",
            const_cast<geniex_VlmContent*>(user_rounds[i].data()),
            static_cast<int64_t>(user_rounds[i].size())});

        geniex_VlmApplyChatTemplateInput input{
            const_cast<geniex_VlmChatMessage*>(history.data()), static_cast<int>(history.size()), nullptr, false};
        geniex_VlmApplyChatTemplateOutput template_result{};
        int32_t                           res = geniex_vlm_apply_chat_template(vlm, &input, &template_result);
        CHECK_ML_ERROR(res);
        REQUIRE(template_result.formatted_text != nullptr);

        geniex_VlmGenerateInput  gen_input{template_result.formatted_text, &cfg, stream_callback, nullptr};
        geniex_VlmGenerateOutput output{};
        res = geniex_vlm_generate(vlm, &gen_input, &output);
        CHECK_ML_ERROR(res);
        CHECK(template_result.formatted_text != nullptr);

        std::cout << std::endl;  // Add newline after streaming output
        GENIEX_LOG_INFO("Round {} output: {}", i + 1, output);

        geniex_VlmContent assistant_content{};
        assistant_content.type = "text";
        assistant_content.text = strdup(output.full_text);
        history.push_back({"assistant", &assistant_content, 1});

        // if (output.full_text) free(output.full_text);
        if (template_result.formatted_text) free(template_result.formatted_text);
    }

    // Validate that all streamed tokens were valid UTF-8
    validate_and_reset_utf8_state();
}

void test_generate_with_sampling(geniex_VLM* vlm, const std::string& model_name) {
    auto sampler_config               = geniex_SamplerConfig{};
    sampler_config.temperature        = 100.0;
    sampler_config.top_p              = 0.9;
    sampler_config.top_k              = 10;
    sampler_config.min_p              = 0.0;
    sampler_config.repetition_penalty = 1.2;
    sampler_config.presence_penalty   = 1.0;
    sampler_config.frequency_penalty  = 1.0;
    sampler_config.seed               = -1;

    geniex_GenerationConfig cfg{};
    cfg.max_tokens     = 32;
    cfg.sampler_config = &sampler_config;
    cfg.image_paths    = nullptr;
    cfg.image_count    = 0;
    cfg.audio_paths    = nullptr;
    cfg.audio_count    = 0;

    geniex_VlmGenerateInput input{};
    input.prompt_utf8 = test_prompt.c_str();
    input.config      = &cfg;
    input.on_token    = nullptr;
    input.user_data   = nullptr;

    geniex_VlmGenerateOutput output{};

    int32_t res = geniex_vlm_generate(vlm, &input, &output);
    CHECK_ML_ERROR(res);
    GENIEX_LOG_INFO("Output: {}", output);
    if (output.full_text) {
        free(output.full_text);
    }
}

void test_generate_json(geniex_VLM* vlm, const std::string& model_name) {
    auto sampler_config        = geniex_SamplerConfig{};
    sampler_config.enable_json = true;

    geniex_GenerationConfig cfg{};
    cfg.max_tokens     = 32;
    cfg.sampler_config = &sampler_config;
    cfg.image_paths    = nullptr;
    cfg.image_count    = 0;
    cfg.audio_paths    = nullptr;
    cfg.audio_count    = 0;

    geniex_VlmGenerateInput input{};
    input.prompt_utf8 = "Give me a name for a cat.";
    input.config      = &cfg;
    input.on_token    = nullptr;
    input.user_data   = nullptr;

    geniex_VlmGenerateOutput output{};

    int32_t res = geniex_vlm_generate(vlm, &input, &output);
    CHECK_ML_ERROR(res);
    GENIEX_LOG_INFO("Output: {}", output);

    REQUIRE(output.full_text != nullptr);
    auto parse_json = nlohmann::json::parse(output.full_text);
    CHECK(parse_json.size() >= 0);

    free(output.full_text);
}

// Register all VLM tests - conditionally based on plugin capabilities
template <typename PluginType>
void register_vlm_tests(TestRegistry<geniex_VLM>& registry) {
    // Common tests for all plugins
    REGISTER_TEST(registry, GenerateBasic, test_generate_basic(model, model_name););
    REGISTER_TEST(registry, ApplyChatTemplate, test_apply_chat_template(model, model_name););
    REGISTER_TEST(registry, GenerateMultiRound, test_generate_multi_round(model, model_name););
    REGISTER_TEST(registry, GenerateWithSampling, test_generate_with_sampling(model, model_name););

    // QAIRT-specific JSON test
    if (std::is_same_v<PluginType, qairt>) {
        REGISTER_TEST(registry, GenerateJson, test_generate_json(model, model_name););
    }
}

// Generate test cases for all plugins
#define GEN(Plugin) TEST_CASE_FOR_PLUGIN(geniex_VLM, Plugin, setup_guard, register_vlm_tests<Plugin>)
PLUGINS(GEN)
#undef GEN

}  // namespace

TEST_MAIN()
