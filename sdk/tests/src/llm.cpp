#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <thread>
#include <utility>

#include "doctest.h"
#include "external/json.hpp"
#include "geniex.h"
#include "logging.h"
#include "utf8.h"  // IWYU pragma: export
#include "util.h"

namespace {

// Declare the list of plugins for this test file using an X-macro list
#define PLUGINS(M) \
    M(llama_cpp)   \
    M(qairt)
using Param = std::tuple<std::string, std::string, std::string, std::optional<std::string>>;

// Plugin Capabilities Configuration
// Add plugin name to set to enable that capability for the plugin
struct PluginCapabilities {
    static inline const std::set<std::string> KV_CACHE  = {"llama_cpp"};
    static inline const std::set<std::string> JSON      = {"qairt"};
    static inline const std::set<std::string> TOOL_CALL = {"llama_cpp"};

    static bool has_kv_cache(const std::string &plugin) { return KV_CACHE.count(plugin) > 0; }
    static bool has_json(const std::string &plugin) { return JSON.count(plugin) > 0; }
    static bool has_tool_call(const std::string &plugin) { return TOOL_CALL.count(plugin) > 0; }
};

Setup<Param, geniex_LLM> setup_guard(
    SetupMap<Param>{
        {llama_cpp::value,
            {
#if defined(__ANDROID__)
                {"granite-4.0-h-350m-Q8_0",
                    "granite-4.0-h-350m-Q8_0",
                    "/data/local/tmp/geniex/modelfiles/llama_cpp/"
                    "granite-4.0-h-350m-Q8_0.gguf",
                    std::nullopt},
                {"Llama-3.2-3B-Instruct-Q4_0",
                    "Llama-3.2-3B-Instruct-Q4_0",
                    "/data/local/tmp/geniex/modelfiles/llama_cpp/"
                    "Llama-3.2-3B-Instruct-Q4_0.gguf",
                    std::nullopt},
                {"Qwen3-4B-Q4_0",
                    "Qwen3-4B-Q4_0",
                    "/data/local/tmp/geniex/modelfiles/llama_cpp/Qwen3-4B-Q4_0.gguf",
                    std::nullopt},
#elif defined(_WIN32) | defined(unix)
                {"Qwen3-0.6B-Q4_0", "Qwen3-0.6B-Q4_0", "modelfiles/llama_cpp/Qwen3-0.6B-Q4_0.gguf", std::nullopt},
#endif
            }},
        {qairt::value,
            {
#if defined(__ANDROID__)
                {"LFM2-1.2B-npu",
                    "liquid-v2",
                    "/data/local/tmp/geniex/modelfiles/LFM2-1.2B-npu/weights-1-2.nexa",
                    std::nullopt},
                {"Granite-4-Micro-NPU",
                    "granite4",
                    "/data/local/tmp/geniex/modelfiles/Granite-4-Micro-NPU/"
                    "weights-1-3.nexa",
                    std::nullopt},
                {"Granite-4.0-h-350M-NPU",
                    "granite4-nano",
                    "/data/local/tmp/geniex/modelfiles/Granite-4.0-h-350M-NPU/"
                    "weights-1-2.nexa",
                    std::nullopt},
                {"Llama3.2-3B-NPU-Turbo",
                    "llama3-3b",
                    "/data/local/tmp/geniex/modelfiles/Llama3.2-3B-NPU-Turbo/"
                    "weights-1-3.nexa",
                    std::nullopt},
                {"phi4-mini-npu-turbo",
                    "phi4",
                    "/data/local/tmp/geniex/modelfiles/phi4-mini-npu-turbo/"
                    "weights-1-3.nexa",
                    std::nullopt},
                {"phi3.5-mini-npu",
                    "phi3.5",
                    "/data/local/tmp/geniex/modelfiles/phi3.5-mini-npu/"
                    "weights-1-3.nexa",
                    std::nullopt},
                {"Qwen3-4B-Instruct-2507-NPU",
                    "qwen3-4b",
                    "/data/local/tmp/geniex/modelfiles/Qwen3-4B-Instruct-2507-NPU/"
                    "weights-1-3.nexa",
                    std::nullopt},
#elif defined(_WIN32)
                {"granite4_micro", "granite4", "modelfiles/qairt/granite4_micro/tokenizer.json", std::nullopt},
                {"phi4", "phi4", "modelfiles/qairt/phi4/tokenizer.json", std::nullopt},
                {"phi3_5_aihub", "phi3.5-aihub", "modelfiles/qairt/phi3_5_aihub/tokenizer.json", std::nullopt},
                {"qwen3_4b_aihub", "qwen3-4b-aihub", "modelfiles/qairt/qwen3_4b_aihub/tokenizer.json", std::nullopt},
                {"qwen3_4b_instruct_2507_aihub",
                    "qwen3-4b",
                    "modelfiles/qairt/qwen3_4b_instruct_2507_aihub/tokenizer.json",
                    std::nullopt},
#elif defined(__linux__)
                {"LFM2-1.2B-npu", "liquid-v2", "modelfiles/qairt/LFM2-1.2B-npu/weights-1-2.nexa", std::nullopt},
#endif
            }},
    },
    [](geniex_PluginId plugin, Param param) {
        geniex_LLM *llm = nullptr;

        auto [test_id, name, model_path, tokenizer_path] = std::move(param);

        // Check if model file exists
        if (!std::filesystem::exists(model_path)) {
            GENIEX_LOG_WARN("Model file not found: {}", model_path);
            GENIEX_LOG_WARN("Skipping tests for model: {}", name);
            g_test_summary.add_skipped_model(name, model_path);
            return static_cast<geniex_LLM *>(nullptr);  // Return nullptr to indicate skip
        }

        geniex_LlmCreateInput input{};
        input.model_name             = name.c_str();
        input.model_path             = model_path.c_str();
        input.tokenizer_path         = tokenizer_path ? tokenizer_path->c_str() : nullptr;
        input.config.n_ctx           = 8192;
        input.config.n_seq_max       = 64;
        input.config.n_gpu_layers    = 0;  // specify n_gpu_layers to 0 to use CPU
        input.config.enable_sampling = false;
        input.plugin_id              = plugin;
        input.device_id              = "CPU";
        int32_t res                  = geniex_llm_create(&input, &llm);
        CHECK_ML_ERROR(res);
        REQUIRE(llm != nullptr);

        return llm;
    },
    geniex_llm_reset, geniex_llm_destroy);

// Global state to track UTF-8 validation
static size_t g_total_tokens_checked        = 0;
static size_t g_invalid_tokens_count        = 0;
static bool   g_last_token_was_invalid_utf8 = false;

// Stream callback function
bool stream_callback(const char *token, void *_) {
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
// The stream_callback tracks UTF-8 validity for each token as it's streamed.
// This check ensures that the model outputs valid UTF-8 text, which is critical
// for proper text handling in applications.
//
// Why allow invalid UTF-8 in the last token:
// When generation stops (for any reason: max_tokens, EOS, stop sequences,
// etc.), the streaming buffer may flush an incomplete multi-byte UTF-8 sequence
// as the final token. This is expected behavior, as UTF-8 characters can span
// 1-4 bytes, and truncation at an arbitrary point may split a character.
//
// Validation logic:
// - PASS: All tokens are valid UTF-8
// - PASS: Only the last token is invalid UTF-8 (incomplete sequence at
// generation end)
// - FAIL: Multiple tokens are invalid, or invalid tokens appear in non-final
// positions
void validate_and_reset_utf8_state(const geniex_LlmGenerateOutput *output = nullptr) {
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
            GENIEX_LOG_INFO(
                "Allowing invalid UTF-8 in last token (incomplete sequence "
                "at generation end)");
        }
    }

    // Reset state for next test
    g_total_tokens_checked        = 0;
    g_invalid_tokens_count        = 0;
    g_last_token_was_invalid_utf8 = false;
}

// Test Definitions and Registration
// All tests are defined inline with REGISTER_TEST macro in registration
// functions below

// Register all LLM tests - conditionally based on plugin capabilities
template <typename PluginType>
void register_llm_tests(TestRegistry<geniex_LLM> &registry) {
    std::string plugin_name = PluginType::value;

    // Common tests for all plugins
    REGISTER_TEST(registry, GenerateBasic, auto cfg = geniex_GenerationConfig{}; cfg.max_tokens = 32;
        geniex_LlmGenerateInput input{};
        input.prompt_utf8 = " 🥳 🎂 Once upon a time";
        input.config      = &cfg;
        geniex_LlmGenerateOutput output{};
        int32_t                  res = geniex_llm_generate(model, &input, &output);
        CHECK_ML_ERROR(res);
        GENIEX_LOG_INFO("{}", output);

        REQUIRE(output.full_text != nullptr);
        CHECK(utf8::is_valid(output.full_text, output.full_text + std::strlen(output.full_text)));
        free(output.full_text);

        CHECK(output.profile_data.prompt_tokens > 0);
        CHECK(output.profile_data.generated_tokens > 0););

    //     // Test reading input from file (Android NPU only)
    // #if defined(__ANDROID__)
    //     if (plugin_name == "qairt") {
    //         REGISTER_TEST(registry, GenerateFromFile,
    //             // Read prompt from test.txt file
    //             std::string test_file_path =
    //             "/data/local/tmp/geniex/modelfiles/test.txt"; std::ifstream
    //             test_file(test_file_path); REQUIRE(test_file.is_open());

    //             std::string
    //             file_content((std::istreambuf_iterator<char>(test_file)),
    //                                       std::istreambuf_iterator<char>());
    //             test_file.close();

    //             REQUIRE(!file_content.empty());
    //             GENIEX_LOG_INFO("Read prompt from file: {}", file_content);

    //             // Generate with file content as prompt
    //             auto cfg = geniex_GenerationConfig{};
    //             cfg.max_tokens = 128;
    //             geniex_LlmGenerateInput input{};
    //             input.prompt_utf8 = file_content.c_str();
    //             input.config      = &cfg;
    //             geniex_LlmGenerateOutput output{};
    //             int32_t              res = geniex_llm_generate(model, &input,
    //             &output); CHECK_ML_ERROR(res); GENIEX_LOG_INFO("Generated output:
    //             {}", output);

    //             REQUIRE(output.full_text != nullptr);
    //             CHECK(utf8::is_valid(output.full_text, output.full_text +
    //             std::strlen(output.full_text))); free(output.full_text);
    //             GENIEX_LOG_INFO("Generated output: {}", output.full_text);
    //             // print profiling data, ttft, prompt_time, decode_time,
    //             decoding_speed, prompt_tokens, generated_tokens
    //             GENIEX_LOG_INFO("Profiling data: ttft: {}, prompt_time: {},
    //             decode_time: {}, prefill_speed: {}, decoding_speed: {},
    //             prompt_tokens: {}, generated_tokens: {}",
    //                 output.profile_data.ttft,
    //                 output.profile_data.prompt_time,
    //                 output.profile_data.decode_time,
    //                 output.profile_data.prefill_speed,
    //                 output.profile_data.decoding_speed,
    //                 output.profile_data.prompt_tokens,
    //                 output.profile_data.generated_tokens);

    //             CHECK(output.profile_data.prompt_tokens > 0);
    //             CHECK(output.profile_data.generated_tokens > 0););
    //     }
    // #endif

    REGISTER_TEST(registry, GenerateStream, auto cfg = geniex_GenerationConfig{}; cfg.max_tokens = 32;
        geniex_LlmGenerateInput input{};
        input.prompt_utf8 = " 🥳 🎂 Once upon a time";
        input.config      = &cfg;
        input.on_token    = stream_callback;
        geniex_LlmGenerateOutput output{};

        int32_t res = geniex_llm_generate(model, &input, &output);
        CHECK_ML_ERROR(res);

        std::cout << std::endl;  // Add newline after streaming output

        // Validate that all streamed tokens were valid UTF-8
        validate_and_reset_utf8_state(&output););

    REGISTER_TEST(registry, GenerateWithTokenIds, auto cfg = geniex_GenerationConfig{}; cfg.max_tokens = 32;
        geniex_LlmGenerateInput input{};
        // token id in this model's tokenizer for ` 🥳 🎂 Once upon a time`
        std::vector<int32_t> token_ids = {11162, 98, 111, 11162, 236, 224, 9646, 5193, 264, 882};
        input.input_ids                = token_ids.data();
        input.input_ids_count          = (int32_t)token_ids.size();
        input.config                   = &cfg;
        geniex_LlmGenerateOutput output{};
        int32_t                  res = geniex_llm_generate(model, &input, &output);
        CHECK_ML_ERROR(res);
        GENIEX_LOG_INFO("{}", output);

        REQUIRE(output.full_text != nullptr);
        CHECK(utf8::is_valid(output.full_text, output.full_text + std::strlen(output.full_text)));
        free(output.full_text);

        CHECK(output.profile_data.prompt_tokens > 0);
        CHECK(output.profile_data.generated_tokens > 0););

    REGISTER_TEST(registry, GenerateChat, std::vector<geniex_LlmChatMessage> messages;
        messages.push_back({"user", " 🥳 🎂 Once upon a time"});

        geniex_LlmApplyChatTemplateInput apply_input{};
        apply_input.messages              = messages.data();
        apply_input.message_count         = (int32_t)messages.size();
        apply_input.add_generation_prompt = true;
        geniex_LlmApplyChatTemplateOutput apply_output{};
        int32_t apply_res = geniex_llm_apply_chat_template(model, &apply_input, &apply_output);
        REQUIRE_ML_ERROR(apply_res);

        auto cfg       = geniex_GenerationConfig{};
        cfg.max_tokens = 32;
        geniex_LlmGenerateInput input{};
        input.prompt_utf8 = apply_output.formatted_text;
        input.config      = &cfg;
        input.on_token    = stream_callback;
        input.user_data   = nullptr;
        geniex_LlmGenerateOutput output{};

        int32_t res = geniex_llm_generate(model, &input, &output);
        CHECK_ML_ERROR(res);

        std::cout << std::endl;  // Add newline after streaming output

        REQUIRE(output.full_text != nullptr);
        GENIEX_LOG_INFO("Output from basic generation: {}", output.full_text);
        CHECK(utf8::is_valid(output.full_text, output.full_text + std::strlen(output.full_text)));
        if (apply_output.formatted_text) free(apply_output.formatted_text);
        free(output.full_text);

        // Validate that all streamed tokens were valid UTF-8
        validate_and_reset_utf8_state(););

    REGISTER_TEST(
        registry, GenerateChatMultiRound, struct Rounds { std::string user_message; };

        std::vector<Rounds> turns = {{"let a = 1; let b = 2;"}, {"what is the value of a + b?"}};

        std::vector<geniex_LlmChatMessage>
                                history;
        geniex_GenerationConfig cfg{};
        cfg.max_tokens = 64;

        for (size_t i = 0; i < turns.size(); ++i) {
            history.push_back({"user", turns[i].user_message.c_str()});

            geniex_LlmApplyChatTemplateInput apply_input{};
            apply_input.messages              = history.data();
            apply_input.message_count         = static_cast<int32_t>(history.size());
            apply_input.add_generation_prompt = true;

            geniex_LlmApplyChatTemplateOutput apply_output{};
            int32_t apply_res = geniex_llm_apply_chat_template(model, &apply_input, &apply_output);
            CHECK_ML_ERROR(apply_res);
            REQUIRE(apply_output.formatted_text != nullptr);

            geniex_LlmGenerateInput gen_input{};
            gen_input.prompt_utf8 = apply_output.formatted_text;
            gen_input.config      = &cfg;
            gen_input.on_token    = stream_callback;
            gen_input.user_data   = nullptr;

            geniex_LlmGenerateOutput output{};
            int32_t                  res = geniex_llm_generate(model, &gen_input, &output);
            CHECK_ML_ERROR(res);

            std::cout << std::endl;  // Add newline after streaming output
            GENIEX_LOG_INFO("Turn {} reply: {}", i + 1, output.full_text);

            if (output.full_text) {
                history.push_back({"assistant", output.full_text});
            }

            if (apply_output.formatted_text) free(apply_output.formatted_text);
        }

        // Validate that all streamed tokens were valid UTF-8
        validate_and_reset_utf8_state(););

    REGISTER_TEST(registry, GenerateWithSampling, auto sampler_config = geniex_SamplerConfig{};
        sampler_config.temperature        = 100.0;
        sampler_config.top_p              = 0.9;
        sampler_config.top_k              = 10;
        sampler_config.min_p              = 0.0;
        sampler_config.repetition_penalty = 1.2;
        sampler_config.presence_penalty   = 1.0;
        sampler_config.frequency_penalty  = 1.0;
        sampler_config.seed               = 42;

        auto cfg           = geniex_GenerationConfig{};
        cfg.max_tokens     = 32;
        cfg.sampler_config = &sampler_config;

        geniex_LlmGenerateInput input{};
        input.prompt_utf8 = " 🥳 🎂 Once upon a time";
        input.config      = &cfg;
        geniex_LlmGenerateOutput output{};

        int32_t res = geniex_llm_generate(model, &input, &output);
        CHECK_ML_ERROR(res);
        GENIEX_LOG_INFO("{}", output);

        REQUIRE(output.full_text != nullptr);
        CHECK(utf8::is_valid(output.full_text, output.full_text + std::strlen(output.full_text)));
        free(output.full_text);

        CHECK(output.profile_data.prompt_tokens > 0);
        CHECK(output.profile_data.generated_tokens > 0););

    REGISTER_TEST(
        registry,
        Threading,
        // Create and start a single worker thread
        auto thread_worker =
            [](geniex_LLM *llm) {
                auto cfg       = geniex_GenerationConfig{};
                cfg.max_tokens = 32;
                geniex_LlmGenerateInput input{};
                input.prompt_utf8 = " 🥳 🎂 Once upon a time";
                input.config      = &cfg;
                geniex_LlmGenerateOutput output{};
                int32_t                  res = geniex_llm_generate(llm, &input, &output);
                CHECK_ML_ERROR(res);

                REQUIRE(output.full_text != nullptr);
                GENIEX_LOG_INFO("Output from basic generation: {}", output.full_text);
                CHECK(utf8::is_valid(output.full_text, output.full_text + std::strlen(output.full_text)));
                free(output.full_text);
            };
        std::thread worker_thread(thread_worker, model);

        // Wait for the worker thread to complete
        if (worker_thread.joinable()) {
            worker_thread.join();
            // If we get here without crashes, threading test passed
            CHECK(true);
        });

    // Plugin-specific tests - conditionally registered based on capabilities
    if (PluginCapabilities::has_kv_cache(plugin_name)) {
        REGISTER_TEST(
            registry,
            KVCacheSave,
            // Create unique cache path for this model
            std::string cache_path = "./kvcache_" + model_name;

            // First run a generation to populate the KV cache
            auto cfg       = geniex_GenerationConfig{};
            cfg.max_tokens = 32;
            geniex_LlmGenerateInput input{};
            input.prompt_utf8 = " 🥳 🎂 Once upon a time";
            input.config      = &cfg;
            geniex_LlmGenerateOutput output{};
            int32_t                  gen_res = geniex_llm_generate(model, &input, &output);
            REQUIRE_ML_ERROR(gen_res);

            if (output.full_text) { free(output.full_text); }

            geniex_KvCacheSaveInput save_input{};
            save_input.path = cache_path.c_str();
            geniex_KvCacheSaveOutput save_output{};
            int32_t                  res = geniex_llm_save_kv_cache(model, &save_input, &save_output);
            CHECK_ML_ERROR(res);

            std::filesystem::remove(cache_path););

        REGISTER_TEST(
            registry,
            KVCacheLoad,
            // Create unique cache path for this model
            std::string cache_path = "./kvcache_" + model_name;

            // First run a generation to populate the KV cache
            auto cfg       = geniex_GenerationConfig{};
            cfg.max_tokens = 32;
            geniex_LlmGenerateInput input{};
            input.prompt_utf8 = " 🥳 🎂 Once upon a time";
            input.config      = &cfg;
            geniex_LlmGenerateOutput output{};
            int32_t                  gen_res = geniex_llm_generate(model, &input, &output);
            REQUIRE_ML_ERROR(gen_res);

            if (output.full_text) { free(output.full_text); }

            // First save, then load
            geniex_KvCacheSaveInput save_input{};
            save_input.path = cache_path.c_str();
            geniex_KvCacheSaveOutput save_output{};
            int32_t                  save_res = geniex_llm_save_kv_cache(model, &save_input, &save_output);
            REQUIRE_ML_ERROR(save_res);

            geniex_llm_reset(model);

            geniex_KvCacheLoadInput load_input{};
            load_input.path = cache_path.c_str();
            geniex_KvCacheLoadOutput load_output{};
            int32_t                  load_res = geniex_llm_load_kv_cache(model, &load_input, &load_output);
            CHECK_ML_ERROR(load_res);

            std::filesystem::remove(cache_path););
    }

    if (PluginCapabilities::has_tool_call(plugin_name)) {
        REGISTER_TEST(registry, ToolCallBasic, geniex_LlmChatMessage message; message.role = "user";
            message.content = "What is the weather in San Francisco?";

            geniex_LlmApplyChatTemplateInput input{};
            input.messages      = &message;
            input.message_count = 1;
            input.tools         = R"([{
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }
            }])";

            geniex_LlmApplyChatTemplateOutput output{};
            int32_t                           res = geniex_llm_apply_chat_template(model, &input, &output);
            CHECK_ML_ERROR(res);

            REQUIRE(output.formatted_text != nullptr);
            GENIEX_LOG_INFO("{}", output);

            auto cfg       = geniex_GenerationConfig{};
            cfg.max_tokens = 32;
            geniex_LlmGenerateInput gen_input{};
            gen_input.prompt_utf8 = output.formatted_text;
            gen_input.config      = &cfg;
            geniex_LlmGenerateOutput gen_output{};
            int32_t                  gen_res = geniex_llm_generate(model, &gen_input, &gen_output);
            CHECK_ML_ERROR(gen_res);
            GENIEX_LOG_INFO("{}", gen_output);

            CHECK(gen_output.full_text != nullptr);
            if (gen_output.full_text) free(gen_output.full_text);
            free(output.formatted_text););
    }

    if (PluginCapabilities::has_json(plugin_name)) {
        REGISTER_TEST(registry, GenerateJson, auto sampler_config = geniex_SamplerConfig{};
            sampler_config.enable_json = true;

            auto cfg = geniex_GenerationConfig{};
            // Using a larger max_tokens to allow the model to generate a complete
            // json
            cfg.max_tokens     = 64;
            cfg.sampler_config = &sampler_config;

            geniex_LlmGenerateInput input{};
            input.prompt_utf8 =
                "Generate information of a cat, including only the "
                "following information: name, breed, age, color";
            input.config = &cfg;
            geniex_LlmGenerateOutput output{};

            int32_t res = geniex_llm_generate(model, &input, &output);
            CHECK_ML_ERROR(res);
            GENIEX_LOG_INFO("{}", output);

            REQUIRE(output.full_text != nullptr);
            CHECK(utf8::is_valid(output.full_text, output.full_text + std::strlen(output.full_text)));

            auto parse_json = nlohmann::json::parse(output.full_text);
            CHECK(parse_json.size() >= 0);

            free(output.full_text););
    }
}

// Generate test cases for all plugins using the PLUGINS X-macro list
#define GEN(Plugin) TEST_CASE_FOR_PLUGIN(geniex_LLM, Plugin, setup_guard, register_llm_tests<Plugin>)
PLUGINS(GEN)
#undef GEN

}  // namespace

TEST_MAIN()
