#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>

#include "doctest.h"
#include "geniex.h"
#include "logging.h"
#include "utf8.h"  // IWYU pragma: export
#include "util.h"

namespace {

#define PLUGINS(M)
using Param = std::tuple<std::string, std::string, std::string, std::optional<std::string>>;

Setup<Param, geniex_TTS> setup_guard(
    SetupMap<Param>{},
    [](geniex_PluginId plugin, Param param) {
        geniex_TTS *tts = nullptr;

        auto [test_id, model_name, model_path, vocoder] = std::move(param);

        // Check if model file exists
        if (!std::filesystem::exists(model_path)) {
            GENIEX_LOG_WARN("Model file not found: {}", model_path);
            GENIEX_LOG_WARN("Skipping tests for model: {}", model_name);
            g_test_summary.add_skipped_model(model_name, model_path);
            return static_cast<geniex_TTS *>(nullptr);  // Return nullptr to indicate skip
        }

        geniex_TtsCreateInput input{};
        input.model_name   = model_name.c_str();
        input.model_path   = model_path.c_str();
        input.vocoder_path = vocoder.has_value() ? vocoder.value().c_str() : "";
        input.device_id    = nullptr;
        input.plugin_id    = plugin;

        int32_t res = geniex_tts_create(&input, &tts);
        CHECK_ML_ERROR(res);
        REQUIRE(tts != nullptr);

        return tts;
    },
    nullptr, geniex_tts_destroy);

std::string test_text = "Hello world! This is a test of the text to speech system.";

// Test function definitions
void test_tts_synthesize_basic(geniex_TTS *tts, const std::string &model_name) {
    // Ensure output directory exists
    std::filesystem::create_directories("output");

    // Set up TTS configuration
    geniex_TTSConfig config{};
    config.voice       = "af_heart";
    config.speed       = 1.0f;
    config.seed        = 42;
    config.sample_rate = 24000;

    geniex_TtsSynthesizeInput input{};
    input.text_utf8   = test_text.c_str();
    input.config      = &config;
    input.output_path = "output/tts_test_basic.wav";  // Generate audio under output/ directory

    geniex_TtsSynthesizeOutput output{};
    int32_t                    res = geniex_tts_synthesize(tts, &input, &output);
    CHECK_ML_ERROR(res);
    GENIEX_LOG_INFO("{}", output);

    // Verify audio path is not null and contains valid UTF-8
    REQUIRE(output.result.audio_path != nullptr);
    CHECK(utf8::is_valid(output.result.audio_path, output.result.audio_path + std::strlen(output.result.audio_path)));

    // Verify the generated audio file exists
    CHECK(std::filesystem::exists(output.result.audio_path));

    // Verify output metadata
    CHECK(output.result.duration_seconds > 0.0f);
    CHECK(output.result.sample_rate > 0);
    CHECK(output.result.channels > 0);
    CHECK(output.result.num_samples > 0);

    // Verify audio parameters are reasonable
    CHECK(output.result.duration_seconds <= 10.0f);  // Should not be too long for test text
    CHECK(output.result.sample_rate >= 16000);       // Common sample rates
    CHECK(output.result.sample_rate <= 48000);
    CHECK(output.result.channels >= 1);
    CHECK(output.result.channels <= 2);  // Mono or stereo

    // Verify the file size is reasonable (not empty, not enormous)
    if (std::filesystem::exists(output.result.audio_path)) {
        auto file_size = std::filesystem::file_size(output.result.audio_path);
        CHECK(file_size > 1000);      // At least 1KB for reasonable audio
        CHECK(file_size < 10000000);  // Less than 10MB for short test text
    }

    // Clean up allocated memory
    if (output.result.audio_path) {
        free(const_cast<char *>(output.result.audio_path));
    }
}

void test_tts_list_available_voices(geniex_TTS *tts, const std::string &model_name) {
    geniex_TtsListAvailableVoicesInput  input{};
    geniex_TtsListAvailableVoicesOutput output{};

    int32_t res = geniex_tts_list_available_voices(tts, &input, &output);
    CHECK_ML_ERROR(res);

    REQUIRE(output.voice_count >= 0);
    GENIEX_LOG_INFO("Available voices count: {}", output.voice_count);

    if (output.voice_count > 0) {
        REQUIRE(output.voice_ids != nullptr);

        // Print first few available voices
        int max_print = std::min(10, output.voice_count);
        GENIEX_LOG_INFO("First {} available voices:", max_print);
        for (int32_t i = 0; i < max_print; i++) {
            REQUIRE(output.voice_ids[i] != nullptr);
            GENIEX_LOG_INFO("  [{}]: {}", i, output.voice_ids[i]);
            CHECK(utf8::is_valid(output.voice_ids[i], output.voice_ids[i] + std::strlen(output.voice_ids[i])));
        }

        // Check that common voices are available
        std::vector<std::string> expected_voices = {"af_heart", "af_bella", "am_adam", "am_michael"};
        for (const auto &expected_voice : expected_voices) {
            bool found = false;
            for (int32_t i = 0; i < output.voice_count; i++) {
                if (std::strcmp(output.voice_ids[i], expected_voice.c_str()) == 0) {
                    found = true;
                    GENIEX_LOG_INFO("Found expected voice: '{}'", expected_voice);
                    break;
                }
            }
            if (!found) {
                GENIEX_LOG_WARN("Expected voice '{}' not found in available voices", expected_voice);
            }
        }

        // Clean up allocated voice IDs
        for (int32_t i = 0; i < output.voice_count; i++) {
            free(const_cast<char *>(output.voice_ids[i]));
        }
        free(output.voice_ids);
    }
}

void test_tts_error_handling(geniex_TTS *tts, const std::string &model_name) {
    // Test with null text input
    {
        geniex_TtsSynthesizeInput input{};
        input.text_utf8   = nullptr;  // Invalid input
        input.config      = nullptr;
        input.output_path = nullptr;

        geniex_TtsSynthesizeOutput output{};
        int32_t                    res = geniex_tts_synthesize(tts, &input, &output);

        // Should fail with appropriate error
        CHECK(res < 0);
        GENIEX_LOG_INFO(
            "Expected error for null text: {}", geniex_get_error_message(static_cast<geniex_ErrorCode>(res)));
    }

    // Test with empty text input
    {
        geniex_TtsSynthesizeInput input{};
        input.text_utf8   = "";  // Empty text
        input.config      = nullptr;
        input.output_path = nullptr;

        geniex_TtsSynthesizeOutput output{};
        int32_t                    res = geniex_tts_synthesize(tts, &input, &output);

        // This might succeed (empty audio) or fail, both are valid
        if (res >= 0) {
            GENIEX_LOG_INFO("TTS handled empty text gracefully");
            if (output.result.audio_path) {
                if (std::filesystem::exists(output.result.audio_path)) {
                    std::filesystem::remove(output.result.audio_path);
                }
                free(const_cast<char *>(output.result.audio_path));
            }
        } else {
            GENIEX_LOG_INFO(
                "TTS rejected empty text with error: {}", geniex_get_error_message(static_cast<geniex_ErrorCode>(res)));
        }
    }

    // Test with invalid output directory
    {
        geniex_TTSConfig config{};
        config.voice       = "af_heart";
        config.speed       = 1.0f;
        config.seed        = 42;
        config.sample_rate = 24000;

        geniex_TtsSynthesizeInput input{};
        input.text_utf8   = "Test text";
        input.config      = &config;
        input.output_path = "/invalid/directory/path/output.wav";  // Invalid directory

        geniex_TtsSynthesizeOutput output{};
        int32_t                    res = geniex_tts_synthesize(tts, &input, &output);

        // Should either succeed (if TTS creates directory) or fail gracefully
        if (res < 0) {
            GENIEX_LOG_INFO("Expected error for invalid output path: {}",
                geniex_get_error_message(static_cast<geniex_ErrorCode>(res)));
        } else {
            GENIEX_LOG_INFO("TTS handled invalid path gracefully");
            if (output.result.audio_path) {
                if (std::filesystem::exists(output.result.audio_path)) {
                    std::filesystem::remove(output.result.audio_path);
                }
                free(const_cast<char *>(output.result.audio_path));
            }
        }
    }
}

// Register all TTS tests
template <typename PluginType>
void register_tts_tests(TestRegistry<geniex_TTS> &registry) {
    REGISTER_TEST(registry, TtsSynthesizeBasic, test_tts_synthesize_basic(model, model_name););
    REGISTER_TEST(registry, TtsListAvailableVoices, test_tts_list_available_voices(model, model_name););
    REGISTER_TEST(registry, TtsErrorHandling, test_tts_error_handling(model, model_name););
}

// Generate test cases for all plugins
#define GEN(Plugin) TEST_CASE_FOR_PLUGIN(geniex_TTS, Plugin, setup_guard, register_tts_tests<Plugin>)
PLUGINS(GEN)
#undef GEN

}  // namespace

TEST_MAIN()