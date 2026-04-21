#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <thread>
#include <vector>

#include "doctest.h"
#include "geniex.h"
#include "logging.h"
#include "utf8.h"  // IWYU pragma: export
#include "util.h"

namespace {

#define PLUGINS(M) M(qairt)
using Param = std::tuple<std::string, std::string, std::string, std::optional<std::string>, std::optional<std::string>>;

Setup<Param, geniex_ASR> setup_guard(
    SetupMap<Param>{
        {qairt::value,
            {
#if defined(__ANDROID__)
                {"parakeet-tdt-0.6b-v3-npu",
                    "parakeet",
                    "/data/local/tmp/geniex/modelfiles/parakeet-tdt-0.6b-v3-npu/"
                    "weights-1-5.nexa",
                    std::nullopt,
                    std::nullopt},
                {"wav2vec2-base-960h-npu",
                    "wav2vec2",
                    "/data/local/tmp/geniex/modelfiles/wav2vec2-base-960h-npu/"
                    "weights-1-1.nexa",
                    std::nullopt,
                    std::nullopt},
#elif defined(_WIN32)
                {"parakeet-tdt-0.6b-v3-npu",
                    "parakeet",
                    "modelfiles/qairt/parakeet-tdt-0.6b-v3-npu/weights-1-5.nexa",
                    std::nullopt,
                    std::nullopt},
                {"wav2vec2-base-960h-npu",
                    "wav2vec2",
                    "modelfiles/qairt/wav2vec2-base-960h-npu/weights-1-1.nexa",
                    std::nullopt,
                    std::nullopt},
#endif
                // Add more qairt models here as needed
            }},
    },
    [](geniex_PluginId plugin, Param param) {
        geniex_ASR *asr = nullptr;

        auto [test_id, name, model, tokenizer, language] = std::move(param);

        // Check if model file exists
        if (!std::filesystem::exists(model)) {
            GENIEX_LOG_WARN("Model file not found: {}", model);
            GENIEX_LOG_WARN("Skipping tests for model: {}", name);
            g_test_summary.add_skipped_model(name, model);
            return static_cast<geniex_ASR *>(nullptr);  // Return nullptr to indicate skip
        }

        geniex_AsrCreateInput input{};
        input.model_name     = name.c_str();
        input.model_path     = model.c_str();
        input.tokenizer_path = tokenizer.has_value() ? tokenizer.value().c_str() : nullptr;
        input.language       = language.has_value() ? language.value().c_str() : "en";
        input.device_id      = nullptr;
        input.plugin_id      = plugin;

        int32_t res = geniex_asr_create(&input, &asr);
        CHECK_ML_ERROR(res);
        REQUIRE(asr != nullptr);

        return asr;
    },
    nullptr, geniex_asr_destroy);

std::string test_audio_path = "modelfiles/assets/OSR_us_000_0010_16k.wav";

// Simple WAV file reader for testing purposes
struct WavHeader {
    char     riff[4];
    uint32_t chunk_size;
    char     wave[4];
    char     fmt[4];
    uint32_t fmt_chunk_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char     data[4];
    uint32_t data_size;
};

std::vector<float> load_wav_as_float32(const std::string &file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        GENIEX_LOG_ERROR("Failed to open WAV file: {}", file_path);
        return {};
    }

    WavHeader header;
    file.read(reinterpret_cast<char *>(&header), sizeof(header));

    if (std::string(header.riff, 4) != "RIFF" || std::string(header.wave, 4) != "WAVE") {
        GENIEX_LOG_ERROR("Invalid WAV file format: {}", file_path);
        return {};
    }

    GENIEX_LOG_INFO(
        "WAV file info - Sample rate: {}Hz, Channels: {}, Bits: {}, "
        "Data size: {} bytes",
        header.sample_rate,
        header.num_channels,
        header.bits_per_sample,
        header.data_size);

    std::vector<float> audio_data;

    if (header.bits_per_sample == 16) {
        // Read 16-bit PCM and convert to float32
        std::vector<int16_t> pcm_data(header.data_size / sizeof(int16_t));
        file.read(reinterpret_cast<char *>(pcm_data.data()), header.data_size);

        audio_data.reserve(pcm_data.size());
        for (int16_t sample : pcm_data) {
            audio_data.push_back(static_cast<float>(sample) / 32768.0f);
        }
    } else if (header.bits_per_sample == 32) {
        // Read 32-bit float directly
        audio_data.resize(header.data_size / sizeof(float));
        file.read(reinterpret_cast<char *>(audio_data.data()), header.data_size);
    } else {
        GENIEX_LOG_ERROR("Unsupported bits per sample: {}", header.bits_per_sample);
        return {};
    }

    GENIEX_LOG_INFO("Loaded {} audio samples from WAV file", audio_data.size());
    return audio_data;
}

// Streaming test callback data
struct StreamingTestData {
    std::string              accumulated_text;
    std::vector<std::string> streaming_updates;
    bool                     callback_called = false;

    void reset() {
        accumulated_text.clear();
        streaming_updates.clear();
        callback_called = false;
    }
};

// Callback function for streaming transcription
void streaming_transcription_callback(const char *text, void *user_data) {
    auto *test_data = static_cast<StreamingTestData *>(user_data);
    if (text && strlen(text) > 0) {
        test_data->callback_called = true;
        test_data->streaming_updates.push_back(std::string(text));
        test_data->accumulated_text = text;  // Keep the latest complete transcription
    }
}

// Test function definitions
void test_asr_transcribe_basic(geniex_ASR *asr, const std::string &model_name) {
    // Check if test audio file exists
    if (!std::filesystem::exists(test_audio_path)) {
        GENIEX_LOG_ERROR("Test audio file not found: {}", test_audio_path);
        REQUIRE(false);
    }

    // Set up ASR configuration
    geniex_ASRConfig config{};
    config.timestamps = "segment";
    config.beam_size  = 5;
    config.stream     = false;

    geniex_AsrTranscribeInput input{};
    input.audio_path = test_audio_path.c_str();
    input.config     = &config;
    input.language   = nullptr;  // Use default language from creation

    geniex_AsrTranscribeOutput output{};
    int32_t                    res = geniex_asr_transcribe(asr, &input, &output);
    CHECK_ML_ERROR(res);
    GENIEX_LOG_INFO("{}", output);

    REQUIRE(output.result.transcript != nullptr);

    CHECK(utf8::is_valid(output.result.transcript, output.result.transcript + std::strlen(output.result.transcript)));

    std::string transcript_lower(output.result.transcript);
    std::transform(transcript_lower.begin(), transcript_lower.end(), transcript_lower.begin(), ::tolower);

    std::vector<std::string> expected_keywords = {"plank", "lemon", "garbage"};
    for (const auto &keyword : expected_keywords) {
        bool found = transcript_lower.find(keyword) != std::string::npos;
        auto error_msg =
            fmt::format("Expected keyword '{}' not found in transcript: '{}'", keyword, output.result.transcript);
        CHECK_MESSAGE(found, error_msg.c_str());
        if (found) {
        } else {
            GENIEX_LOG_WARN("Missing expected keyword: '{}'", keyword);
        }
    }

    // Verify confidence scores
    if (output.result.confidence_count > 0) {
        REQUIRE(output.result.confidence_scores != nullptr);

        // Print first few confidence scores
        int max_print = std::min(5, static_cast<int>(output.result.confidence_count));
        for (int i = 0; i < max_print; i++) {
            CHECK(output.result.confidence_scores[i] >= 0.0f);
            CHECK(output.result.confidence_scores[i] <= 1.0f);
        }
    }

    // Verify timestamps
    if (output.result.timestamp_count > 0) {
        REQUIRE(output.result.timestamps != nullptr);

        // Print first few timestamps
        int max_print = std::min(3, static_cast<int>(output.result.timestamp_count));
        for (int i = 0; i < max_print; i++) {
            float start = output.result.timestamps[i * 2];
            float end   = output.result.timestamps[i * 2 + 1];
            CHECK(start >= 0.0f);
            CHECK(end >= start);
        }
    }

    // Clean up allocated memory
    if (output.result.transcript) {
        free(output.result.transcript);
    }
    if (output.result.confidence_scores) {
        free(output.result.confidence_scores);
    }
    if (output.result.timestamps) {
        free(output.result.timestamps);
    }
}

void test_asr_list_supported_languages(geniex_ASR *asr, const std::string &model_name) {
    geniex_AsrListSupportedLanguagesInput  input{};
    geniex_AsrListSupportedLanguagesOutput output{};

    int32_t res = geniex_asr_list_supported_languages(asr, &input, &output);
    CHECK_ML_ERROR(res);

    REQUIRE(output.language_count >= 0);
    GENIEX_LOG_INFO("{}", output);

    if (output.language_count > 0) {
        REQUIRE(output.language_codes != nullptr);

        // Print first few supported languages
        int max_print = std::min(10, output.language_count);
        for (int32_t i = 0; i < max_print; i++) {
            REQUIRE(output.language_codes[i] != nullptr);
            CHECK(utf8::is_valid(
                output.language_codes[i], output.language_codes[i] + std::strlen(output.language_codes[i])));
        }

        // Check that "en" is supported (common requirement)
        bool found_en = false;
        for (int32_t i = 0; i < output.language_count; i++) {
            if (std::strcmp(output.language_codes[i], "en") == 0) {
                found_en = true;
                break;
            }
        }
        CHECK(found_en);

        // Clean up allocated language codes
        for (int32_t i = 0; i < output.language_count; i++) {
            free(const_cast<char *>(output.language_codes[i]));
        }
        free(output.language_codes);
    }
}

void test_asr_error_handling(geniex_ASR *asr, const std::string &model_name) {
    // Test with invalid audio path
    geniex_AsrTranscribeInput input{};
    input.audio_path = "nonexistent_file.wav";
    input.config     = nullptr;
    input.language   = nullptr;

    geniex_AsrTranscribeOutput output{};
    int32_t                    res = geniex_asr_transcribe(asr, &input, &output);

    // Should fail with appropriate error
    CHECK(res < 0);
    GENIEX_LOG_INFO(
        "Expected error for nonexistent file: {}", geniex_get_error_message(static_cast<geniex_ErrorCode>(res)));
}

void test_asr_streaming_transcription(geniex_ASR *asr, const std::string &model_name) {
    // Only parakeet models support streaming - skip for other models
    if (model_name.find("parakeet") == std::string::npos) {
        GENIEX_LOG_INFO(
            "Skipping streaming test for model '{}' - only parakeet "
            "models support streaming",
            model_name);
        return;
    }

    // Check if test audio file exists
    if (!std::filesystem::exists(test_audio_path)) {
        GENIEX_LOG_ERROR("Test audio file not found: {}", test_audio_path);
        REQUIRE(false);
    }

    // Load audio data as float32 samples
    std::vector<float> audio_data = load_wav_as_float32(test_audio_path);
    REQUIRE(!audio_data.empty());
    GENIEX_LOG_INFO("Loaded {} audio samples for streaming test", audio_data.size());

    // Set up streaming callback data
    StreamingTestData test_data;
    test_data.reset();

    // Set up explicit stream configuration (matching CLI behavior)
    geniex_ASRStreamConfig stream_config{};
    stream_config.chunk_duration   = 4.0f;
    stream_config.overlap_duration = 3.5f;
    stream_config.sample_rate      = 16000;
    stream_config.max_queue_size   = 10;
    stream_config.buffer_size      = 512;
    stream_config.timestamps       = "segment";
    stream_config.beam_size        = 4;

    // Begin streaming with consolidated configuration
    geniex_AsrStreamBeginInput begin_input{};
    begin_input.stream_config    = &stream_config;
    begin_input.language         = "en";
    begin_input.on_transcription = streaming_transcription_callback;
    begin_input.user_data        = &test_data;

    geniex_AsrStreamBeginOutput begin_output{};
    int32_t                     res = geniex_asr_stream_begin(asr, &begin_input, &begin_output);
    CHECK_ML_ERROR(res);

    // Simulate streaming by pushing audio data in chunks
    // Audio push buffer: 512 float32 samples at 16kHz (~32ms per chunk)
    const int32_t chunk_size    = 512;
    int32_t       total_samples = static_cast<int32_t>(audio_data.size());

    GENIEX_LOG_INFO("Starting to stream {} total samples in chunks of {}", total_samples, chunk_size);

    for (int32_t offset = 0; offset < total_samples; offset += chunk_size) {
        int32_t samples_to_send = std::min(chunk_size, total_samples - offset);

        geniex_AsrStreamPushAudioInput push_input{};
        push_input.audio_data = audio_data.data() + offset;
        push_input.length     = samples_to_send;

        res = geniex_asr_stream_push_audio(asr, &push_input);
        // Silent check - only break if there's an error
        if (res < 0) {
            GENIEX_LOG_ERROR(
                "Failed to push audio chunk: {}", geniex_get_error_message(static_cast<geniex_ErrorCode>(res)));
            break;
        }

        // Small delay to simulate real-time streaming (reduced for smaller chunks)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // Stop streaming gracefully
    geniex_AsrStreamStopInput stop_input{};
    stop_input.graceful = true;
    res                 = geniex_asr_stream_stop(asr, &stop_input);
    CHECK_ML_ERROR(res);

    // Wait longer for processing to complete.
    GENIEX_LOG_INFO("Waiting for transcription processing to complete...");
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Verify that we received streaming callbacks
    REQUIRE(test_data.callback_called);
    REQUIRE(!test_data.accumulated_text.empty());

    GENIEX_LOG_INFO("Final streaming transcription: '{}'", test_data.accumulated_text);
    GENIEX_LOG_INFO("Received {} streaming updates", test_data.streaming_updates.size());

    // Print all streaming updates
    for (size_t i = 0; i < test_data.streaming_updates.size(); i++) {
        GENIEX_LOG_INFO("Streaming update {}: '{}'", i + 1, test_data.streaming_updates[i]);
    }

    // Validate UTF-8 encoding of the final transcription
    CHECK(utf8::is_valid(
        test_data.accumulated_text.c_str(), test_data.accumulated_text.c_str() + test_data.accumulated_text.length()));

    // Convert to lowercase for keyword matching
    std::string transcript_lower = test_data.accumulated_text;
    std::transform(transcript_lower.begin(), transcript_lower.end(), transcript_lower.begin(), ::tolower);

    // Verify expected keywords appear in streaming transcription
    std::vector<std::string> expected_keywords = {"plank", "lemon", "garbage"};
    for (const auto &keyword : expected_keywords) {
        bool found     = transcript_lower.find(keyword) != std::string::npos;
        auto error_msg = fmt::format(
            "Expected keyword '{}' not found in streaming transcript: '{}'", keyword, test_data.accumulated_text);
        CHECK_MESSAGE(found, error_msg.c_str());
        if (found) {
            GENIEX_LOG_INFO("Found expected keyword '{}' in streaming transcript", keyword);
        } else {
            GENIEX_LOG_WARN("Missing expected keyword: '{}'", keyword);
        }
    }
}

// Register all ASR tests - conditionally based on plugin capabilities
template <typename PluginType>
void register_asr_tests(TestRegistry<geniex_ASR> &registry) {
    // Common tests for all plugins
    REGISTER_TEST(registry, AsrTranscribeBasic, test_asr_transcribe_basic(model, model_name););
    REGISTER_TEST(registry, AsrErrorHandling, test_asr_error_handling(model, model_name););

    // QAIRT-specific tests
    if (std::is_same_v<PluginType, qairt>) {
        REGISTER_TEST(registry, AsrListSupportedLanguages, test_asr_list_supported_languages(model, model_name););
        REGISTER_TEST(registry, AsrStreamingTranscription, test_asr_streaming_transcription(model, model_name););
    }
}

// Generate test cases for all plugins
#define GEN(Plugin) TEST_CASE_FOR_PLUGIN(geniex_ASR, Plugin, setup_guard, register_asr_tests<Plugin>)
PLUGINS(GEN)
#undef GEN

}  // namespace

TEST_MAIN()
