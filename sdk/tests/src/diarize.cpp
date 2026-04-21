#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <vector>

#include "doctest.h"
#include "geniex.h"
#include "logging.h"
#include "utf8.h"  // IWYU pragma: export
#include "util.h"

namespace {

#define PLUGINS(M) M(qairt)
using Param = std::tuple<std::string, std::string, std::string>;

Setup<Param, geniex_Diarize> setup_guard(
    SetupMap<Param>{
        {qairt::value,
            {
                {"Pyannote-NPU", "pyannote", "modelfiles/qairt/Pyannote-NPU/weights-1-2.nexa"},
                // Add more qairt diarization models here as needed
            }},
    },
    [](geniex_PluginId plugin, Param param) {
        geniex_Diarize* diarize = nullptr;

        auto [test_id, name, model_folder] = std::move(param);

        // Check if model folder exists
        if (!std::filesystem::exists(model_folder)) {
            GENIEX_LOG_WARN("Model folder not found: {}", model_folder);
            GENIEX_LOG_WARN("Skipping tests for model: {}", name);
            g_test_summary.add_skipped_model(name, model_folder);
            return static_cast<geniex_Diarize*>(nullptr);  // Return nullptr to indicate skip
        }

        geniex_DiarizeCreateInput input{};
        input.model_name = name.c_str();
        input.model_path = model_folder.c_str();
        input.plugin_id  = plugin;
        input.device_id  = nullptr;

        // Model config
        input.config.verbose = false;

        int32_t res = geniex_diarize_create(&input, &diarize);
        CHECK_ML_ERROR(res);
        REQUIRE(diarize != nullptr);

        return diarize;
    },
    nullptr, geniex_diarize_destroy);

std::string test_audio_path = "modelfiles/assets/conversation_16k.wav";

// Comprehensive diarization test
void test_diarize(geniex_Diarize* diarize, const std::string& model_name) {
    // Check if test audio file exists
    if (!std::filesystem::exists(test_audio_path)) {
        GENIEX_LOG_ERROR("Test audio file not found: {}", test_audio_path);
        GENIEX_LOG_ERROR("Please ensure the conversation audio file exists at: {}", test_audio_path);
        REQUIRE(false);
    }

    GENIEX_LOG_INFO("Running diarization on: {}", test_audio_path);

    // Set up diarization configuration
    geniex_DiarizeConfig config{};
    config.min_speakers = 0;  // Auto-detect
    config.max_speakers = 0;  // No limit

    geniex_DiarizeInferInput input{};
    input.audio_path = test_audio_path.c_str();
    input.config     = &config;

    geniex_DiarizeInferOutput output{};

    auto    start_time  = std::chrono::high_resolution_clock::now();
    int32_t res         = geniex_diarize_infer(diarize, &input, &output);
    auto    end_time    = std::chrono::high_resolution_clock::now();
    auto    duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    CHECK_ML_ERROR(res);
    GENIEX_LOG_INFO("{}", output);

    // === ASSERTION 1: Check for 2 speakers ===
    REQUIRE(output.num_speakers == 2);
    GENIEX_LOG_INFO("✓ Detected {} speakers (expected: 2)", output.num_speakers);

    // === ASSERTION 2: Check for at least 3 segments ===
    REQUIRE(output.segment_count >= 3);
    GENIEX_LOG_INFO("✓ Detected {} segments (expected: >= 3)", output.segment_count);

    // Verify segments are valid
    REQUIRE(output.segments != nullptr);

    // Print all segments and verify they're valid
    GENIEX_LOG_INFO("Speech segments:");
    for (int32_t i = 0; i < output.segment_count; ++i) {
        REQUIRE(output.segments[i].speaker_label != nullptr);

        // Verify segment times are valid
        CHECK(output.segments[i].start_time >= 0.0f);
        CHECK(output.segments[i].end_time > output.segments[i].start_time);
        CHECK(output.segments[i].end_time <= output.duration + 0.1f);  // Allow small tolerance

        // Verify segment ordering (segments should be time-ordered)
        if (i > 0) {
            CHECK(output.segments[i].start_time >= output.segments[i - 1].start_time);
        }

        // Verify speaker labels are valid UTF-8
        const char* label = output.segments[i].speaker_label;
        CHECK(utf8::is_valid(label, label + std::strlen(label)));

        GENIEX_LOG_INFO("  [{}s - {}s] {}",
            output.segments[i].start_time,
            output.segments[i].end_time,
            output.segments[i].speaker_label);
    }

    // === ASSERTION 3: Check speaker sequence pattern A -> B -> A ===
    // Extract distinct speaker turns (ignoring consecutive segments with same speaker)
    std::vector<std::string> speaker_turns;
    std::string              prev_speaker = "";

    for (int32_t i = 0; i < output.segment_count; ++i) {
        std::string current_speaker = output.segments[i].speaker_label;
        if (current_speaker != prev_speaker) {
            speaker_turns.push_back(current_speaker);
            prev_speaker = current_speaker;
        }
    }

    GENIEX_LOG_INFO("Speaker turn sequence ({} turns):", speaker_turns.size());
    for (size_t i = 0; i < speaker_turns.size(); ++i) {
        GENIEX_LOG_INFO("  Turn {}: {}", i + 1, speaker_turns[i]);
    }

    // Check for A -> B -> A pattern (at least 3 distinct turns)
    if (speaker_turns.size() >= 3) {
        std::string first_speaker  = speaker_turns[0];
        std::string second_speaker = speaker_turns[1];
        std::string third_speaker  = speaker_turns[2];

        // Check pattern: first speaker == third speaker != second speaker
        bool pattern_valid = (first_speaker == third_speaker) && (first_speaker != second_speaker);

        if (pattern_valid) {
            GENIEX_LOG_INFO("✓ Speaker turn pattern: {} -> {} -> {} (A->B->A pattern)",
                first_speaker,
                second_speaker,
                third_speaker);
        } else {
            GENIEX_LOG_WARN("Speaker turn pattern: {} -> {} -> {}", first_speaker, second_speaker, third_speaker);
            GENIEX_LOG_WARN("Expected A->B->A pattern (first == third != second)");
        }

        REQUIRE(pattern_valid);
    } else {
        GENIEX_LOG_WARN(
            "Only {} distinct speaker turns found, need at least 3 for pattern check", speaker_turns.size());
        REQUIRE(speaker_turns.size() >= 3);
    }

    // Verify audio duration is reasonable
    CHECK(output.duration > 0.0f);
    GENIEX_LOG_INFO("Total audio duration: {}s", output.duration);

    // Verify profile data
    CHECK(output.profile_data.audio_duration > 0);
    CHECK(output.profile_data.real_time_factor >= 0.0);

    GENIEX_LOG_INFO("Performance metrics:");
    GENIEX_LOG_INFO("  Processing time: {}s", output.profile_data.prompt_time / 1000000.0);
    GENIEX_LOG_INFO("  Real-time factor: {}x", output.profile_data.real_time_factor);
    GENIEX_LOG_INFO("  Total inference time: {}ms", duration_ms.count());

    // Clean up allocated memory
    if (output.segments) {
        for (int32_t i = 0; i < output.segment_count; ++i) {
            if (output.segments[i].speaker_label) {
                free(output.segments[i].speaker_label);
            }
        }
        free(output.segments);
    }
}

// Register diarization test
template <typename PluginType>
void register_diarize_tests(TestRegistry<geniex_Diarize>& registry) {
    REGISTER_TEST(registry, Diarize, test_diarize(model, model_name););
}

// Generate test cases for all plugins
#define GEN(Plugin) TEST_CASE_FOR_PLUGIN(geniex_Diarize, Plugin, setup_guard, register_diarize_tests<Plugin>)
PLUGINS(GEN)
#undef GEN

}  // namespace

TEST_MAIN()
