// SPDX-License-Identifier: BSD-3-Clause

#include "xray.h"

#include <chrono>
#include <cstdint>
#include <iostream>

using namespace geniex::xray;

namespace {

int failures = 0;

void check(bool condition, const char* expression, int line) {
    if (condition) return;
    ++failures;
    std::cerr << "CHECK failed at line " << line << ": " << expression << '\n';
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

constexpr uint64_t kAllEnvelopes =
    envelope_bit(Envelope::Artifact) | envelope_bit(Envelope::Storage) |
    envelope_bit(Envelope::PeakRam) | envelope_bit(Envelope::ResidentWeights) |
    envelope_bit(Envelope::DynamicKv) | envelope_bit(Envelope::Context) |
    envelope_bit(Envelope::Runtime) | envelope_bit(Envelope::Safety) |
    envelope_bit(Envelope::Backend) | envelope_bit(Envelope::Thermal) |
    envelope_bit(Envelope::Energy) | envelope_bit(Envelope::OutputValidity) |
    envelope_bit(Envelope::Lifecycle) | envelope_bit(Envelope::Residue);

Event passing_event(Phase phase, uint64_t timestamp) {
    Event event;
    event.timestamp_us = timestamp;
    event.execution_id = 7;
    event.module_id = 5;
    event.module_version = 1;
    event.input_hash = 13;
    event.explicit_state_hash = 17 + static_cast<uint8_t>(phase);
    event.phase = phase;
    event.requested_route_id = 11;
    event.selected_route_id = 11;
    event.observed_route_id = 11;
    event.elapsed_us = 100;
    event.work_units = 1;
    event.active_envelopes = kAllEnvelopes;
    event.passed_envelopes = kAllEnvelopes;
    event.validation = Validation::Pass;
    event.lifecycle = Lifecycle::Running;
    event.thermal_headroom_millic = 5000;
    return event;
}

LapResult valid_lap(uint64_t decode_elapsed_us = 100) {
    LapScorer scorer;
    uint64_t timestamp = 1000;
    for (uint8_t raw = 0; raw < static_cast<uint8_t>(Phase::Count); ++raw) {
        auto event = passing_event(static_cast<Phase>(raw), timestamp);
        timestamp += 100;
        if (event.phase == Phase::Prefill) event.prompt_tokens_delta = 32;
        if (event.phase == Phase::Decode) {
            event.generated_tokens_delta = 128;
            event.energy_delta_uj = 1000000;
            event.energy_confidence = EnergyConfidence::Measured;
            event.output_checked = true;
            event.output_valid = true;
            event.elapsed_us = decode_elapsed_us;
        }
        if (event.phase == Phase::Recovery) {
            event.lifecycle = Lifecycle::Released;
            event.live_handles = 0;
            event.live_threads = 0;
            event.live_file_descriptors = 0;
            event.live_services = 0;
            event.live_jobs = 0;
            event.live_wake_locks = 0;
            event.mapped_model_bytes = 0;
        }
        scorer.record(event);
    }
    return scorer.finish();
}

void test_valid_full_lap() {
    const auto result = valid_lap();
    CHECK(result.valid);
    CHECK(result.missing_phases == 0);
    CHECK(result.failed_envelopes == 0);
    CHECK(result.generated_tokens == 128);
    CHECK(result.energy_confidence == EnergyConfidence::Measured);
    CHECK(result.energy_qualified);
    CHECK(result.tokens_per_joule == 128.0);
    CHECK(result.identity_complete);
    CHECK(result.identity_consistent);
    CHECK(result.execution_id == 7);
    CHECK(result.module_id == 5);
    CHECK(result.module_version == 1);
    CHECK(result.input_hash == 13);
    CHECK(result.requested_route_id == 11);
    CHECK(result.selected_route_id == 11);
    CHECK(result.observed_route_id == 11);
    CHECK(result.final_lifecycle == Lifecycle::Released);
    CHECK(result.residue_count == 0);
}

void test_missing_phase_fails_closed() {
    LapScorer scorer;
    scorer.record(passing_event(Phase::Startup, 1));
    const auto result = scorer.finish();
    CHECK(!result.valid);
    CHECK((result.missing_phases & phase_bit(Phase::Recovery)) != 0);
}

void test_unknown_active_envelope_fails_closed() {
    LapScorer scorer;
    auto event = passing_event(Phase::Startup, 1);
    event.passed_envelopes &= ~envelope_bit(Envelope::PeakRam);
    scorer.record(event);
    const auto result = scorer.finish();
    CHECK(!result.valid);
    CHECK((result.failed_envelopes & envelope_bit(Envelope::PeakRam)) != 0);
}

void test_explicit_fail_cannot_be_masked() {
    LapScorer scorer;
    auto event = passing_event(Phase::Startup, 1);
    event.validation = Validation::Fail;
    event.error = {0, Envelope::Runtime, Phase::Startup};
    scorer.record(event);
    const auto result = scorer.finish();
    CHECK(result.validation_failed);
    CHECK((result.failed_envelopes & envelope_bit(Envelope::Runtime)) != 0);
    CHECK(!result.valid);
}

void test_unknown_validation_fails_closed() {
    LapScorer scorer;
    auto event = passing_event(Phase::Startup, 1);
    event.validation = Validation::Unknown;
    scorer.record(event);
    const auto result = scorer.finish();
    CHECK(!result.validation_complete);
    CHECK((result.failed_envelopes & envelope_bit(Envelope::Runtime)) != 0);
    CHECK(!result.valid);
}

void test_throttling_fails_endurance() {
    LapScorer scorer;
    auto event = passing_event(Phase::ThermalEquilibrium, 1);
    event.throttled = true;
    scorer.record(event);
    CHECK(scorer.finish().throttled);
    CHECK(!scorer.finish().valid);
}

void test_recovery_residue_fails() {
    LapScorer scorer;
    auto event = passing_event(Phase::Recovery, 1);
    event.lifecycle = Lifecycle::Released;
    event.live_handles = 1;
    scorer.record(event);
    const auto result = scorer.finish();
    CHECK(result.residue_count == 1);
    CHECK(!result.valid);
}

void test_later_invalid_output_cannot_be_masked() {
    LapScorer scorer;
    auto valid_output = passing_event(Phase::Decode, 1);
    valid_output.output_checked = true;
    valid_output.output_valid = true;
    scorer.record(valid_output);
    auto invalid_output = passing_event(Phase::Decode, 2);
    invalid_output.output_checked = true;
    invalid_output.output_valid = false;
    scorer.record(invalid_output);
    const auto result = scorer.finish();
    CHECK(!result.output_valid);
    CHECK((result.failed_envelopes & envelope_bit(Envelope::OutputValidity)) != 0);
}

void test_energy_requires_authoritative_measurement() {
    LapScorer scorer;
    auto event = passing_event(Phase::Decode, 1);
    event.generated_tokens_delta = 128;
    event.energy_delta_uj = 1000000;
    event.energy_confidence = EnergyConfidence::Estimated;
    scorer.record(event);
    const auto result = scorer.finish();
    CHECK(result.energy_confidence == EnergyConfidence::Estimated);
    CHECK(!result.energy_qualified);
    CHECK((result.failed_envelopes & envelope_bit(Envelope::Energy)) != 0);
    CHECK(!result.valid);
}

void test_zero_energy_never_claims_efficiency() {
    LapScorer scorer;
    scorer.record(passing_event(Phase::Decode, 1));
    const auto result = scorer.finish();
    CHECK(result.energy_confidence == EnergyConfidence::Unavailable);
    CHECK(!result.energy_qualified);
    CHECK(result.tokens_per_joule == 0.0);
}

void test_identity_and_route_fields_are_mandatory() {
    LapScorer scorer;
    auto event = passing_event(Phase::Startup, 1);
    event.observed_route_id = 0;
    scorer.record(event);
    const auto result = scorer.finish();
    CHECK(!result.identity_complete);
    CHECK((result.failed_envelopes & envelope_bit(Envelope::Safety)) != 0);
    CHECK(!result.valid);

    LapScorer drift_scorer;
    drift_scorer.record(passing_event(Phase::Startup, 1));
    auto drift = passing_event(Phase::Residency, 2);
    drift.observed_route_id = 12;
    drift_scorer.record(drift);
    const auto drift_result = drift_scorer.finish();
    CHECK(!drift_result.identity_consistent);
    CHECK((drift_result.failed_envelopes & envelope_bit(Envelope::Safety)) != 0);
}

void test_validity_precedes_speed() {
    const auto slower_valid = valid_lap(1000);
    auto invalid = valid_lap(10);
    invalid.valid = false;
    invalid.failed_envelopes = envelope_bit(Envelope::OutputValidity);
    CHECK(slower_valid.better_than(invalid));

    const auto faster_valid = valid_lap(10);
    CHECK(faster_valid.better_than(slower_valid));
}

class CountingSink final : public Sink {
public:
    void emit(const Event&) noexcept override { ++count; }
    uint64_t count{0};
};

uint64_t benchmark_recorder(Recorder& recorder, const Event& event, uint64_t iterations) {
    const auto start = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < iterations; ++i) recorder.record(event);
    const auto end = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

void test_counters_and_measured_overhead() {
    constexpr uint64_t iterations = 200000;
    auto event = passing_event(Phase::Decode, 1);
    event.elapsed_us = 4;
    event.work_units = 2;
    event.kv_bytes = 4096;
    event.peak_rss_bytes = 8192;

    Recorder off(Mode::Off);
    Recorder light(Mode::Light);
    CountingSink sink;
    Recorder deep(Mode::Deep, &sink);

    const auto off_ns = benchmark_recorder(off, event, iterations);
    const auto light_ns = benchmark_recorder(light, event, iterations);
    const auto deep_ns = benchmark_recorder(deep, event, iterations);

    const auto snapshot = light.snapshot();
    const auto index = static_cast<size_t>(Phase::Decode);
    CHECK(snapshot.events[index] == iterations);
    CHECK(snapshot.elapsed_us[index] == iterations * 4);
    CHECK(snapshot.work_units[index] == iterations * 2);
    CHECK(snapshot.max_kv_bytes == 4096);
    CHECK(snapshot.peak_rss_bytes == 8192);
    CHECK(sink.count == iterations);

    const double off_per_event = static_cast<double>(off_ns) / iterations;
    const double light_per_event = static_cast<double>(light_ns) / iterations;
    const double deep_per_event = static_cast<double>(deep_ns) / iterations;
    std::cout << "xray overhead ns/event: off=" << off_per_event
              << " light=" << light_per_event
              << " deep=" << deep_per_event << '\n';

    // Report only. Device-specific, statistically derived budgets are separate gates.
}

}  // namespace

int main() {
    test_valid_full_lap();
    test_missing_phase_fails_closed();
    test_unknown_active_envelope_fails_closed();
    test_explicit_fail_cannot_be_masked();
    test_unknown_validation_fails_closed();
    test_throttling_fails_endurance();
    test_recovery_residue_fails();
    test_later_invalid_output_cannot_be_masked();
    test_energy_requires_authoritative_measurement();
    test_zero_energy_never_claims_efficiency();
    test_identity_and_route_fields_are_mandatory();
    test_validity_precedes_speed();
    test_counters_and_measured_overhead();
    if (failures != 0) {
        std::cerr << failures << " xray contract test(s) failed\n";
        return 1;
    }
    std::cout << "xray contract tests passed\n";
    return 0;
}
