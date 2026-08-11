// SPDX-License-Identifier: BSD-3-Clause

#include "xray_c.h"

#include <cstdint>
#include <iostream>

namespace {
int failures = 0;
void check(bool condition, const char* expression, int line) {
    if (condition) return;
    ++failures;
    std::cerr << "CHECK failed at line " << line << ": " << expression << '\n';
}
#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

constexpr uint64_t all_envelopes =
    (UINT64_C(1) << (GENIEX_XRAY_ENVELOPE_RESIDUE + 1)) - 1;

geniex_XrayEvent passing_event(uint32_t phase, uint64_t timestamp) {
    geniex_XrayEvent event{};
    event.schema_version = GENIEX_XRAY_SCHEMA_VERSION;
    event.timestamp_us = timestamp;
    event.execution_id = 7;
    event.module_id = 5;
    event.module_version = 1;
    event.input_hash = 13;
    event.explicit_state_hash = 17 + phase;
    event.phase = phase;
    event.requested_route_id = 11;
    event.selected_route_id = 11;
    event.observed_route_id = 11;
    event.elapsed_us = 100;
    event.work_units = 1;
    event.energy_confidence = GENIEX_XRAY_ENERGY_UNAVAILABLE;
    event.thermal_headroom_millic = 5000;
    event.active_envelopes = all_envelopes;
    event.passed_envelopes = all_envelopes;
    event.validation = GENIEX_XRAY_VALIDATION_PASS;
    event.lifecycle = GENIEX_XRAY_LIFECYCLE_RUNNING;
    event.error_envelope = GENIEX_XRAY_ENVELOPE_RUNTIME;
    return event;
}

void test_valid_lap() {
    auto* scorer = geniex_xray_scorer_create();
    CHECK(scorer != nullptr);
    for (uint32_t phase = 0; phase < GENIEX_XRAY_PHASE_COUNT; ++phase) {
        auto event = passing_event(phase, 1000 + phase * 100);
        if (phase == GENIEX_XRAY_PREFILL) event.prompt_tokens_delta = 32;
        if (phase == GENIEX_XRAY_DECODE) {
            event.generated_tokens_delta = 128;
            event.energy_delta_uj = 1000000;
            event.energy_confidence = GENIEX_XRAY_ENERGY_MEASURED;
            event.output_checked = true;
            event.output_valid = true;
        }
        if (phase == GENIEX_XRAY_RECOVERY)
            event.lifecycle = GENIEX_XRAY_LIFECYCLE_RELEASED;
        CHECK(geniex_xray_scorer_record(scorer, &event) == 0);
    }
    geniex_XrayLapResult result{};
    CHECK(geniex_xray_scorer_finish(scorer, &result) == 0);
    CHECK(result.valid);
    CHECK(result.tokens_per_joule == 128.0);
    CHECK(result.requested_route_id == 11);
    CHECK(result.selected_route_id == 11);
    CHECK(result.observed_route_id == 11);
    CHECK(result.validation_complete);
    CHECK(result.identity_complete);
    CHECK(result.identity_consistent);
    CHECK(result.minimum_thermal_headroom_millic == 5000);
    geniex_xray_scorer_destroy(scorer);
}

void test_adapter_and_contract_fail_closed() {
    auto* scorer = geniex_xray_scorer_create();
    auto invalid = passing_event(GENIEX_XRAY_PHASE_COUNT, 1);
    CHECK(geniex_xray_scorer_record(scorer, &invalid) == -2);
    geniex_XrayLapResult result{};
    CHECK(geniex_xray_scorer_finish(scorer, &result) == 0);
    CHECK(!result.valid);
    CHECK((result.failed_envelopes &
           GENIEX_XRAY_ENVELOPE_BIT(GENIEX_XRAY_ENVELOPE_SAFETY)) != 0);
    geniex_xray_scorer_destroy(scorer);

    scorer = geniex_xray_scorer_create();
    auto failed = passing_event(GENIEX_XRAY_STARTUP, 1);
    failed.validation = GENIEX_XRAY_VALIDATION_FAIL;
    failed.error_code = 0;
    CHECK(geniex_xray_scorer_record(scorer, &failed) == 0);
    CHECK(geniex_xray_scorer_finish(scorer, &result) == 0);
    CHECK(result.validation_failed);
    CHECK(!result.valid);
    geniex_xray_scorer_destroy(scorer);

    scorer = geniex_xray_scorer_create();
    auto first = passing_event(GENIEX_XRAY_STARTUP, 1);
    auto drift = passing_event(GENIEX_XRAY_RESIDENCY, 2);
    drift.observed_route_id = 12;
    CHECK(geniex_xray_scorer_record(scorer, &first) == 0);
    CHECK(geniex_xray_scorer_record(scorer, &drift) == 0);
    CHECK(geniex_xray_scorer_finish(scorer, &result) == 0);
    CHECK(!result.identity_consistent);
    CHECK(!result.valid);
    geniex_xray_scorer_destroy(scorer);
}

void test_unqualified_energy_never_claims_efficiency() {
    auto* scorer = geniex_xray_scorer_create();
    auto event = passing_event(GENIEX_XRAY_DECODE, 1);
    event.generated_tokens_delta = 32;
    event.energy_delta_uj = 1000;
    event.energy_confidence = GENIEX_XRAY_ENERGY_ESTIMATED;
    CHECK(geniex_xray_scorer_record(scorer, &event) == 0);
    geniex_XrayLapResult result{};
    CHECK(geniex_xray_scorer_finish(scorer, &result) == 0);
    CHECK(!result.energy_qualified);
    CHECK(result.tokens_per_joule == 0.0);
    geniex_xray_scorer_destroy(scorer);
}
}  // namespace

int main() {
    test_valid_lap();
    test_adapter_and_contract_fail_closed();
    test_unqualified_energy_never_claims_efficiency();
    if (failures != 0) {
        std::cerr << failures << " Xray C adapter test(s) failed\n";
        return 1;
    }
    std::cout << "Xray C adapter tests passed\n";
    return 0;
}
