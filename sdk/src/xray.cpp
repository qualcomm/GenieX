// SPDX-License-Identifier: BSD-3-Clause

#include "xray_c.h"

#include <new>

#include "xray.h"

namespace gx = geniex::xray;

struct geniex_XrayScorer {
    gx::LapScorer scorer;
    bool adapter_failed{false};
};

namespace {

bool valid_event(const geniex_XrayEvent& event) {
    return event.schema_version == GENIEX_XRAY_SCHEMA_VERSION &&
           event.phase < GENIEX_XRAY_PHASE_COUNT &&
           event.validation <= GENIEX_XRAY_VALIDATION_FAIL &&
           event.lifecycle <= GENIEX_XRAY_LIFECYCLE_FAILED &&
           event.energy_confidence <= GENIEX_XRAY_ENERGY_CALIBRATED &&
           event.error_envelope <= GENIEX_XRAY_ENVELOPE_RESIDUE;
}

gx::Event convert(const geniex_XrayEvent& source) {
    gx::Event event;
    event.schema_version = source.schema_version;
    event.timestamp_us = source.timestamp_us;
    event.execution_id = source.execution_id;
    event.module_id = source.module_id;
    event.module_version = source.module_version;
    event.input_hash = source.input_hash;
    event.explicit_state_hash = source.explicit_state_hash;
    event.phase = static_cast<gx::Phase>(source.phase);
    event.requested_route_id = source.requested_route_id;
    event.selected_route_id = source.selected_route_id;
    event.observed_route_id = source.observed_route_id;
    event.elapsed_us = source.elapsed_us;
    event.work_units = source.work_units;
    event.prompt_tokens_delta = source.prompt_tokens_delta;
    event.generated_tokens_delta = source.generated_tokens_delta;
    event.artifact_bytes = source.artifact_bytes;
    event.resident_weight_bytes = source.resident_weight_bytes;
    event.kv_bytes = source.kv_bytes;
    event.kv_delta_bytes = source.kv_delta_bytes;
    event.runtime_bytes = source.runtime_bytes;
    event.peak_rss_bytes = source.peak_rss_bytes;
    event.memory_read_bytes = source.memory_read_bytes;
    event.memory_write_bytes = source.memory_write_bytes;
    event.energy_delta_uj = source.energy_delta_uj;
    event.energy_confidence = static_cast<gx::EnergyConfidence>(source.energy_confidence);
    event.thermal_headroom_millic = source.thermal_headroom_millic;
    event.thermal_slope_millic_per_min = source.thermal_slope_millic_per_min;
    event.throttled = source.throttled;
    event.active_envelopes = source.active_envelopes;
    event.passed_envelopes = source.passed_envelopes;
    event.validation = static_cast<gx::Validation>(source.validation);
    event.lifecycle = static_cast<gx::Lifecycle>(source.lifecycle);
    event.live_handles = source.live_handles;
    event.live_threads = source.live_threads;
    event.live_file_descriptors = source.live_file_descriptors;
    event.live_services = source.live_services;
    event.live_jobs = source.live_jobs;
    event.live_wake_locks = source.live_wake_locks;
    event.mapped_model_bytes = source.mapped_model_bytes;
    event.output_checked = source.output_checked;
    event.output_valid = source.output_valid;
    event.error.code = source.error_code;
    event.error.envelope = static_cast<gx::Envelope>(source.error_envelope);
    event.error.phase = event.phase;
    return event;
}

void convert(const gx::LapResult& source, geniex_XrayLapResult& result) {
    result = {};
    result.schema_version = source.schema_version;
    result.valid = source.valid;
    result.observed_phases = source.observed_phases;
    result.missing_phases = source.missing_phases;
    result.failed_envelopes = source.failed_envelopes;
    result.execution_id = source.execution_id;
    result.module_id = source.module_id;
    result.module_version = source.module_version;
    result.input_hash = source.input_hash;
    result.explicit_state_hash = source.explicit_state_hash;
    result.requested_route_id = source.requested_route_id;
    result.selected_route_id = source.selected_route_id;
    result.observed_route_id = source.observed_route_id;
    result.end_to_end_us = source.end_to_end_us;
    result.prompt_tokens = source.prompt_tokens;
    result.generated_tokens = source.generated_tokens;
    result.energy_uj = source.energy_uj;
    result.energy_confidence = static_cast<uint32_t>(source.energy_confidence);
    result.energy_qualified = source.energy_qualified;
    result.tokens_per_joule = source.tokens_per_joule;
    result.minimum_thermal_headroom_millic =
        source.minimum_thermal_headroom_millic;
    result.maximum_thermal_slope_millic_per_min =
        source.maximum_thermal_slope_millic_per_min;
    result.throttled = source.throttled;
    result.validation_failed = source.validation_failed;
    result.validation_complete = source.validation_complete;
    result.identity_complete = source.identity_complete;
    result.identity_consistent = source.identity_consistent;
    result.output_valid = source.output_valid;
    result.final_lifecycle = static_cast<uint32_t>(source.final_lifecycle);
    result.residue_count = source.residue_count;
    result.first_error_code = source.first_error.code;
}

}  // namespace

extern "C" geniex_XrayScorer* geniex_xray_scorer_create(void) {
    return new (std::nothrow) geniex_XrayScorer;
}

extern "C" void geniex_xray_scorer_destroy(geniex_XrayScorer* scorer) {
    delete scorer;
}

extern "C" int32_t geniex_xray_scorer_record(
    geniex_XrayScorer* scorer, const geniex_XrayEvent* event) {
    if (!scorer || !event) return -1;
    if (!valid_event(*event)) {
        scorer->adapter_failed = true;
        return -2;
    }
    scorer->scorer.record(convert(*event));
    return 0;
}

extern "C" int32_t geniex_xray_scorer_finish(
    const geniex_XrayScorer* scorer, geniex_XrayLapResult* result) {
    if (!scorer || !result) return -1;
    auto lap = scorer->scorer.finish();
    if (scorer->adapter_failed) {
        lap.valid = false;
        lap.failed_envelopes |= gx::envelope_bit(gx::Envelope::Safety);
    }
    convert(lap, *result);
    return 0;
}
