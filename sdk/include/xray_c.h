// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GENIEX_XRAY_SCHEMA_VERSION 1u
#define GENIEX_XRAY_PHASE_COUNT 9u

typedef enum {
    GENIEX_XRAY_STARTUP = 0,
    GENIEX_XRAY_RESIDENCY,
    GENIEX_XRAY_PREFILL,
    GENIEX_XRAY_DECODE,
    GENIEX_XRAY_CONTEXT_GROWTH,
    GENIEX_XRAY_THERMAL_EQUILIBRIUM,
    GENIEX_XRAY_STOP,
    GENIEX_XRAY_UNLOAD,
    GENIEX_XRAY_RECOVERY
} geniex_XrayPhase;

typedef enum {
    GENIEX_XRAY_VALIDATION_UNKNOWN = 0,
    GENIEX_XRAY_VALIDATION_PASS = 1,
    GENIEX_XRAY_VALIDATION_FAIL = 2
} geniex_XrayValidation;

typedef enum {
    GENIEX_XRAY_ENERGY_UNAVAILABLE = 0,
    GENIEX_XRAY_ENERGY_ESTIMATED,
    GENIEX_XRAY_ENERGY_MEASURED,
    GENIEX_XRAY_ENERGY_CALIBRATED
} geniex_XrayEnergyConfidence;

typedef enum {
    GENIEX_XRAY_LIFECYCLE_UNKNOWN = 0,
    GENIEX_XRAY_LIFECYCLE_STARTING,
    GENIEX_XRAY_LIFECYCLE_RESIDENT,
    GENIEX_XRAY_LIFECYCLE_RUNNING,
    GENIEX_XRAY_LIFECYCLE_STOPPING,
    GENIEX_XRAY_LIFECYCLE_UNLOADING,
    GENIEX_XRAY_LIFECYCLE_RECOVERING,
    GENIEX_XRAY_LIFECYCLE_RELEASED,
    GENIEX_XRAY_LIFECYCLE_FAILED
} geniex_XrayLifecycle;

typedef enum {
    GENIEX_XRAY_ENVELOPE_ARTIFACT = 0,
    GENIEX_XRAY_ENVELOPE_STORAGE,
    GENIEX_XRAY_ENVELOPE_PEAK_RAM,
    GENIEX_XRAY_ENVELOPE_RESIDENT_WEIGHTS,
    GENIEX_XRAY_ENVELOPE_DYNAMIC_KV,
    GENIEX_XRAY_ENVELOPE_CONTEXT,
    GENIEX_XRAY_ENVELOPE_RUNTIME,
    GENIEX_XRAY_ENVELOPE_SAFETY,
    GENIEX_XRAY_ENVELOPE_BACKEND,
    GENIEX_XRAY_ENVELOPE_THERMAL,
    GENIEX_XRAY_ENVELOPE_ENERGY,
    GENIEX_XRAY_ENVELOPE_OUTPUT_VALIDITY,
    GENIEX_XRAY_ENVELOPE_LIFECYCLE,
    GENIEX_XRAY_ENVELOPE_RESIDUE
} geniex_XrayEnvelope;

#define GENIEX_XRAY_ENVELOPE_BIT(value) (UINT64_C(1) << (value))

typedef struct {
    uint32_t schema_version;
    uint64_t timestamp_us;
    uint64_t execution_id;
    uint64_t module_id;
    uint32_t module_version;
    uint64_t input_hash;
    uint64_t explicit_state_hash;
    uint32_t phase;
    uint64_t requested_route_id;
    uint64_t selected_route_id;
    uint64_t observed_route_id;
    uint64_t elapsed_us;
    uint64_t work_units;
    uint64_t prompt_tokens_delta;
    uint64_t generated_tokens_delta;
    uint64_t artifact_bytes;
    uint64_t resident_weight_bytes;
    uint64_t kv_bytes;
    int64_t kv_delta_bytes;
    uint64_t runtime_bytes;
    uint64_t peak_rss_bytes;
    uint64_t memory_read_bytes;
    uint64_t memory_write_bytes;
    uint64_t energy_delta_uj;
    uint32_t energy_confidence;
    int32_t thermal_headroom_millic;
    int32_t thermal_slope_millic_per_min;
    bool throttled;
    uint64_t active_envelopes;
    uint64_t passed_envelopes;
    uint32_t validation;
    uint32_t lifecycle;
    uint32_t live_handles;
    uint32_t live_threads;
    uint32_t live_file_descriptors;
    uint32_t live_services;
    uint32_t live_jobs;
    uint32_t live_wake_locks;
    uint64_t mapped_model_bytes;
    bool output_checked;
    bool output_valid;
    int32_t error_code;
    uint32_t error_envelope;
} geniex_XrayEvent;

typedef struct {
    uint32_t schema_version;
    bool valid;
    uint64_t observed_phases;
    uint64_t missing_phases;
    uint64_t failed_envelopes;
    uint64_t execution_id;
    uint64_t module_id;
    uint32_t module_version;
    uint64_t input_hash;
    uint64_t explicit_state_hash;
    uint64_t requested_route_id;
    uint64_t selected_route_id;
    uint64_t observed_route_id;
    uint64_t end_to_end_us;
    uint64_t prompt_tokens;
    uint64_t generated_tokens;
    uint64_t energy_uj;
    uint32_t energy_confidence;
    bool energy_qualified;
    double tokens_per_joule;
    int32_t minimum_thermal_headroom_millic;
    int32_t maximum_thermal_slope_millic_per_min;
    bool throttled;
    bool validation_failed;
    bool validation_complete;
    bool identity_complete;
    bool identity_consistent;
    bool output_valid;
    uint32_t final_lifecycle;
    uint64_t residue_count;
    int32_t first_error_code;
} geniex_XrayLapResult;

typedef struct geniex_XrayScorer geniex_XrayScorer;

geniex_XrayScorer* geniex_xray_scorer_create(void);
void geniex_xray_scorer_destroy(geniex_XrayScorer* scorer);
int32_t geniex_xray_scorer_record(geniex_XrayScorer* scorer, const geniex_XrayEvent* event);
int32_t geniex_xray_scorer_finish(const geniex_XrayScorer* scorer, geniex_XrayLapResult* result);

#ifdef __cplusplus
}
#endif
