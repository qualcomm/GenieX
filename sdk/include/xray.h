// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace geniex::xray {

inline constexpr uint32_t kSchemaVersion = 1;

enum class Mode : uint8_t { Off = 0, Light = 1, Deep = 2 };

enum class Phase : uint8_t {
    Startup = 0,
    Residency,
    Prefill,
    Decode,
    ContextGrowth,
    ThermalEquilibrium,
    Stop,
    Unload,
    Recovery,
    Count,
};

inline constexpr size_t kPhaseCount = static_cast<size_t>(Phase::Count);
inline constexpr uint64_t phase_bit(Phase phase) {
    return uint64_t{1} << static_cast<uint8_t>(phase);
}
inline constexpr uint64_t kFullLapPhases =
    phase_bit(Phase::Startup) | phase_bit(Phase::Residency) |
    phase_bit(Phase::Prefill) | phase_bit(Phase::Decode) |
    phase_bit(Phase::ContextGrowth) | phase_bit(Phase::ThermalEquilibrium) |
    phase_bit(Phase::Stop) | phase_bit(Phase::Unload) |
    phase_bit(Phase::Recovery);

enum class Envelope : uint8_t {
    Artifact = 0,
    Storage,
    PeakRam,
    ResidentWeights,
    DynamicKv,
    Context,
    Runtime,
    Safety,
    Backend,
    Thermal,
    Energy,
    OutputValidity,
    Lifecycle,
    Residue,
};

inline constexpr uint64_t envelope_bit(Envelope envelope) {
    return uint64_t{1} << static_cast<uint8_t>(envelope);
}

enum class Validation : uint8_t { Unknown = 0, Pass = 1, Fail = 2 };
enum class Lifecycle : uint8_t {
    Unknown = 0,
    Starting,
    Resident,
    Running,
    Stopping,
    Unloading,
    Recovering,
    Released,
    Failed,
};
enum class EnergyConfidence : uint8_t {
    Unavailable = 0,
    Estimated,
    Measured,
    Calibrated,
};

struct CausalError {
    int32_t code{0};
    Envelope envelope{Envelope::Runtime};
    Phase phase{Phase::Startup};
};

struct Event {
    uint32_t schema_version{kSchemaVersion};
    uint64_t timestamp_us{0};
    uint64_t execution_id{0};
    uint64_t module_id{0};
    uint32_t module_version{0};
    uint64_t input_hash{0};
    uint64_t explicit_state_hash{0};
    Phase phase{Phase::Startup};
    uint64_t requested_route_id{0};
    uint64_t selected_route_id{0};
    uint64_t observed_route_id{0};

    uint64_t elapsed_us{0};
    uint64_t work_units{0};
    uint64_t prompt_tokens_delta{0};
    uint64_t generated_tokens_delta{0};

    uint64_t artifact_bytes{0};
    uint64_t resident_weight_bytes{0};
    uint64_t kv_bytes{0};
    int64_t kv_delta_bytes{0};
    uint64_t runtime_bytes{0};
    uint64_t peak_rss_bytes{0};
    uint64_t memory_read_bytes{0};
    uint64_t memory_write_bytes{0};

    uint64_t energy_delta_uj{0};
    EnergyConfidence energy_confidence{EnergyConfidence::Unavailable};
    int32_t thermal_headroom_millic{std::numeric_limits<int32_t>::max()};
    int32_t thermal_slope_millic_per_min{0};
    bool throttled{false};

    uint64_t active_envelopes{0};
    uint64_t passed_envelopes{0};
    Validation validation{Validation::Unknown};
    Lifecycle lifecycle{Lifecycle::Unknown};

    uint32_t live_handles{0};
    uint32_t live_threads{0};
    uint32_t live_file_descriptors{0};
    uint32_t live_services{0};
    uint32_t live_jobs{0};
    uint32_t live_wake_locks{0};
    uint64_t mapped_model_bytes{0};
    bool output_checked{false};
    bool output_valid{false};
    CausalError error{};
};

struct CounterSnapshot {
    std::array<uint64_t, kPhaseCount> events{};
    std::array<uint64_t, kPhaseCount> elapsed_us{};
    std::array<uint64_t, kPhaseCount> work_units{};
    uint64_t peak_rss_bytes{0};
    uint64_t max_kv_bytes{0};
    uint64_t energy_uj{0};
    uint64_t errors{0};
};

class Counters {
public:
    Counters() {
        for (auto& value : events_) value.store(0, std::memory_order_relaxed);
        for (auto& value : elapsed_us_) value.store(0, std::memory_order_relaxed);
        for (auto& value : work_units_) value.store(0, std::memory_order_relaxed);
    }

    void record(const Event& event) noexcept {
        const auto index = static_cast<size_t>(event.phase);
        if (index >= kPhaseCount) return;
        events_[index].fetch_add(1, std::memory_order_relaxed);
        elapsed_us_[index].fetch_add(event.elapsed_us, std::memory_order_relaxed);
        work_units_[index].fetch_add(event.work_units, std::memory_order_relaxed);
        atomic_max(peak_rss_bytes_, event.peak_rss_bytes);
        atomic_max(max_kv_bytes_, event.kv_bytes);
        energy_uj_.fetch_add(event.energy_delta_uj, std::memory_order_relaxed);
        if (event.error.code != 0 || event.validation == Validation::Fail) {
            errors_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    CounterSnapshot snapshot() const noexcept {
        CounterSnapshot result;
        for (size_t i = 0; i < kPhaseCount; ++i) {
            result.events[i] = events_[i].load(std::memory_order_relaxed);
            result.elapsed_us[i] = elapsed_us_[i].load(std::memory_order_relaxed);
            result.work_units[i] = work_units_[i].load(std::memory_order_relaxed);
        }
        result.peak_rss_bytes = peak_rss_bytes_.load(std::memory_order_relaxed);
        result.max_kv_bytes = max_kv_bytes_.load(std::memory_order_relaxed);
        result.energy_uj = energy_uj_.load(std::memory_order_relaxed);
        result.errors = errors_.load(std::memory_order_relaxed);
        return result;
    }

private:
    static void atomic_max(std::atomic<uint64_t>& target, uint64_t value) noexcept {
        auto current = target.load(std::memory_order_relaxed);
        while (current < value &&
               !target.compare_exchange_weak(
                   current, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
        }
    }

    std::array<std::atomic<uint64_t>, kPhaseCount> events_;
    std::array<std::atomic<uint64_t>, kPhaseCount> elapsed_us_;
    std::array<std::atomic<uint64_t>, kPhaseCount> work_units_;
    std::atomic<uint64_t> peak_rss_bytes_{0};
    std::atomic<uint64_t> max_kv_bytes_{0};
    std::atomic<uint64_t> energy_uj_{0};
    std::atomic<uint64_t> errors_{0};
};

class Sink {
public:
    virtual ~Sink() = default;
    virtual void emit(const Event& event) noexcept = 0;
};

class Recorder {
public:
    explicit Recorder(Mode mode = Mode::Light, Sink* sink = nullptr) noexcept
        : mode_(mode), sink_(sink) {}

    void set_mode(Mode mode) noexcept { mode_.store(mode, std::memory_order_relaxed); }
    Mode mode() const noexcept { return mode_.load(std::memory_order_relaxed); }

    void record(const Event& event) noexcept {
        const auto current = mode();
        if (current == Mode::Off) return;
        counters_.record(event);
        if (current == Mode::Deep && sink_) sink_->emit(event);
    }

    CounterSnapshot snapshot() const noexcept { return counters_.snapshot(); }

private:
    std::atomic<Mode> mode_;
    Sink* sink_;
    Counters counters_;
};

struct LapResult {
    uint32_t schema_version{kSchemaVersion};
    bool valid{false};
    uint64_t observed_phases{0};
    uint64_t missing_phases{kFullLapPhases};
    uint64_t failed_envelopes{0};
    uint64_t execution_id{0};
    uint64_t module_id{0};
    uint32_t module_version{0};
    uint64_t input_hash{0};
    uint64_t explicit_state_hash{0};
    uint64_t requested_route_id{0};
    uint64_t selected_route_id{0};
    uint64_t observed_route_id{0};
    uint64_t end_to_end_us{0};
    uint64_t prompt_tokens{0};
    uint64_t generated_tokens{0};
    uint64_t energy_uj{0};
    EnergyConfidence energy_confidence{EnergyConfidence::Unavailable};
    bool energy_qualified{false};
    double tokens_per_joule{0.0};
    int32_t minimum_thermal_headroom_millic{std::numeric_limits<int32_t>::max()};
    int32_t maximum_thermal_slope_millic_per_min{0};
    bool throttled{false};
    bool validation_failed{false};
    bool validation_complete{false};
    bool identity_complete{false};
    bool identity_consistent{false};
    bool output_valid{false};
    Lifecycle final_lifecycle{Lifecycle::Unknown};
    uint64_t residue_count{0};
    CausalError first_error{};

    bool better_than(const LapResult& other) const noexcept {
        if (valid != other.valid) return valid;
        if (!valid) {
            if (failed_envelopes != other.failed_envelopes) {
                return failed_envelopes < other.failed_envelopes;
            }
            return missing_phases < other.missing_phases;
        }
        if (end_to_end_us != other.end_to_end_us) return end_to_end_us < other.end_to_end_us;
        if (tokens_per_joule != other.tokens_per_joule) {
            return tokens_per_joule > other.tokens_per_joule;
        }
        return minimum_thermal_headroom_millic >
               other.minimum_thermal_headroom_millic;
    }
};

class LapScorer {
public:
    void record(const Event& event) noexcept {
        observed_phases_ |= phase_bit(event.phase);
        failed_envelopes_ |= event.active_envelopes & ~event.passed_envelopes;
        if (event.validation == Validation::Fail) {
            validation_failed_ = true;
            failed_envelopes_ |= envelope_bit(event.error.envelope);
        }
        validation_complete_ =
            validation_complete_ && event.validation != Validation::Unknown;

        const bool event_identity_complete =
            event.schema_version == kSchemaVersion && event.execution_id != 0 &&
            event.module_id != 0 &&
            event.module_version != 0 && event.input_hash != 0 &&
            event.explicit_state_hash != 0 && event.requested_route_id != 0 &&
            event.selected_route_id != 0 && event.observed_route_id != 0;
        identity_complete_ = identity_complete_ && event_identity_complete;
        if (!identity_initialized_) {
            execution_id_ = event.execution_id;
            module_id_ = event.module_id;
            module_version_ = event.module_version;
            input_hash_ = event.input_hash;
            explicit_state_hash_ = event.explicit_state_hash;
            requested_route_id_ = event.requested_route_id;
            selected_route_id_ = event.selected_route_id;
            observed_route_id_ = event.observed_route_id;
            identity_initialized_ = true;
        } else if (execution_id_ != event.execution_id || module_id_ != event.module_id ||
                   module_version_ != event.module_version ||
                   input_hash_ != event.input_hash ||
                   requested_route_id_ != event.requested_route_id ||
                   selected_route_id_ != event.selected_route_id ||
                   observed_route_id_ != event.observed_route_id) {
            identity_consistent_ = false;
        }

        if (!started_ || event.timestamp_us < first_timestamp_us_) {
            first_timestamp_us_ = event.timestamp_us;
            started_ = true;
        }
        if (event.timestamp_us + event.elapsed_us > last_timestamp_us_) {
            last_timestamp_us_ = event.timestamp_us + event.elapsed_us;
        }

        prompt_tokens_ += event.prompt_tokens_delta;
        generated_tokens_ += event.generated_tokens_delta;
        energy_uj_ += event.energy_delta_uj;
        if (event.energy_delta_uj != 0) {
            if (!energy_observed_ || event.energy_confidence < energy_confidence_) {
                energy_confidence_ = event.energy_confidence;
            }
            energy_observed_ = true;
        }
        throttled_ = throttled_ || event.throttled;
        if (event.output_checked) {
            output_observed_ = true;
            output_invalid_ = output_invalid_ || !event.output_valid;
            if (!event.output_valid) {
                failed_envelopes_ |= envelope_bit(Envelope::OutputValidity);
            }
        }
        final_lifecycle_ = event.lifecycle;

        if (event.thermal_headroom_millic != std::numeric_limits<int32_t>::max() &&
            event.thermal_headroom_millic < minimum_thermal_headroom_millic_) {
            minimum_thermal_headroom_millic_ = event.thermal_headroom_millic;
        }
        const int32_t absolute_slope =
            event.thermal_slope_millic_per_min < 0
                ? -event.thermal_slope_millic_per_min
                : event.thermal_slope_millic_per_min;
        if (absolute_slope > maximum_thermal_slope_millic_per_min_) {
            maximum_thermal_slope_millic_per_min_ = absolute_slope;
        }

        if (event.phase == Phase::Recovery) {
            residue_count_ =
                static_cast<uint64_t>(event.live_handles) + event.live_threads +
                event.live_file_descriptors + event.live_services + event.live_jobs +
                event.live_wake_locks + event.mapped_model_bytes;
        }
        if (first_error_.code == 0 && event.error.code != 0) first_error_ = event.error;
    }

    LapResult finish() const noexcept {
        LapResult result;
        result.observed_phases = observed_phases_;
        result.missing_phases = kFullLapPhases & ~observed_phases_;
        result.failed_envelopes = failed_envelopes_;
        result.execution_id = execution_id_;
        result.module_id = module_id_;
        result.module_version = module_version_;
        result.input_hash = input_hash_;
        result.explicit_state_hash = explicit_state_hash_;
        result.requested_route_id = requested_route_id_;
        result.selected_route_id = selected_route_id_;
        result.observed_route_id = observed_route_id_;
        result.end_to_end_us =
            started_ && last_timestamp_us_ >= first_timestamp_us_
                ? last_timestamp_us_ - first_timestamp_us_
                : 0;
        result.prompt_tokens = prompt_tokens_;
        result.generated_tokens = generated_tokens_;
        result.energy_uj = energy_uj_;
        result.energy_confidence =
            energy_observed_ ? energy_confidence_ : EnergyConfidence::Unavailable;
        result.energy_qualified =
            energy_uj_ != 0 && result.energy_confidence >= EnergyConfidence::Measured;
        result.tokens_per_joule =
            !result.energy_qualified
                ? 0.0
                : static_cast<double>(generated_tokens_) * 1000000.0 /
                      static_cast<double>(energy_uj_);
        result.minimum_thermal_headroom_millic = minimum_thermal_headroom_millic_;
        result.maximum_thermal_slope_millic_per_min =
            maximum_thermal_slope_millic_per_min_;
        result.throttled = throttled_;
        result.validation_failed = validation_failed_;
        result.validation_complete = validation_complete_;
        result.identity_complete = identity_complete_;
        result.identity_consistent = identity_consistent_;
        result.output_valid = output_observed_ && !output_invalid_;
        result.final_lifecycle = final_lifecycle_;
        result.residue_count = residue_count_;
        result.first_error = first_error_;
        if (!result.energy_qualified) {
            result.failed_envelopes |= envelope_bit(Envelope::Energy);
        }
        if (!result.identity_complete || !result.identity_consistent) {
            result.failed_envelopes |= envelope_bit(Envelope::Safety);
        }
        if (!result.validation_complete) {
            result.failed_envelopes |= envelope_bit(Envelope::Runtime);
        }
        result.valid =
            result.missing_phases == 0 && result.failed_envelopes == 0 &&
            !result.throttled && !result.validation_failed && result.validation_complete &&
            result.identity_complete && result.identity_consistent &&
            result.energy_qualified && result.output_valid &&
            result.final_lifecycle == Lifecycle::Released &&
            result.residue_count == 0 && result.first_error.code == 0;
        return result;
    }

private:
    bool started_{false};
    uint64_t first_timestamp_us_{0};
    uint64_t last_timestamp_us_{0};
    uint64_t observed_phases_{0};
    uint64_t failed_envelopes_{0};
    uint64_t prompt_tokens_{0};
    uint64_t generated_tokens_{0};
    uint64_t energy_uj_{0};
    EnergyConfidence energy_confidence_{EnergyConfidence::Calibrated};
    bool energy_observed_{false};
    int32_t minimum_thermal_headroom_millic_{std::numeric_limits<int32_t>::max()};
    int32_t maximum_thermal_slope_millic_per_min_{0};
    bool throttled_{false};
    bool validation_failed_{false};
    bool validation_complete_{true};
    bool identity_initialized_{false};
    bool identity_complete_{true};
    bool identity_consistent_{true};
    uint64_t execution_id_{0};
    uint64_t module_id_{0};
    uint32_t module_version_{0};
    uint64_t input_hash_{0};
    uint64_t explicit_state_hash_{0};
    uint64_t requested_route_id_{0};
    uint64_t selected_route_id_{0};
    uint64_t observed_route_id_{0};
    bool output_observed_{false};
    bool output_invalid_{false};
    Lifecycle final_lifecycle_{Lifecycle::Unknown};
    uint64_t residue_count_{0};
    CausalError first_error_{};
};

}  // namespace geniex::xray
