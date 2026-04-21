#include "profiler.h"

namespace common {

static const char* stop_reason_to_string(StopReason reason) {
    switch (reason) {
        case StopReason::GENIEX_STOP_REASON_EOS:
            return "eos";
        case StopReason::GENIEX_STOP_REASON_LENGTH:
            return "length";
        case StopReason::GENIEX_STOP_REASON_USER:
            return "user";
        case StopReason::GENIEX_STOP_REASON_STOP_SEQUENCE:
            return "stop_sequence";
        case StopReason::GENIEX_STOP_REASON_COMPLETED:
            return "completed";
        default:
            return "unknown";
    }
}

void Profiler::start() { start_time = clock::now(); }

void Profiler::prompt_start() { prompt_start_time = clock::now(); }

void Profiler::prompt_end() { prompt_end_time = clock::now(); }

void Profiler::decode_start() { decode_start_time = clock::now(); }

void Profiler::decode_end() { decode_end_time = clock::now(); }

void Profiler::record_ttft() {
    if (!ttft_recorded && first_token_time == timestamp{}) {
        first_token_time = clock::now();
        ttft_recorded    = true;
    }
}

void Profiler::update_prompt_tokens(uint32_t count) { prompt_tokens = count; }

void Profiler::update_generated_tokens(uint32_t count) { generated_tokens = count; }

void Profiler::set_stop_reason(StopReason reason) { stop_reason = reason; }

StopReason Profiler::get_stop_reason() const { return stop_reason; }

void Profiler::end() { end_time = clock::now(); }

void Profiler::to_profile_data(ProfileData& pd) {
    end();

    if (first_token_time != timestamp{}) {
        pd.ttft = to_us(first_token_time - start_time);
    }

    if (prompt_start_time != timestamp{} && prompt_end_time != timestamp{}) {
        pd.prompt_time = to_us(prompt_end_time - prompt_start_time);
    }

    if (decode_start_time != timestamp{} && decode_end_time != timestamp{}) {
        pd.decode_time = to_us(decode_end_time - decode_start_time);
    }

    pd.prompt_tokens    = prompt_tokens;
    pd.generated_tokens = generated_tokens;
    pd.stop_reason      = stop_reason_to_string(stop_reason);
}

int64_t Profiler::to_us(std::chrono::nanoseconds ns) {
    return std::chrono::duration_cast<std::chrono::microseconds>(ns).count();
}

}  // namespace common
