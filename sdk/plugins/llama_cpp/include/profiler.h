
#pragma once

#include <chrono>
#include <cstdint>

#include "geniex.h"

using ProfileData = geniex_ProfileData;

namespace common {

enum StopReason {
    GENIEX_STOP_REASON_UNKNOWN = 0,
    GENIEX_STOP_REASON_EOS,
    GENIEX_STOP_REASON_LENGTH,
    GENIEX_STOP_REASON_USER,
    GENIEX_STOP_REASON_STOP_SEQUENCE,
    GENIEX_STOP_REASON_COMPLETED
};

class Profiler {
   public:
    Profiler() { start(); }

    void       prompt_start();
    void       prompt_end();
    void       decode_start();
    void       decode_end();
    void       record_ttft();
    void       update_prompt_tokens(uint32_t);
    void       update_generated_tokens(uint32_t);
    void       set_stop_reason(StopReason);
    StopReason get_stop_reason() const;
    void       end();
    void       to_profile_data(ProfileData& data);

   private:
    using timestamp = std::chrono::steady_clock::time_point;
    using clock     = std::chrono::steady_clock;

    static int64_t to_us(std::chrono::nanoseconds);
    void           start();

    timestamp start_time{};
    timestamp prompt_start_time{};
    timestamp prompt_end_time{};
    timestamp decode_start_time{};
    timestamp decode_end_time{};
    timestamp first_token_time{};
    timestamp end_time{};

    bool       ttft_recorded    = false;
    StopReason stop_reason      = StopReason::GENIEX_STOP_REASON_UNKNOWN;
    uint32_t   prompt_tokens    = 0;
    uint32_t   generated_tokens = 0;
};

}  // namespace common