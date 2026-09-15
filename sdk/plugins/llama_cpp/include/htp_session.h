// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "geniex.h"

namespace geniex {

// A QAIRT plugin spun up after llama.cpp collides on the same CDSP domain
// with llama.cpp's still-open FastRPC handles ("Failed to create device:
// 1002"/1007). Release only on plugin handoff — cycling release/reacquire
// per llama.cpp load accumulates DSP-side state that fails dspqueue_read
// with 0x0d after ~20 handoffs.
namespace htp {

void reacquire_before_load();

bool htp_backend_present();

// Close all HTP sessions iff no SessionGuard is holding a reference.
void release_sessions_if_idle();

// Change the DCVS/HMX power mode requested for HTP sessions (see
// sdk/patches/llama-hexagon-power-mode-setter.patch). Only affects sessions
// created after this call: if a session is already open (a SessionGuard is
// still holding a reference), the new mode is logged as deferred and only
// takes effect once that session is torn down and reacquired. Call before
// reacquire_before_load() so a released session picks up the new mode.
void set_power_mode(geniex_PowerMode mode);

class SessionGuard {
   public:
    SessionGuard() = default;
    ~SessionGuard() { release_ref(); }

    SessionGuard(const SessionGuard&)            = delete;
    SessionGuard& operator=(const SessionGuard&) = delete;

    void mark_htp();

    bool uses_htp() const { return uses_htp_; }

   private:
    void release_ref();

    bool uses_htp_ = false;
};

}  // namespace htp
}  // namespace geniex
