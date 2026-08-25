// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

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
