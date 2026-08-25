// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "htp_session.h"

#include <atomic>

#include "ggml-backend.h"
#include "logging.h"

namespace geniex::htp {

namespace {
std::atomic<int> g_htp_refcount{0};

using htp_reg_fn = void (*)(ggml_backend_reg_t);

void call_htp_proc(const char* name, const char* log_msg) {
    auto* reg = ggml_backend_reg_by_name("HTP");
    if (!reg) return;
    auto fn = reinterpret_cast<htp_reg_fn>(ggml_backend_reg_get_proc_address(reg, name));
    if (!fn) return;
    GENIEX_LOG_DEBUG("{}", log_msg);
    fn(reg);
}
}  // namespace

void reacquire_before_load() {
    call_htp_proc("ggml_backend_hexagon_reacquire_sessions", "Reacquiring HTP sessions before llama.cpp load");
}

bool htp_backend_present() { return ggml_backend_reg_by_name("HTP") != nullptr; }

void release_sessions_if_idle() {
    if (g_htp_refcount.load(std::memory_order_acquire) != 0) return;
    call_htp_proc("ggml_backend_hexagon_release_sessions", "Releasing HTP sessions for foreign-plugin handoff");
}

void SessionGuard::mark_htp() {
    if (uses_htp_) return;
    uses_htp_ = true;
    g_htp_refcount.fetch_add(1, std::memory_order_acq_rel);
}

void SessionGuard::release_ref() {
    if (!uses_htp_) return;
    g_htp_refcount.fetch_sub(1, std::memory_order_acq_rel);
}

}  // namespace geniex::htp
