#include "htp_session.h"

#include <atomic>
#include <cstring>

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

bool devices_include_htp(ggml_backend_dev_t* device_array) {
    if (!device_array) return false;
    for (size_t i = 0; device_array[i] != nullptr; ++i) {
        auto*       reg = ggml_backend_dev_backend_reg(device_array[i]);
        const char* rn  = reg ? ggml_backend_reg_name(reg) : nullptr;
        if (rn && strcmp(rn, "HTP") == 0) return true;
    }
    return false;
}

void SessionGuard::mark_htp() {
    if (uses_htp_) return;
    uses_htp_ = true;
    g_htp_refcount.fetch_add(1, std::memory_order_acq_rel);
}

void SessionGuard::release_if_last() {
    if (!uses_htp_) return;
    if (g_htp_refcount.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
    call_htp_proc(
        "ggml_backend_hexagon_release_sessions", "Releasing HTP sessions on last llama.cpp HTP handle destroy");
}

}  // namespace geniex::htp
