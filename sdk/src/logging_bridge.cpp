// Trampoline used by the Rust model manager to reach the geniex log
// callback (`geniex_log`, declared in sdk/include/logging.h).
//
// Rust's `extern "C" { static geniex_log: ... }` works fine on ELF
// targets but lld-link cannot resolve it across object files on Windows
// without dllimport metadata, which Rust cannot emit for a static
// reference. A plain function call is portable, so the Rust side calls
// geniex_model_log_emit() and we dispatch from here.

#include "logging.h"

extern "C" void geniex_model_log_emit(int level, const char* msg) {
    if (geniex_log != nullptr && msg != nullptr) {
        geniex_log(static_cast<geniex_LogLevel>(level), msg);
    }
}
