// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <string.h>

// A library's abort() (failed assert, std::terminate, heap corruption) raises
// SIGABRT. glibc's abort() force-resets the disposition and re-raises, so it
// cannot be intercepted from Go via os/signal — the handler goroutine never
// wins the race. A C signal handler runs synchronously in the aborting thread,
// so it reliably replaces the runtime's dump of every goroutine's stack.
//
// The handler prints a short line, then calls back into Go to print only the
// current goroutine's stack (runtime.Stack(buf, false)); chaining to the
// runtime's own handler instead would force-print every goroutine, which
// GOTRACEBACK cannot suppress on the cgo path.

// Exported from Go (see ml.go): prints the current goroutine's stack.
extern void go_dump_current(void);

#ifdef _WIN32

#include <io.h>       // _write
#include <process.h>  // _exit
#include <signal.h>
#include <stdlib.h>

static void geniex_fatal_handler(int sig) {
    (void)sig;
    static const char msg[] = "fatal: backend aborted (SIGABRT)\n";
    _write(2, msg, (unsigned int)(sizeof(msg) - 1));
    go_dump_current();
    _exit(3);  // MSVCRT abort() convention
}

void geniex_install_crash_handler(void) {
    // Suppress the "application error" dialog and Watson fault reporting;
    // otherwise abort() blocks waiting for a human to dismiss it.
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    signal(SIGABRT, geniex_fatal_handler);
    // SIGSEGV on Windows is an SEH exception, not reliably caught via signal();
    // it would need SetUnhandledExceptionFilter, omitted here.
}

#else  // POSIX: Linux / Android / macOS

#include <signal.h>
#include <unistd.h>

static void geniex_fatal_handler(int sig) {
    const char* msg;
    switch (sig) {
        case SIGABRT:
            msg = "fatal: backend aborted (SIGABRT)\n";
            break;
        case SIGSEGV:
            msg = "fatal: backend segfault (SIGSEGV)\n";
            break;
        default:
            msg = "fatal: backend crashed\n";
            break;
    }
    ssize_t n = write(STDERR_FILENO, msg, strlen(msg));
    (void)n;
    go_dump_current();
    _exit(134);  // 128 + SIGABRT
}

void geniex_install_crash_handler(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = geniex_fatal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND;  // handle once, then restore default to avoid recursion
    sigaction(SIGABRT, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);
}

#endif
