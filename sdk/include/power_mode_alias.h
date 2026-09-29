// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// Header-only power-mode alias table for C/C++ callers that build a
// geniex_ModelConfig directly (sdk/benchmark, examples/, Android/JNI).
// Plain C99, so it works from both .c and .cpp translation units without
// needing an extern "C" guard (each includer gets its own static copy).
// Go and Python resolve their own strings natively instead of including this.

#ifndef GENIEX_POWER_MODE_ALIAS_H
#define GENIEX_POWER_MODE_ALIAS_H

#include <ctype.h>
#include <stddef.h>
#include <string.h>

#include "geniex.h"

static inline int geniex_power_mode_from_alias(const char* mode, geniex_PowerMode* out) {
    static const struct {
        const char*      name;
        geniex_PowerMode value;
    } kAliases[] = {
        {"low_power_saver", GENIEX_POWER_MODE_LOW_POWER_SAVER},
        {"power_saver", GENIEX_POWER_MODE_POWER_SAVER},
        {"high_power_saver", GENIEX_POWER_MODE_HIGH_POWER_SAVER},
        {"low_balanced", GENIEX_POWER_MODE_LOW_BALANCED},
        {"balanced", GENIEX_POWER_MODE_BALANCED},
        {"high_performance", GENIEX_POWER_MODE_HIGH_PERFORMANCE},
        {"sustained_high_performance", GENIEX_POWER_MODE_SUSTAINED_HIGH_PERFORMANCE},
        {"burst", GENIEX_POWER_MODE_BURST},
    };
    char   normalized[32];
    size_t len = mode ? strlen(mode) : 0;

    if (len == 0) {
        *out = GENIEX_POWER_MODE_BURST;
        return 1;
    }

    size_t start = 0, end = len;
    while (start < len && isspace((unsigned char)mode[start])) ++start;
    while (end > start && isspace((unsigned char)mode[end - 1])) --end;
    len = end - start;
    if (len == 0 || len >= sizeof(normalized)) {
        *out = GENIEX_POWER_MODE_BURST;
        return len == 0;
    }
    for (size_t i = 0; i < len; ++i) normalized[i] = (char)tolower((unsigned char)mode[start + i]);
    normalized[len] = '\0';

    if (strcmp(normalized, "default") == 0) {
        *out = GENIEX_POWER_MODE_BURST;
        return 1;
    }
    for (size_t i = 0; i < sizeof(kAliases) / sizeof(kAliases[0]); ++i) {
        if (strcmp(normalized, kAliases[i].name) == 0) {
            *out = kAliases[i].value;
            return 1;
        }
    }
    return 0;
}

#endif  // GENIEX_POWER_MODE_ALIAS_H
