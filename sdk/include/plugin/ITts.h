// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "geniex_tts.h"

namespace geniex {
class ITts {
   public:
    virtual ~ITts()                                                                                        = default;
    virtual int32_t create(const geniex_TtsCreateInput* input)                                             = 0;
    virtual int32_t synthesize(const geniex_TtsSynthesizeInput* input, geniex_TtsSynthesizeOutput* output) = 0;
    virtual int32_t cancel()                                                                               = 0;
};

// Optional export: geniex_create_tts_v1. Versioned separately from Plugin so
// existing LLM/VLM plugin vtables and binaries remain unchanged.
using CreateTtsV1 = ITts* (*)();
}  // namespace geniex
