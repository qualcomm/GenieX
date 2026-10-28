// SPDX-License-Identifier: BSD-3-Clause
#include <stddef.h>

#include "geniex_tts.h"

_Static_assert(offsetof(geniex_TtsCreateInput, flags) == 4, "C ABI flags offset");
_Static_assert(offsetof(geniex_TtsSynthesizeOutput, samples) == 8, "C ABI sample offset");
int tts_c_header_compiles(void) { return sizeof(geniex_TtsSynthesizeOutput) == 56; }
