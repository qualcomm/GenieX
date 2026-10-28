// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "geniex.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Additive TTS API. Call geniex_init first and destroy all handles before deinit.
 * A handle serves one synchronous synthesis at a time. cancel may be called from
 * another thread or the audio callback. Do not destroy or deinit during a call.
 * All strings are UTF-8 and borrowed for the duration of the call.
 */
typedef struct geniex_TTS geniex_TTS;

/** Mono float32 PCM in [-1,1]. Borrowed until callback returns. Return false to
 * stop generation. The callback runs synchronously on the synthesizing thread.
 * Do not throw across the callback or recursively synthesize on the same handle.
 */
typedef bool (*geniex_audio_callback)(const float* pcm, uint32_t samples, uint32_t sample_rate, void* user_data);

#define GENIEX_TTS_VERIFY_NPU 1u

typedef struct {
    uint32_t    struct_size; /**< sizeof(geniex_TtsCreateInput) */
    uint32_t    flags;       /**< 0 or GENIEX_TTS_VERIFY_NPU */
    const char* plugin_id;
    const char* model_path;   /**< Compiled bundle, never arbitrary code from a download */
    const char* device;       /**< Phonon currently accepts only "npu"; no fallback */
    const char* runtime_path; /**< Optional assertion of the global QAIRT selection, not a per-model override.
                               * Set geniex_set_qairt_runtime_path before geniex_init. NULL/empty inherits it. */
    const char* text_library; /**< Trusted Phonon native text frontend library */
    const char* language;
    const char* soc_model; /**< Must match the compiled bundle */
} geniex_TtsCreateInput;

typedef struct {
    uint32_t              struct_size; /**< sizeof(geniex_TtsSynthesizeInput) */
    const char*           text;
    const char*           voice; /**< NULL or empty selects the bundle default */
    geniex_audio_callback on_audio;
    void*                 user_data;
} geniex_TtsSynthesizeInput;

typedef struct {
    uint32_t struct_size; /**< Set to sizeof(geniex_TtsSynthesizeOutput) before calling */
    uint32_t sample_rate;
    uint64_t samples;
    double   first_audio_ms; /**< Text submission to first PCM callback, excludes model load */
    double   generation_ms;  /**< Includes synchronous callback time, excludes model load */
    double   prefill_npu_ms; /**< Per-request deltas, zero if profiling is disabled */
    double   step_npu_ms;
    uint32_t prefill_calls;
    uint32_t step_calls;
} geniex_TtsSynthesizeOutput;

GENIEX_API int32_t geniex_tts_create(const geniex_TtsCreateInput* input, geniex_TTS** out_handle);
GENIEX_API int32_t geniex_tts_destroy(geniex_TTS* handle);
GENIEX_API int32_t geniex_tts_synthesize(
    geniex_TTS* handle, const geniex_TtsSynthesizeInput* input, geniex_TtsSynthesizeOutput* output);
GENIEX_API int32_t geniex_tts_cancel(geniex_TTS* handle);

#ifdef __cplusplus
}
#endif
