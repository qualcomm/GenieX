// SPDX-License-Identifier: BSD-3-Clause
package geniex_sdk

/*
#include <stdlib.h>
#include "geniex_tts.h"
extern bool go_tts_audio(float*, uint32_t, uint32_t, uintptr_t);
static bool tts_audio_bridge(const float* pcm, uint32_t n, uint32_t rate, void* ctx) {
    return go_tts_audio((float*)pcm, n, rate, (uintptr_t)ctx);
}
static int32_t tts_run(geniex_TTS* h, const char* text, const char* voice, uintptr_t ctx,
                       geniex_TtsSynthesizeOutput* out) {
    geniex_TtsSynthesizeInput in = {sizeof(in), text, voice, tts_audio_bridge, (void*)ctx};
    return geniex_tts_synthesize(h, &in, out);
}
*/
import "C"

import (
	"fmt"
	"runtime/cgo"
	"strings"
	"sync"
	"unsafe"
)

type TtsCreateInput struct {
	// RuntimePath asserts the shared QAIRT selection; call SetQairtRuntimePath before Init.
	PluginID, ModelPath, Device, RuntimePath, TextLibrary, Language, SocModel string
	VerifyNPU                                                                 bool
}
type TtsResult struct {
	SampleRate                                          uint32
	Samples                                             uint64
	FirstAudioMs, GenerationMs, PrefillNPUMs, StepNPUMs float64
	PrefillCalls, StepCalls                             uint32
	Cancelled                                           bool
}
type TTS struct {
	handle *C.geniex_TTS
	mu     sync.Mutex
	// Protect handle lifetime during Cancel and Close, without blocking synthesis.
	life sync.RWMutex
}
type ttsCallback struct {
	fn         func([]float32, uint32) bool
	panicValue any
}

//export go_tts_audio
func go_tts_audio(pcm *C.float, count C.uint32_t, rate C.uint32_t, context C.uintptr_t) (more C.bool) {
	cb := cgo.Handle(context).Value().(*ttsCallback)
	defer func() {
		if value := recover(); value != nil {
			cb.panicValue = value
			more = false
		}
	}()
	data := append([]float32(nil), unsafe.Slice((*float32)(unsafe.Pointer(pcm)), int(count))...)
	return C.bool(cb.fn(data, uint32(rate)))
}

func NewTTS(input TtsCreateInput) (*TTS, error) {
	values := []string{input.PluginID, input.ModelPath, input.Device, input.RuntimePath,
		input.TextLibrary, input.Language, input.SocModel}
	cvalues := make([]*C.char, len(values))
	for i, v := range values {
		if strings.ContainsRune(v, '\x00') {
			return nil, ErrCommonInvalidInput
		}
		cvalues[i] = C.CString(v)
		defer C.free(unsafe.Pointer(cvalues[i]))
	}
	in := C.geniex_TtsCreateInput{struct_size: C.sizeof_geniex_TtsCreateInput,
		plugin_id: cvalues[0], model_path: cvalues[1], device: cvalues[2], runtime_path: cvalues[3],
		text_library: cvalues[4], language: cvalues[5], soc_model: cvalues[6]}
	if input.VerifyNPU {
		in.flags = C.GENIEX_TTS_VERIFY_NPU
	}
	t := &TTS{}
	if code := C.geniex_tts_create(&in, &t.handle); code != 0 {
		return nil, SDKError(code)
	}
	return t, nil
}

// Synthesize calls onAudio synchronously with copied PCM. False cancels.
// Close must not be called from onAudio; Cancel may be called there.
func (t *TTS) Synthesize(text, voice string, onAudio func([]float32, uint32) bool) (TtsResult, error) {
	if strings.ContainsRune(text, '\x00') || strings.ContainsRune(voice, '\x00') {
		return TtsResult{}, ErrCommonInvalidInput
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.life.RLock()
	defer t.life.RUnlock()
	if t.handle == nil || onAudio == nil {
		return TtsResult{}, ErrCommonInvalidInput
	}
	ctext, cvoice := C.CString(text), C.CString(voice)
	defer C.free(unsafe.Pointer(ctext))
	defer C.free(unsafe.Pointer(cvoice))
	cb := &ttsCallback{fn: onAudio}
	ctx := cgo.NewHandle(cb)
	defer ctx.Delete()
	out := C.geniex_TtsSynthesizeOutput{struct_size: C.sizeof_geniex_TtsSynthesizeOutput}
	code := C.tts_run(t.handle, ctext, cvoice, C.uintptr_t(ctx), &out)
	result := TtsResult{uint32(out.sample_rate), uint64(out.samples), float64(out.first_audio_ms),
		float64(out.generation_ms), float64(out.prefill_npu_ms), float64(out.step_npu_ms),
		uint32(out.prefill_calls), uint32(out.step_calls), code == C.GENIEX_ERROR_COMMON_CANCELLED}
	if cb.panicValue != nil {
		return result, fmt.Errorf("TTS callback panicked: %v", cb.panicValue)
	}
	if code != 0 && !result.Cancelled {
		return result, SDKError(code)
	}
	return result, nil
}

func (t *TTS) Cancel() error {
	t.life.RLock()
	defer t.life.RUnlock()
	if t.handle == nil {
		return ErrCommonInvalidInput
	}
	if code := C.geniex_tts_cancel(t.handle); code != 0 {
		return SDKError(code)
	}
	return nil
}
func (t *TTS) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.life.Lock()
	defer t.life.Unlock()
	if t.handle == nil {
		return nil
	}
	code := C.geniex_tts_destroy(t.handle)
	t.handle = nil
	if code != 0 {
		return SDKError(code)
	}
	return nil
}
