// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package geniex_sdk

/*
#include <stdlib.h>
#include "geniex.h"
*/
import "C"

import (
	"unsafe"
)

/* ========================================================================== */
/*                              Utilities					  				 */
/* ========================================================================== */

func mlFree(ptr unsafe.Pointer) {
	C.geniex_free(ptr)
}

func sliceToCCharArray(slice []string) (**C.char, C.int32_t) {
	if len(slice) == 0 {
		return nil, 0
	}
	count := C.size_t(len(slice))
	raw := C.malloc(C.size_t(count) * C.size_t(unsafe.Sizeof(uintptr(0))))
	if raw == nil {
		panic("C.malloc failed")
	}

	cArray := unsafe.Slice((**C.char)(raw), len(slice))
	for i, s := range slice {
		cArray[i] = C.CString(s)
	}
	return (**C.char)(raw), C.int32_t(count)
}

func cCharArrayToSlice(ptr **C.char, count C.int32_t) []string {
	if ptr == nil || count == 0 {
		return nil
	}
	arr := unsafe.Slice(ptr, int(count))
	out := make([]string, count)
	for i, cstr := range arr {
		out[i] = C.GoString(cstr)
	}
	return out
}

// free the C char array in Go side allocated
func freeCCharArray(ptr **C.char, count C.int32_t) {
	if ptr == nil || count == 0 {
		return
	}
	arr := unsafe.Slice(ptr, int(count))
	for _, p := range arr {
		C.free(unsafe.Pointer(p))
	}
	C.free(unsafe.Pointer(ptr))
}

func mlFreeCCharArray(ptr **C.char, count C.int32_t) {
	if ptr == nil || count == 0 {
		return
	}
	arr := unsafe.Slice(ptr, int(count))
	for _, p := range arr {
		mlFree(unsafe.Pointer(p))
	}
	mlFree(unsafe.Pointer(ptr))
}

/* ========================================================================== */
/*                              Callbacks								  */
/* ========================================================================== */

// TODO: use this to manage callbacks
// type CallbackManager struct {
// 	mu      sync.RWMutex
// 	callbacks map[unsafe.Pointer]OnTokenCallback
// }

type OnTokenCallback func(token string) bool

var onToken OnTokenCallback = nil

//export go_generate_stream_on_token
func go_generate_stream_on_token(token *C.char, _ *C.void) C.bool {
	if onToken == nil {
		return C.bool(true)
	}
	return C.bool(onToken(C.GoString(token)))
}

var onASRTranscription ASRTranscriptionCallback = nil

//export go_asr_stream_on_transcription
func go_asr_stream_on_transcription(text *C.char, _ *C.void) {
	if onASRTranscription == nil {
		return
	}
	onASRTranscription(C.GoString(text), nil)
}

/* ========================================================================== */
/*                              Common Types & Utilities					  */
/* ========================================================================== */

type ProfileData struct {
	TTFT            int64
	PromptTime      int64
	DecodeTime      int64
	PromptTokens    int64
	GeneratedTokens int64
	AudioDuration   int64
	PrefillSpeed    float64
	DecodingSpeed   float64
	RealTimeFactor  float64
	StopReason      string
}

func (p ProfileData) TotalTokens() int64 {
	return p.PromptTokens + p.GeneratedTokens
}

func (p ProfileData) TotalTimeUs() int64 {
	return p.PromptTime + p.DecodeTime
}

func newProfileDataFromCPtr(c C.geniex_ProfileData) ProfileData {
	return ProfileData{
		TTFT:            int64(c.ttft),
		PromptTime:      int64(c.prompt_time),
		DecodeTime:      int64(c.decode_time),
		PromptTokens:    int64(c.prompt_tokens),
		GeneratedTokens: int64(c.generated_tokens),
		AudioDuration:   int64(c.audio_duration),
		PrefillSpeed:    float64(c.prefill_speed),
		DecodingSpeed:   float64(c.decoding_speed),
		RealTimeFactor:  float64(c.real_time_factor),
		StopReason:      C.GoString(c.stop_reason),
	}
}

/* ========================================================================== */
/*                              LANGUAGE MODELS (LLM) */
/* ========================================================================== */
type SamplerConfig struct {
	Temperature       float32
	TopP              float32
	TopK              int32
	MinP              float32
	RepetitionPenalty float32
	PresencePenalty   float32
	FrequencyPenalty  float32
	Seed              int32
	GrammarPath       string
	GrammarString     string
	EnableJson        bool

	C *C.geniex_SamplerConfig
}

func (sc SamplerConfig) toCPtr() *C.geniex_SamplerConfig {
	// Allocate C structure
	cPtr := (*C.geniex_SamplerConfig)(C.malloc(C.sizeof_geniex_SamplerConfig))
	*cPtr = C.geniex_SamplerConfig{}

	cPtr.temperature = C.float(sc.Temperature)
	cPtr.top_p = C.float(sc.TopP)
	cPtr.top_k = C.int32_t(sc.TopK)
	cPtr.min_p = C.float(sc.MinP)
	cPtr.repetition_penalty = C.float(sc.RepetitionPenalty)
	cPtr.presence_penalty = C.float(sc.PresencePenalty)
	cPtr.frequency_penalty = C.float(sc.FrequencyPenalty)
	cPtr.seed = C.int32_t(sc.Seed)
	cPtr.enable_json = C.bool(sc.EnableJson)

	if sc.GrammarPath != "" {
		cPtr.grammar_path = C.CString(sc.GrammarPath)
	}
	if sc.GrammarString != "" {
		cPtr.grammar_string = C.CString(sc.GrammarString)
	}

	return cPtr
}

func freeSamplerConfig(cPtr *C.geniex_SamplerConfig) {
	if cPtr != nil {
		if cPtr.grammar_path != nil {
			C.free(unsafe.Pointer(cPtr.grammar_path))
		}
		if cPtr.grammar_string != nil {
			C.free(unsafe.Pointer(cPtr.grammar_string))
		}
		C.free(unsafe.Pointer(cPtr))
	}
}

type GenerationConfig struct {
	MaxTokens      int32
	Stop           []string
	NPast          int32
	SamplerConfig  *SamplerConfig
	ImagePaths     []string
	ImageMaxLength int32
	AudioPaths     []string
}

func (gc GenerationConfig) toCPtr() *C.geniex_GenerationConfig {
	// Allocate C structure
	cPtr := (*C.geniex_GenerationConfig)(C.malloc(C.sizeof_geniex_GenerationConfig))
	*cPtr = C.geniex_GenerationConfig{}

	cPtr.max_tokens = C.int32_t(gc.MaxTokens)
	cPtr.n_past = C.int32_t(gc.NPast)
	cPtr.image_max_length = C.int32_t(gc.ImageMaxLength)
	if len(gc.Stop) > 0 {
		cPtr.stop, cPtr.stop_count = sliceToCCharArray(gc.Stop)
	}

	if len(gc.ImagePaths) > 0 {
		imagePaths, imageCount := sliceToCCharArray(gc.ImagePaths)
		cPtr.image_paths = (*C.geniex_Path)(imagePaths)
		cPtr.image_count = C.int32_t(imageCount)
	}

	if len(gc.AudioPaths) > 0 {
		audioPaths, audioCount := sliceToCCharArray(gc.AudioPaths)
		cPtr.audio_paths = (*C.geniex_Path)(audioPaths)
		cPtr.audio_count = C.int32_t(audioCount)
	}

	if gc.SamplerConfig != nil {
		cPtr.sampler_config = gc.SamplerConfig.toCPtr()
	}

	return cPtr
}

func freeGenerationConfig(ptr *C.geniex_GenerationConfig) {
	if ptr == nil {
		return
	}

	freeCCharArray(ptr.stop, ptr.stop_count)
	freeCCharArray((**C.char)(unsafe.Pointer(ptr.image_paths)), ptr.image_count)
	freeCCharArray((**C.char)(unsafe.Pointer(ptr.audio_paths)), ptr.audio_count)

	if ptr.sampler_config != nil {
		freeSamplerConfig(ptr.sampler_config)
	}
}

type ModelConfig struct {
	NCtx                int32
	NThreads            int32
	NThreadsBatch       int32
	NBatch              int32
	NUbatch             int32
	NSeqMax             int32
	NGpuLayers          int32
	ChatTemplatePath    string
	ChatTemplateContent string
}
