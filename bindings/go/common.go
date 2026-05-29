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

// LCOV_EXCL_START
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

// LCOV_EXCL_STOP

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
}

// LCOV_EXCL_START
func (sc SamplerConfig) toCPtr() *C.geniex_SamplerConfig {
	cPtr := (*C.geniex_SamplerConfig)(cMalloc(C.sizeof_geniex_SamplerConfig))
	*cPtr = C.geniex_SamplerConfig{
		temperature:        C.float(sc.Temperature),
		top_p:              C.float(sc.TopP),
		top_k:              C.int32_t(sc.TopK),
		min_p:              C.float(sc.MinP),
		repetition_penalty: C.float(sc.RepetitionPenalty),
		presence_penalty:   C.float(sc.PresencePenalty),
		frequency_penalty:  C.float(sc.FrequencyPenalty),
		seed:               C.int32_t(sc.Seed),
		grammar_path:       cStringIfSet(sc.GrammarPath),
		grammar_string:     cStringIfSet(sc.GrammarString),
		enable_json:        C.bool(sc.EnableJson),
	}
	return cPtr
}

func freeSamplerConfig(cPtr *C.geniex_SamplerConfig) {
	if cPtr == nil {
		return
	}
	cFreeIfSet(unsafe.Pointer(cPtr.grammar_path))
	cFreeIfSet(unsafe.Pointer(cPtr.grammar_string))
	C.free(unsafe.Pointer(cPtr))
}

// LCOV_EXCL_STOP

type GenerationConfig struct {
	MaxTokens      int32
	Stop           []string
	NPast          int32
	SamplerConfig  *SamplerConfig
	ImagePaths     []string
	ImageMaxLength int32
	AudioPaths     []string
}

// LCOV_EXCL_START
func (gc GenerationConfig) toCPtr() *C.geniex_GenerationConfig {
	cPtr := (*C.geniex_GenerationConfig)(cMalloc(C.sizeof_geniex_GenerationConfig))
	*cPtr = C.geniex_GenerationConfig{
		max_tokens:       C.int32_t(gc.MaxTokens),
		n_past:           C.int32_t(gc.NPast),
		image_max_length: C.int32_t(gc.ImageMaxLength),
	}

	if len(gc.Stop) > 0 {
		cPtr.stop, cPtr.stop_count = sliceToCCharArray(gc.Stop)
	}
	if len(gc.ImagePaths) > 0 {
		paths, count := sliceToCCharArray(gc.ImagePaths)
		cPtr.image_paths = (*C.geniex_Path)(unsafe.Pointer(paths))
		cPtr.image_count = count
	}
	if len(gc.AudioPaths) > 0 {
		paths, count := sliceToCCharArray(gc.AudioPaths)
		cPtr.audio_paths = (*C.geniex_Path)(unsafe.Pointer(paths))
		cPtr.audio_count = count
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
	freeSamplerConfig(ptr.sampler_config)
	C.free(unsafe.Pointer(ptr))
}

// LCOV_EXCL_STOP

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

// fillC writes mc into an embedded C struct; pair with freeCModelConfig to
// release the heap-allocated string fields.
// LCOV_EXCL_START
func (mc ModelConfig) fillC(out *C.geniex_ModelConfig) {
	*out = C.geniex_ModelConfig{
		n_ctx:                 C.int32_t(mc.NCtx),
		n_threads:             C.int32_t(mc.NThreads),
		n_threads_batch:       C.int32_t(mc.NThreadsBatch),
		n_batch:               C.int32_t(mc.NBatch),
		n_ubatch:              C.int32_t(mc.NUbatch),
		n_seq_max:             C.int32_t(mc.NSeqMax),
		n_gpu_layers:          C.int32_t(mc.NGpuLayers),
		chat_template_path:    cStringIfSet(mc.ChatTemplatePath),
		chat_template_content: cStringIfSet(mc.ChatTemplateContent),
	}
}

func freeCModelConfig(c *C.geniex_ModelConfig) {
	if c == nil {
		return
	}
	cFreeIfSet(unsafe.Pointer(c.chat_template_path))
	cFreeIfSet(unsafe.Pointer(c.chat_template_content))
}

// LCOV_EXCL_STOP
