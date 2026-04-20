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
	"log/slog"
	"unsafe"
)

// TTSConfig represents TTS synthesis configuration
type TTSConfig struct {
	Voice      string
	Speed      float32
	Seed       int32
	SampleRate int32
}

func (tc TTSConfig) toCPtr() *C.geniex_TTSConfig {
	cPtr := (*C.geniex_TTSConfig)(C.malloc(C.size_t(unsafe.Sizeof(C.geniex_TTSConfig{}))))
	*cPtr = C.geniex_TTSConfig{}

	if tc.Voice != "" {
		cPtr.voice = C.CString(tc.Voice)
	}
	cPtr.speed = C.float(tc.Speed)
	cPtr.seed = C.int32_t(tc.Seed)
	cPtr.sample_rate = C.int32_t(tc.SampleRate)

	return cPtr
}

func freeTTSConfig(cPtr *C.geniex_TTSConfig) {
	if cPtr != nil {
		if cPtr.voice != nil {
			C.free(unsafe.Pointer(cPtr.voice))
		}
		C.free(unsafe.Pointer(cPtr))
	}
}

// // TTSSamplerConfig represents TTS sampling parameters
// type TTSSamplerConfig struct {
// 	Temperature  float32
// 	NoiseScale   float32
// 	LengthScale  float32
// }

// func (tsc TTSSamplerConfig) toCPtr() *C.geniex_TTSSamplerConfig {
// 	cPtr := (*C.geniex_TTSSamplerConfig)(C.malloc(C.size_t(unsafe.Sizeof(C.geniex_TTSSamplerConfig{}))))
// 	*cPtr = C.geniex_TTSSamplerConfig{}

// 	cPtr.temperature = C.float(tsc.Temperature)
// 	cPtr.noise_scale = C.float(tsc.NoiseScale)
// 	cPtr.length_scale = C.float(tsc.LengthScale)

// 	return cPtr
// }

// func freeTTSSamplerConfig(cPtr *C.geniex_TTSSamplerConfig) {
// 	if cPtr != nil {
// 		C.free(unsafe.Pointer(cPtr))
// 	}
// }

// TTSResult represents TTS synthesis result
type TTSResult struct {
	AudioPath       string
	DurationSeconds float32
	SampleRate      int32
	Channels        int32
	NumSamples      int32
}

func newTTSResultFromCPtr(c *C.geniex_TTSResult) TTSResult {
	result := TTSResult{}

	if c == nil {
		return result
	}

	if c.audio_path != nil {
		result.AudioPath = C.GoString(c.audio_path)
	}
	result.DurationSeconds = float32(c.duration_seconds)
	result.SampleRate = int32(c.sample_rate)
	result.Channels = int32(c.channels)
	result.NumSamples = int32(c.num_samples)

	return result
}

func freeTTSResult(ptr *C.geniex_TTSResult) {
	if ptr == nil {
		return
	}
	if ptr.audio_path != nil {
		mlFree(unsafe.Pointer(ptr.audio_path))
	}
}

// TtsCreateInput represents input parameters for TTS creation
type TtsCreateInput struct {
	ModelName   string
	ModelPath   string
	VocoderPath string
	PluginID    string
	DeviceID    string
}

func (tci TtsCreateInput) toCPtr() *C.geniex_TtsCreateInput {
	cPtr := (*C.geniex_TtsCreateInput)(C.malloc(C.size_t(unsafe.Sizeof(C.geniex_TtsCreateInput{}))))
	*cPtr = C.geniex_TtsCreateInput{}

	if tci.ModelName != "" {
		cPtr.model_name = C.CString(tci.ModelName)
	}
	if tci.ModelPath != "" {
		cPtr.model_path = C.CString(tci.ModelPath)
	}
	if tci.VocoderPath != "" {
		cPtr.vocoder_path = C.CString(tci.VocoderPath)
	}
	if tci.PluginID != "" {
		cPtr.plugin_id = C.CString(tci.PluginID)
	}
	if tci.DeviceID != "" {
		cPtr.device_id = C.CString(tci.DeviceID)
	}

	return cPtr
}

func freeTtsCreateInput(cPtr *C.geniex_TtsCreateInput) {
	if cPtr != nil {
		if cPtr.model_path != nil {
			C.free(unsafe.Pointer(cPtr.model_path))
		}
		if cPtr.vocoder_path != nil {
			C.free(unsafe.Pointer(cPtr.vocoder_path))
		}
		if cPtr.plugin_id != nil {
			C.free(unsafe.Pointer(cPtr.plugin_id))
		}
		if cPtr.device_id != nil {
			C.free(unsafe.Pointer(cPtr.device_id))
		}
		C.free(unsafe.Pointer(cPtr))
	}
}

// TtsSynthesizeInput represents input parameters for TTS synthesis
type TtsSynthesizeInput struct {
	TextUTF8   string
	Config     *TTSConfig
	OutputPath string
}

func (tsi TtsSynthesizeInput) toCPtr() *C.geniex_TtsSynthesizeInput {
	cPtr := (*C.geniex_TtsSynthesizeInput)(C.malloc(C.size_t(unsafe.Sizeof(C.geniex_TtsSynthesizeInput{}))))
	*cPtr = C.geniex_TtsSynthesizeInput{}

	cPtr.text_utf8 = C.CString(tsi.TextUTF8)
	if tsi.Config != nil {
		cPtr.config = tsi.Config.toCPtr()
	} else {
		cPtr.config = nil
	}
	if tsi.OutputPath != "" {
		cPtr.output_path = C.CString(tsi.OutputPath)
	} else {
		cPtr.output_path = nil
	}

	return cPtr
}

func freeTtsSynthesizeInput(cPtr *C.geniex_TtsSynthesizeInput) {
	if cPtr != nil {
		if cPtr.text_utf8 != nil {
			C.free(unsafe.Pointer(cPtr.text_utf8))
		}
		if cPtr.config != nil {
			freeTTSConfig(cPtr.config)
		}
		if cPtr.output_path != nil {
			C.free(unsafe.Pointer(cPtr.output_path))
		}
		C.free(unsafe.Pointer(cPtr))
	}
}

// TtsSynthesizeOutput represents output from TTS synthesis
type TtsSynthesizeOutput struct {
	Result      TTSResult
	ProfileData ProfileData
}

func newTtsSynthesizeOutputFromCPtr(c *C.geniex_TtsSynthesizeOutput) TtsSynthesizeOutput {
	output := TtsSynthesizeOutput{}

	if c == nil {
		return output
	}

	output.Result = newTTSResultFromCPtr(&c.result)
	output.ProfileData = newProfileDataFromCPtr(c.profile_data)
	return output
}

func freeTtsSynthesizeOutput(ptr *C.geniex_TtsSynthesizeOutput) {
	if ptr == nil {
		return
	}
	freeTTSResult(&ptr.result)
}

// TtsListAvailableVoicesInput represents input for listing available voices
type TtsListAvailableVoicesInput struct {
	Reserved any
}

func (tlavi TtsListAvailableVoicesInput) toCPtr() *C.geniex_TtsListAvailableVoicesInput {
	cPtr := (*C.geniex_TtsListAvailableVoicesInput)(C.malloc(C.size_t(unsafe.Sizeof(C.geniex_TtsListAvailableVoicesInput{}))))
	*cPtr = C.geniex_TtsListAvailableVoicesInput{}
	return cPtr
}

func freeTtsListAvailableVoicesInput(cPtr *C.geniex_TtsListAvailableVoicesInput) {
	if cPtr != nil {
		C.free(unsafe.Pointer(cPtr))
	}
}

// TtsListAvailableVoicesOutput represents output for listing available voices
type TtsListAvailableVoicesOutput struct {
	VoiceIDs []string
}

func newTtsListAvailableVoicesOutputFromCPtr(c *C.geniex_TtsListAvailableVoicesOutput) TtsListAvailableVoicesOutput {
	output := TtsListAvailableVoicesOutput{}

	if c == nil {
		return output
	}

	output.VoiceIDs = cCharArrayToSlice(c.voice_ids, c.voice_count)

	return output
}

func freeTtsListAvailableVoicesOutput(ptr *C.geniex_TtsListAvailableVoicesOutput) {
	if ptr == nil {
		return
	}
	if ptr.voice_ids != nil {
		mlFreeCCharArray(ptr.voice_ids, ptr.voice_count)
	}
}

// TTS represents a TTS instance
type TTS struct {
	ptr *C.geniex_TTS
}

// NewTTS creates a new TTS instance
func NewTTS(input TtsCreateInput) (*TTS, error) {
	slog.Debug("NewTTS called", "input", input)

	cInput := input.toCPtr()
	defer freeTtsCreateInput(cInput)

	var cHandle *C.geniex_TTS
	res := C.geniex_tts_create(cInput, &cHandle)
	if res < 0 {
		return nil, SDKError(res)
	}

	return &TTS{ptr: cHandle}, nil
}

// Destroy destroys the TTS instance and frees associated resources
func (t *TTS) Destroy() error {
	slog.Debug("Destroy called", "ptr", t.ptr)

	if t.ptr == nil {
		return nil
	}

	res := C.geniex_tts_destroy(t.ptr)
	if res < 0 {
		return SDKError(res)
	}
	t.ptr = nil
	return nil
}

// Synthesize synthesizes speech from text and saves to filesystem
func (t *TTS) Synthesize(input TtsSynthesizeInput) (TtsSynthesizeOutput, error) {
	slog.Debug("Synthesize called", "input", input)

	cInput := input.toCPtr()
	defer freeTtsSynthesizeInput(cInput)

	var cOutput C.geniex_TtsSynthesizeOutput
	defer freeTtsSynthesizeOutput(&cOutput)

	res := C.geniex_tts_synthesize(t.ptr, cInput, &cOutput)
	if res < 0 {
		return TtsSynthesizeOutput{}, SDKError(res)
	}

	output := newTtsSynthesizeOutputFromCPtr(&cOutput)
	return output, nil
}

// ListAvailableVoices gets list of available voice identifiers
func (t *TTS) ListAvailableVoices() (TtsListAvailableVoicesOutput, error) {
	slog.Debug("ListAvailableVoices called")

	input := TtsListAvailableVoicesInput{}
	cInput := input.toCPtr()
	defer freeTtsListAvailableVoicesInput(cInput)

	var cOutput C.geniex_TtsListAvailableVoicesOutput
	defer freeTtsListAvailableVoicesOutput(&cOutput)

	res := C.geniex_tts_list_available_voices(t.ptr, cInput, &cOutput)
	if res < 0 {
		return TtsListAvailableVoicesOutput{}, SDKError(res)
	}

	output := newTtsListAvailableVoicesOutputFromCPtr(&cOutput)
	return output, nil
}
