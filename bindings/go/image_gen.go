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

package geniex_bridge

/*
#include <stdlib.h>
#include "ml.h"
*/
import "C"
import (
	"log/slog"
	"unsafe"
)

// ImageGenCreateInput represents input parameters for creating an ImageGen instance
type ImageGenCreateInput struct {
	ModelName           string
	ModelPath           string
	SchedulerConfigPath string
	PluginID            string
	DeviceID            string
}

func (igci ImageGenCreateInput) toCPtr() *C.ml_ImageGenCreateInput {
	cPtr := (*C.ml_ImageGenCreateInput)(C.malloc(C.size_t(unsafe.Sizeof(C.ml_ImageGenCreateInput{}))))
	*cPtr = C.ml_ImageGenCreateInput{}

	if igci.ModelName != "" {
		cPtr.model_name = C.CString(igci.ModelName)
	}
	if igci.ModelPath != "" {
		cPtr.model_path = C.CString(igci.ModelPath)
	}
	if igci.SchedulerConfigPath != "" {
		cPtr.scheduler_config_path = C.CString(igci.SchedulerConfigPath)
	}
	if igci.PluginID != "" {
		cPtr.plugin_id = C.CString(igci.PluginID)
	}
	if igci.DeviceID != "" {
		cPtr.device_id = C.CString(igci.DeviceID)
	}

	return cPtr
}

func freeImageGenCreateInput(cPtr *C.ml_ImageGenCreateInput) {
	if cPtr != nil {
		if cPtr.model_path != nil {
			C.free(unsafe.Pointer(cPtr.model_path))
		}
		if cPtr.scheduler_config_path != nil {
			C.free(unsafe.Pointer(cPtr.scheduler_config_path))
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

// ImageSamplerConfig represents image sampling configuration
type ImageSamplerConfig struct {
	Method        string
	Steps         int32
	GuidanceScale float32
	Eta           float32
	Seed          int32
}

// SchedulerConfig represents scheduler configuration
type SchedulerConfig struct {
	Type              string
	NumTrainTimesteps int32
	StepsOffset       int32
	BetaStart         float32
	BetaEnd           float32
	BetaSchedule      string
	PredictionType    string
	TimestepType      string
	TimestepSpacing   string
	InterpolationType string
	ConfigPath        string
}

// ImageGenerationConfig represents image generation configuration
type ImageGenerationConfig struct {
	Prompts         []string
	NegativePrompts []string
	Height          int32
	Width           int32
	SamplerConfig   ImageSamplerConfig
	SchedulerConfig SchedulerConfig
	Strength        float32
}

func (igc ImageGenerationConfig) toCPtr() *C.ml_ImageGenerationConfig {
	cPtr := (*C.ml_ImageGenerationConfig)(C.malloc(C.size_t(unsafe.Sizeof(C.ml_ImageGenerationConfig{}))))
	*cPtr = C.ml_ImageGenerationConfig{}

	if len(igc.Prompts) > 0 {
		cPtr.prompts, cPtr.prompt_count = sliceToCCharArray(igc.Prompts)
	}
	if len(igc.NegativePrompts) > 0 {
		cPtr.negative_prompts, cPtr.negative_prompt_count = sliceToCCharArray(igc.NegativePrompts)
	}
	cPtr.height = C.int32_t(igc.Height)
	cPtr.width = C.int32_t(igc.Width)

	// Convert sampler config to C struct (not pointer)
	if igc.SamplerConfig.Method != "" {
		cPtr.sampler_config.method = C.CString(igc.SamplerConfig.Method)
	}
	cPtr.sampler_config.steps = C.int32_t(igc.SamplerConfig.Steps)
	cPtr.sampler_config.guidance_scale = C.float(igc.SamplerConfig.GuidanceScale)
	cPtr.sampler_config.eta = C.float(igc.SamplerConfig.Eta)
	cPtr.sampler_config.seed = C.int32_t(igc.SamplerConfig.Seed)

	// Convert scheduler config to C struct (not pointer)
	if igc.SchedulerConfig.Type != "" {
		cPtr.scheduler_config._type = C.CString(igc.SchedulerConfig.Type)
	}
	cPtr.scheduler_config.num_train_timesteps = C.int32_t(igc.SchedulerConfig.NumTrainTimesteps)
	cPtr.scheduler_config.steps_offset = C.int32_t(igc.SchedulerConfig.StepsOffset)
	cPtr.scheduler_config.beta_start = C.float(igc.SchedulerConfig.BetaStart)
	cPtr.scheduler_config.beta_end = C.float(igc.SchedulerConfig.BetaEnd)
	if igc.SchedulerConfig.BetaSchedule != "" {
		cPtr.scheduler_config.beta_schedule = C.CString(igc.SchedulerConfig.BetaSchedule)
	}
	if igc.SchedulerConfig.PredictionType != "" {
		cPtr.scheduler_config.prediction_type = C.CString(igc.SchedulerConfig.PredictionType)
	}
	if igc.SchedulerConfig.TimestepType != "" {
		cPtr.scheduler_config.timestep_type = C.CString(igc.SchedulerConfig.TimestepType)
	}
	if igc.SchedulerConfig.TimestepSpacing != "" {
		cPtr.scheduler_config.timestep_spacing = C.CString(igc.SchedulerConfig.TimestepSpacing)
	}
	if igc.SchedulerConfig.InterpolationType != "" {
		cPtr.scheduler_config.interpolation_type = C.CString(igc.SchedulerConfig.InterpolationType)
	}
	if igc.SchedulerConfig.ConfigPath != "" {
		cPtr.scheduler_config.config_path = C.CString(igc.SchedulerConfig.ConfigPath)
	}

	cPtr.strength = C.float(igc.Strength)

	return cPtr
}

func freeImageGenerationConfig(cPtr *C.ml_ImageGenerationConfig) {
	if cPtr == nil {
		return
	}

	freeCCharArray(cPtr.prompts, cPtr.prompt_count)
	freeCCharArray(cPtr.negative_prompts, cPtr.negative_prompt_count)

	// Free sampler config strings
	if cPtr.sampler_config.method != nil {
		C.free(unsafe.Pointer(cPtr.sampler_config.method))
	}

	// Free scheduler config strings
	if cPtr.scheduler_config._type != nil {
		C.free(unsafe.Pointer(cPtr.scheduler_config._type))
	}
	if cPtr.scheduler_config.beta_schedule != nil {
		C.free(unsafe.Pointer(cPtr.scheduler_config.beta_schedule))
	}
	if cPtr.scheduler_config.prediction_type != nil {
		C.free(unsafe.Pointer(cPtr.scheduler_config.prediction_type))
	}
	if cPtr.scheduler_config.timestep_type != nil {
		C.free(unsafe.Pointer(cPtr.scheduler_config.timestep_type))
	}
	if cPtr.scheduler_config.timestep_spacing != nil {
		C.free(unsafe.Pointer(cPtr.scheduler_config.timestep_spacing))
	}
	if cPtr.scheduler_config.interpolation_type != nil {
		C.free(unsafe.Pointer(cPtr.scheduler_config.interpolation_type))
	}
	if cPtr.scheduler_config.config_path != nil {
		C.free(unsafe.Pointer(cPtr.scheduler_config.config_path))
	}

	C.free(unsafe.Pointer(cPtr))
}

// ImageGenTxt2ImgInput represents input parameters for text-to-image generation
type ImageGenTxt2ImgInput struct {
	PromptUTF8 string
	Config     *ImageGenerationConfig
	OutputPath string
}

func (igtii ImageGenTxt2ImgInput) toCPtr() *C.ml_ImageGenTxt2ImgInput {
	cPtr := (*C.ml_ImageGenTxt2ImgInput)(C.malloc(C.size_t(unsafe.Sizeof(C.ml_ImageGenTxt2ImgInput{}))))
	*cPtr = C.ml_ImageGenTxt2ImgInput{}

	cPtr.prompt_utf8 = C.CString(igtii.PromptUTF8)
	if igtii.Config != nil {
		cPtr.config = igtii.Config.toCPtr()
	} else {
		cPtr.config = nil
	}
	if igtii.OutputPath != "" {
		cPtr.output_path = C.CString(igtii.OutputPath)
	} else {
		cPtr.output_path = nil
	}

	return cPtr
}

func freeImageGenTxt2ImgInput(cPtr *C.ml_ImageGenTxt2ImgInput) {
	if cPtr == nil {
		return
	}
	if cPtr.prompt_utf8 != nil {
		C.free(unsafe.Pointer(cPtr.prompt_utf8))
	}
	if cPtr.config != nil {
		freeImageGenerationConfig(cPtr.config)
	}
	if cPtr.output_path != nil {
		C.free(unsafe.Pointer(cPtr.output_path))
	}
	C.free(unsafe.Pointer(cPtr))
}

// ImageGenOutput represents output from image generation
type ImageGenOutput struct {
	OutputImagePath string
}

func newImageGenOutputFromCPtr(c *C.ml_ImageGenOutput) ImageGenOutput {
	output := ImageGenOutput{}

	if c == nil {
		return output
	}

	if c.output_image_path != nil {
		output.OutputImagePath = C.GoString(c.output_image_path)
	}

	return output
}

func freeImageGenOutput(ptr *C.ml_ImageGenOutput) {
	if ptr == nil {
		return
	}
	if ptr.output_image_path != nil {
		mlFree(unsafe.Pointer(ptr.output_image_path))
	}
}

// ImageGen represents an ImageGen instance
type ImageGen struct {
	ptr *C.ml_ImageGen
}

// NewImageGen creates a new ImageGen instance
func NewImageGen(input ImageGenCreateInput) (*ImageGen, error) {
	slog.Debug("NewImageGen called", "input", input)

	cInput := input.toCPtr()
	defer freeImageGenCreateInput(cInput)

	var cHandle *C.ml_ImageGen
	res := C.ml_imagegen_create(cInput, &cHandle)
	if res < 0 {
		return nil, SDKError(res)
	}

	return &ImageGen{ptr: cHandle}, nil
}

// Destroy destroys the ImageGen instance and frees associated resources
func (ig *ImageGen) Destroy() error {
	slog.Debug("Destroy called", "ptr", ig.ptr)

	if ig.ptr == nil {
		return nil
	}

	res := C.ml_imagegen_destroy(ig.ptr)
	if res < 0 {
		return SDKError(res)
	}
	ig.ptr = nil
	return nil
}

// Txt2Img generates an image from text prompt
func (ig *ImageGen) Txt2Img(input ImageGenTxt2ImgInput) (ImageGenOutput, error) {
	slog.Debug("Txt2Img called", "input", input)

	cInput := input.toCPtr()
	defer freeImageGenTxt2ImgInput(cInput)

	var cOutput C.ml_ImageGenOutput
	defer freeImageGenOutput(&cOutput)

	res := C.ml_imagegen_txt2img(ig.ptr, cInput, &cOutput)
	if res < 0 {
		return ImageGenOutput{}, SDKError(res)
	}

	output := newImageGenOutputFromCPtr(&cOutput)
	return output, nil
}
