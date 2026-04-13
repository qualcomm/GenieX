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
	"fmt"
	"log/slog"
	"unsafe"
)

// CVCapabilities represents CV model capabilities
type CVCapabilities int32

const (
	CVCapabilityOCR            CVCapabilities = C.ML_CV_OCR
	CVCapabilityClassification CVCapabilities = C.ML_CV_CLASSIFICATION
	CVCapabilitySegmentation   CVCapabilities = C.ML_CV_SEGMENTATION
	CVCapabilityCustom         CVCapabilities = C.ML_CV_CUSTOM
)

// BoundingBox represents a generic bounding box structure
type BoundingBox struct {
	X      float32
	Y      float32
	Width  float32
	Height float32
}

func newBoundingBoxFromCPtr(c *C.ml_BoundingBox) BoundingBox {
	bbox := BoundingBox{}

	if c == nil {
		return bbox
	}

	bbox.X = float32(c.x)
	bbox.Y = float32(c.y)
	bbox.Width = float32(c.width)
	bbox.Height = float32(c.height)

	return bbox
}

// CVResult represents a generic detection/classification result
type CVResult struct {
	ImagePaths   []string
	ClassID      int32
	Confidence   float32
	BBox         BoundingBox
	Text         string
	Embedding    []float32
	EmbeddingDim int32
	Mask         []float32
	MaskWidth    int32
	MaskHeight   int32
}

func (c CVResult) String() string {
	return fmt.Sprintf("CVResult{ImagePaths: %v, ClassID: %d, Confidence: %.4f, BBox: %v, Text: %s, EmbeddingDim: %d, MaskWidth: %d, MaskHeight: %d}",
		c.ImagePaths, c.ClassID, c.Confidence, c.BBox, c.Text, c.EmbeddingDim, c.MaskWidth, c.MaskHeight)
}

func newCVResultFromCPtr(c *C.ml_CVResult) CVResult {
	result := CVResult{}

	if c == nil {
		return result
	}

	// Convert image paths
	if c.image_paths != nil && c.image_count > 0 {
		imagePaths := unsafe.Slice(c.image_paths, int(c.image_count))
		result.ImagePaths = make([]string, c.image_count)
		for i := range result.ImagePaths {
			if imagePaths[i] != nil {
				result.ImagePaths[i] = C.GoString(imagePaths[i])
			}
		}
	}

	result.ClassID = int32(c.class_id)
	result.Confidence = float32(c.confidence)
	result.BBox = newBoundingBoxFromCPtr(&c.bbox)

	// Convert text
	if c.text != nil {
		result.Text = C.GoString(c.text)
	}

	// Convert embedding
	if c.embedding != nil && c.embedding_dim > 0 {
		embedding := unsafe.Slice((*C.float)(unsafe.Pointer(c.embedding)), int(c.embedding_dim))
		result.Embedding = make([]float32, c.embedding_dim)
		for i := range result.Embedding {
			result.Embedding[i] = float32(embedding[i])
		}
		result.EmbeddingDim = int32(c.embedding_dim)
	}

	// Convert mask
	if c.mask != nil && c.mask_w > 0 && c.mask_h > 0 {
		maskSize := int(c.mask_w * c.mask_h)
		mask := unsafe.Slice((*C.float)(unsafe.Pointer(c.mask)), maskSize)
		result.Mask = make([]float32, maskSize)
		for i := range result.Mask {
			result.Mask[i] = float32(mask[i])
		}
		result.MaskWidth = int32(c.mask_w)
		result.MaskHeight = int32(c.mask_h)
	}

	return result
}

func freeCVResult(ptr *C.ml_CVResult) {
	if ptr == nil {
		return
	}

	// Free image paths
	if ptr.image_paths != nil {
		imagePaths := unsafe.Slice(ptr.image_paths, int(ptr.image_count))
		for i := range imagePaths {
			if imagePaths[i] != nil {
				mlFree(unsafe.Pointer(imagePaths[i]))
			}
		}
		mlFree(unsafe.Pointer(ptr.image_paths))
	}

	// Free text
	if ptr.text != nil {
		mlFree(unsafe.Pointer(ptr.text))
	}

	// Free embedding
	if ptr.embedding != nil {
		mlFree(unsafe.Pointer(ptr.embedding))
	}

	// Free mask
	if ptr.mask != nil {
		mlFree(unsafe.Pointer(ptr.mask))
	}
}

// CVModelConfig represents CV model preprocessing configuration
type CVModelConfig struct {
	Capabilities CVCapabilities
	DetModelPath string // Detection model path
	RecModelPath string // Recognition model path
	CharDictPath string
}

func (cmc CVModelConfig) toCPtr() *C.ml_CVModelConfig {
	cPtr := (*C.ml_CVModelConfig)(C.malloc(C.size_t(unsafe.Sizeof(C.ml_CVModelConfig{}))))
	*cPtr = C.ml_CVModelConfig{}

	cPtr.capabilities = C.ml_CVCapabilities(cmc.Capabilities)

	if cmc.DetModelPath != "" {
		cPtr.det_model_path = C.CString(cmc.DetModelPath)
	}
	if cmc.RecModelPath != "" {
		cPtr.rec_model_path = C.CString(cmc.RecModelPath)
	}
	if cmc.CharDictPath != "" {
		cPtr.char_dict_path = C.CString(cmc.CharDictPath)
	}

	return cPtr
}

// CVCreateInput represents input parameters for CV model creation
type CVCreateInput struct {
	ModelName string
	Config    CVModelConfig
	PluginID  string
	DeviceID  string
}

func (cvi CVCreateInput) toCPtr() *C.ml_CVCreateInput {
	cPtr := (*C.ml_CVCreateInput)(C.malloc(C.size_t(unsafe.Sizeof(C.ml_CVCreateInput{}))))
	*cPtr = C.ml_CVCreateInput{}

	// Convert config - config is a value type, not a pointer
	configPtr := cvi.Config.toCPtr()
	cPtr.config = *configPtr
	C.free(unsafe.Pointer(configPtr))

	if cvi.ModelName != "" {
		cPtr.model_name = C.CString(cvi.ModelName)
	}
	if cvi.PluginID != "" {
		cPtr.plugin_id = C.CString(cvi.PluginID)
	}
	if cvi.DeviceID != "" {
		cPtr.device_id = C.CString(cvi.DeviceID)
	}

	return cPtr
}

func freeCVCreateInput(cPtr *C.ml_CVCreateInput) {
	if cPtr != nil {
		// Note: config is a value type, so we don't need to free it separately
		if cPtr.plugin_id != nil {
			C.free(unsafe.Pointer(cPtr.plugin_id))
		}
		if cPtr.device_id != nil {
			C.free(unsafe.Pointer(cPtr.device_id))
		}
		C.free(unsafe.Pointer(cPtr))
	}
}

// CVInferInput represents input parameters for CV inference
type CVInferInput struct {
	InputImagePath string
}

func (cii CVInferInput) toCPtr() *C.ml_CVInferInput {
	cPtr := (*C.ml_CVInferInput)(C.malloc(C.size_t(unsafe.Sizeof(C.ml_CVInferInput{}))))
	*cPtr = C.ml_CVInferInput{}

	if cii.InputImagePath != "" {
		cPtr.input_image_path = C.CString(cii.InputImagePath)
	}

	return cPtr
}

func freeCVInferInput(cPtr *C.ml_CVInferInput) {
	if cPtr != nil {
		if cPtr.input_image_path != nil {
			C.free(unsafe.Pointer(cPtr.input_image_path))
		}
		C.free(unsafe.Pointer(cPtr))
	}
}

// CVInferOutput represents output from CV inference
type CVInferOutput struct {
	Results []CVResult
}

func newCVInferOutputFromCPtr(c *C.ml_CVInferOutput) CVInferOutput {
	output := CVInferOutput{}

	if c == nil {
		return output
	}

	// Convert results array
	if c.results != nil && c.result_count > 0 {
		results := unsafe.Slice(c.results, int(c.result_count))
		output.Results = make([]CVResult, c.result_count)
		for i := range output.Results {
			output.Results[i] = newCVResultFromCPtr(&results[i])
		}
	}

	return output
}

func freeCVInferOutput(ptr *C.ml_CVInferOutput) {
	if ptr == nil {
		return
	}

	// Free individual results
	if ptr.results != nil && ptr.result_count > 0 {
		results := unsafe.Slice(ptr.results, int(ptr.result_count))
		for i := range results {
			freeCVResult(&results[i])
		}
		mlFree(unsafe.Pointer(ptr.results))
	}
}

// CV represents a CV model instance
type CV struct {
	ptr *C.ml_CV
}

// NewCV creates a new CV model instance
func NewCV(input CVCreateInput) (*CV, error) {
	slog.Debug("NewCV called", "input", input)

	cInput := input.toCPtr()
	defer freeCVCreateInput(cInput)

	var cHandle *C.ml_CV
	res := C.ml_cv_create(cInput, &cHandle)
	if res < 0 {
		return nil, SDKError(res)
	}

	return &CV{ptr: cHandle}, nil
}

// Destroy destroys the CV model instance and frees associated resources
func (c *CV) Destroy() error {
	slog.Debug("Destroy called", "ptr", c.ptr)

	if c.ptr == nil {
		return nil
	}

	res := C.ml_cv_destroy(c.ptr)
	if res < 0 {
		return SDKError(res)
	}
	c.ptr = nil
	return nil
}

// Infer performs inference on a single image
func (c *CV) Infer(input CVInferInput) (CVInferOutput, error) {
	slog.Debug("Infer called", "input", input)

	cInput := input.toCPtr()
	defer freeCVInferInput(cInput)

	var cOutput C.ml_CVInferOutput
	defer freeCVInferOutput(&cOutput)

	res := C.ml_cv_infer(c.ptr, cInput, &cOutput)
	if res < 0 {
		return CVInferOutput{}, SDKError(res)
	}

	output := newCVInferOutputFromCPtr(&cOutput)
	return output, nil
}
