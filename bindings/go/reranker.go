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

// RerankConfig represents reranking configuration
type RerankConfig struct {
	BatchSize       int32
	Normalize       bool
	NormalizeMethod string
}

func (rc RerankConfig) toCPtr() *C.geniex_RerankConfig {
	cPtr := (*C.geniex_RerankConfig)(C.malloc(C.size_t(unsafe.Sizeof(C.geniex_RerankConfig{}))))
	*cPtr = C.geniex_RerankConfig{}

	cPtr.batch_size = C.int32_t(rc.BatchSize)
	cPtr.normalize = C.bool(rc.Normalize)
	if rc.NormalizeMethod != "" {
		cPtr.normalize_method = C.CString(rc.NormalizeMethod)
	}

	return cPtr
}

func freeRerankConfig(cPtr *C.geniex_RerankConfig) {
	if cPtr != nil {
		if cPtr.normalize_method != nil {
			C.free(unsafe.Pointer(cPtr.normalize_method))
		}
		C.free(unsafe.Pointer(cPtr))
	}
}

// RerankerCreateInput represents input parameters for reranker creation
type RerankerCreateInput struct {
	ModelName     string
	ModelPath     string
	TokenizerPath string
	PluginID      string
	DeviceID      string
}

func (rci RerankerCreateInput) toCPtr() *C.geniex_RerankerCreateInput {
	cPtr := (*C.geniex_RerankerCreateInput)(C.malloc(C.size_t(unsafe.Sizeof(C.geniex_RerankerCreateInput{}))))
	*cPtr = C.geniex_RerankerCreateInput{}

	if rci.ModelName != "" {
		cPtr.model_name = C.CString(rci.ModelName)
	}
	if rci.ModelPath != "" {
		cPtr.model_path = C.CString(rci.ModelPath)
	}
	if rci.TokenizerPath != "" {
		cPtr.tokenizer_path = C.CString(rci.TokenizerPath)
	}
	if rci.PluginID != "" {
		cPtr.plugin_id = C.CString(rci.PluginID)
	}
	if rci.DeviceID != "" {
		cPtr.device_id = C.CString(rci.DeviceID)
	}

	return cPtr
}

func freeRerankerCreateInput(cPtr *C.geniex_RerankerCreateInput) {
	if cPtr != nil {
		if cPtr.model_path != nil {
			C.free(unsafe.Pointer(cPtr.model_path))
		}
		if cPtr.tokenizer_path != nil {
			C.free(unsafe.Pointer(cPtr.tokenizer_path))
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

// RerankerRerankInput represents input parameters for reranking operation
type RerankerRerankInput struct {
	Query     string
	Documents []string
	Config    *RerankConfig
}

func (rri RerankerRerankInput) toCPtr() *C.geniex_RerankerRerankInput {
	cPtr := (*C.geniex_RerankerRerankInput)(C.malloc(C.size_t(unsafe.Sizeof(C.geniex_RerankerRerankInput{}))))
	*cPtr = C.geniex_RerankerRerankInput{}

	if rri.Query != "" {
		cPtr.query = C.CString(rri.Query)
	}

	// Convert documents array
	if len(rri.Documents) > 0 {
		documentsArray, documentCount := sliceToCCharArray(rri.Documents)
		cPtr.documents = documentsArray
		cPtr.documents_count = documentCount
	} else {
		cPtr.documents = nil
		cPtr.documents_count = 0
	}

	// Set config
	if rri.Config != nil {
		cPtr.config = rri.Config.toCPtr()
	} else {
		cPtr.config = nil
	}

	return cPtr
}

func freeRerankerRerankInput(cPtr *C.geniex_RerankerRerankInput) {
	if cPtr == nil {
		return
	}

	if cPtr.query != nil {
		C.free(unsafe.Pointer(cPtr.query))
	}

	if cPtr.documents != nil {
		freeCCharArray(cPtr.documents, cPtr.documents_count)
	}

	if cPtr.config != nil {
		freeRerankConfig(cPtr.config)
	}

	C.free(unsafe.Pointer(cPtr))
}

// RerankerRerankOutput represents output from reranking operation
type RerankerRerankOutput struct {
	Scores      []float32
	ScoreCount  int32
	ProfileData ProfileData
}

func newRerankerRerankOutputFromCPtr(c *C.geniex_RerankerRerankOutput) RerankerRerankOutput {
	output := RerankerRerankOutput{}

	if c == nil {
		return output
	}

	// Convert scores array
	if c.scores != nil && c.score_count > 0 {
		scores := unsafe.Slice((*C.float)(unsafe.Pointer(c.scores)), int(c.score_count))
		output.Scores = make([]float32, c.score_count)
		for i := range output.Scores {
			output.Scores[i] = float32(scores[i])
		}
	}

	output.ScoreCount = int32(c.score_count)
	output.ProfileData = newProfileDataFromCPtr(c.profile_data)

	return output
}

func freeRerankerRerankOutput(ptr *C.geniex_RerankerRerankOutput) {
	if ptr == nil {
		return
	}
	if ptr.scores != nil {
		mlFree(unsafe.Pointer(ptr.scores))
	}
}

// Reranker represents a reranker instance
type Reranker struct {
	ptr *C.geniex_Reranker
}

// NewReranker creates a new reranker instance
func NewReranker(input RerankerCreateInput) (*Reranker, error) {
	slog.Debug("NewReranker called", "input", input)

	cInput := input.toCPtr()
	defer freeRerankerCreateInput(cInput)

	var cHandle *C.geniex_Reranker
	res := C.geniex_reranker_create(cInput, &cHandle)
	if res < 0 {
		return nil, SDKError(res)
	}

	return &Reranker{ptr: cHandle}, nil
}

// Destroy destroys the reranker instance and frees associated resources
func (r *Reranker) Destroy() error {
	slog.Debug("Destroy called", "ptr", r.ptr)

	if r.ptr == nil {
		return nil
	}

	res := C.geniex_reranker_destroy(r.ptr)
	if res < 0 {
		return SDKError(res)
	}
	r.ptr = nil
	return nil
}

// Rerank performs reranking operation on documents against a query
func (r *Reranker) Rerank(input RerankerRerankInput) (RerankerRerankOutput, error) {
	slog.Debug("Rerank called", "input", input)

	cInput := input.toCPtr()
	defer freeRerankerRerankInput(cInput)

	var cOutput C.geniex_RerankerRerankOutput
	defer freeRerankerRerankOutput(&cOutput)

	res := C.geniex_reranker_rerank(r.ptr, cInput, &cOutput)
	if res < 0 {
		return RerankerRerankOutput{}, SDKError(res)
	}

	output := newRerankerRerankOutputFromCPtr(&cOutput)
	return output, nil
}
