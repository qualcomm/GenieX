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

extern bool go_generate_stream_on_token(char*, void*);
*/
import "C"
import (
	"log/slog"
	"unsafe"
)

// VlmCreateInput represents input parameters for creating a VLM instance
type VlmCreateInput struct {
	ModelName     string
	ModelPath     string
	MmprojPath    string
	TokenizerPath string
	Config        ModelConfig
	PluginID      string
	DeviceID      string
}

func (vci VlmCreateInput) toCPtr() *C.ml_VlmCreateInput {
	cPtr := (*C.ml_VlmCreateInput)(C.malloc(C.size_t(unsafe.Sizeof(C.ml_VlmCreateInput{}))))
	*cPtr = C.ml_VlmCreateInput{}

	if vci.ModelName != "" {
		cPtr.model_name = C.CString(vci.ModelName)
	}
	if vci.ModelPath != "" {
		cPtr.model_path = C.CString(vci.ModelPath)
	}
	if vci.MmprojPath != "" {
		cPtr.mmproj_path = C.CString(vci.MmprojPath)
	}
	if vci.TokenizerPath != "" {
		cPtr.tokenizer_path = C.CString(vci.TokenizerPath)
	}
	if vci.PluginID != "" {
		cPtr.plugin_id = C.CString(vci.PluginID)
	}
	if vci.DeviceID != "" {
		cPtr.device_id = C.CString(vci.DeviceID)
	}

	// Directly assign the ModelConfig to the C structure because it's not a pointer
	cPtr.config = C.ml_ModelConfig{
		n_ctx:           C.int32_t(vci.Config.NCtx),
		n_threads:       C.int32_t(vci.Config.NThreads),
		n_threads_batch: C.int32_t(vci.Config.NThreadsBatch),
		n_batch:         C.int32_t(vci.Config.NBatch),
		n_ubatch:        C.int32_t(vci.Config.NUbatch),
		n_seq_max:       C.int32_t(vci.Config.NSeqMax),
		n_gpu_layers:    C.int32_t(vci.Config.NGpuLayers),
	}
	if vci.Config.ChatTemplatePath != "" {
		cPtr.config.chat_template_path = C.CString(vci.Config.ChatTemplatePath)
	}
	if vci.Config.ChatTemplateContent != "" {
		cPtr.config.chat_template_content = C.CString(vci.Config.ChatTemplateContent)
	}

	return cPtr
}

func freeVlmCreateInput(cPtr *C.ml_VlmCreateInput) {
	if cPtr != nil {
		// Free string fields
		if cPtr.model_path != nil {
			C.free(unsafe.Pointer(cPtr.model_path))
		}
		if cPtr.mmproj_path != nil {
			C.free(unsafe.Pointer(cPtr.mmproj_path))
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

		// Free nested ModelConfig string fields
		if cPtr.config.chat_template_path != nil {
			C.free(unsafe.Pointer(cPtr.config.chat_template_path))
		}
		if cPtr.config.chat_template_content != nil {
			C.free(unsafe.Pointer(cPtr.config.chat_template_content))
		}

		// Free the main structure
		C.free(unsafe.Pointer(cPtr))
	}
}

type VlmContentType string

const (
	VlmContentTypeText  VlmContentType = "text"
	VlmContentTypeImage VlmContentType = "image"
	VlmContentTypeAudio VlmContentType = "audio"
)

// VlmContent represents a content item in a VLM message
type VlmContent struct {
	Type VlmContentType
	Text string
}

type vlmContents []VlmContent

func (vcs vlmContents) toCPtr() (*C.ml_VlmContent, C.int32_t) {
	if len(vcs) == 0 {
		return nil, 0
	}

	count := len(vcs)
	raw := C.malloc(C.size_t(count * C.sizeof_ml_VlmContent))
	cContents := unsafe.Slice((*C.ml_VlmContent)(raw), count)

	for i, vc := range vcs {
		// Initialize all fields to prevent garbage memory
		cContents[i]._type = nil
		cContents[i].text = nil

		if vc.Type != "" {
			cContents[i]._type = C.CString(string(vc.Type))
		}
		// Always allocate text field, even for empty strings to prevent garbage memory
		cContents[i].text = C.CString(vc.Text)
	}

	return (*C.ml_VlmContent)(raw), C.int32_t(count)
}

func freeVlmContents(cPtr *C.ml_VlmContent, count C.int64_t) {
	if cPtr == nil || count == 0 {
		return
	}

	cContents := unsafe.Slice(cPtr, int(count))
	for i := range count {
		if cContents[i]._type != nil {
			C.free(unsafe.Pointer(cContents[i]._type))
		}
		if cContents[i].text != nil {
			C.free(unsafe.Pointer(cContents[i].text))
		}
	}

	C.free(unsafe.Pointer(cPtr))
}

type VlmRole string

const (
	VlmRoleUser      VlmRole = "user"
	VlmRoleAssistant VlmRole = "assistant"
	VlmRoleSystem    VlmRole = "system"
)

// VlmChatMessage represents a chat message in VLM
type VlmChatMessage struct {
	Role     VlmRole
	Contents []VlmContent
}

type vlmChatMessages []VlmChatMessage

func (vcms vlmChatMessages) toCPtr() (*C.ml_VlmChatMessage, C.int32_t) {
	if len(vcms) == 0 {
		return nil, 0
	}

	count := len(vcms)
	raw := C.malloc(C.size_t(count * C.sizeof_ml_VlmChatMessage))
	cMessages := unsafe.Slice((*C.ml_VlmChatMessage)(raw), count)

	for i, vcm := range vcms {
		if vcm.Role != "" {
			cMessages[i].role = C.CString(string(vcm.Role))
		}
		if len(vcm.Contents) > 0 {
			contents, contentCount := vlmContents(vcm.Contents).toCPtr()
			cMessages[i].contents = contents
			cMessages[i].content_count = C.int64_t(contentCount)
		}
	}

	return (*C.ml_VlmChatMessage)(raw), C.int32_t(count)
}

func freeVlmChatMessages(cPtr *C.ml_VlmChatMessage, count C.int32_t) {
	if cPtr == nil || count == 0 {
		return
	}

	cMessages := unsafe.Slice(cPtr, int(count))
	for i := range count {
		if cMessages[i].role != nil {
			C.free(unsafe.Pointer(cMessages[i].role))
		}
		if cMessages[i].contents != nil {
			freeVlmContents(cMessages[i].contents, cMessages[i].content_count)
		}
	}

	C.free(unsafe.Pointer(cPtr))
}

// VlmApplyChatTemplateInput represents input for applying VLM chat template
type VlmApplyChatTemplateInput struct {
	Messages    []VlmChatMessage
	Tools       string
	EnableThink bool
}

func (vati VlmApplyChatTemplateInput) toCPtr() *C.ml_VlmApplyChatTemplateInput {
	cPtr := (*C.ml_VlmApplyChatTemplateInput)(C.malloc(C.size_t(unsafe.Sizeof(C.ml_VlmApplyChatTemplateInput{}))))
	*cPtr = C.ml_VlmApplyChatTemplateInput{}

	if len(vati.Messages) > 0 {
		cMessages, messageCount := vlmChatMessages(vati.Messages).toCPtr()
		cPtr.messages = cMessages
		cPtr.message_count = C.int32_t(messageCount)
	}

	if vati.Tools != "" {
		cPtr.tools = C.CString(vati.Tools)
	}

	cPtr.enable_thinking = C.bool(vati.EnableThink)

	return cPtr
}

func freeVlmApplyChatTemplateInput(cPtr *C.ml_VlmApplyChatTemplateInput) {
	if cPtr == nil {
		return
	}

	freeVlmChatMessages(cPtr.messages, cPtr.message_count)
	if cPtr.tools != nil {
		C.free(unsafe.Pointer(cPtr.tools))
	}
	C.free(unsafe.Pointer(cPtr))
}

// VlmApplyChatTemplateOutput represents output from applying VLM chat template
type VlmApplyChatTemplateOutput struct {
	FormattedText string
}

func newVlmApplyChatTemplateOutputFromCPtr(c *C.ml_VlmApplyChatTemplateOutput) VlmApplyChatTemplateOutput {
	output := VlmApplyChatTemplateOutput{}

	if c == nil {
		return output
	}

	if c.formatted_text != nil {
		output.FormattedText = C.GoString(c.formatted_text)
	}

	return output
}

func freeVlmApplyChatTemplateOutput(cPtr *C.ml_VlmApplyChatTemplateOutput) {
	if cPtr == nil {
		return
	}
	if cPtr.formatted_text != nil {
		mlFree(unsafe.Pointer(cPtr.formatted_text))
	}
}

// VlmGenerateInput represents input for VLM text generation
type VlmGenerateInput struct {
	PromptUTF8 string
	Config     *GenerationConfig
	OnToken    OnTokenCallback
}

func (vgi VlmGenerateInput) toCPtr() *C.ml_VlmGenerateInput {
	cPtr := (*C.ml_VlmGenerateInput)(C.malloc(C.size_t(unsafe.Sizeof(C.ml_VlmGenerateInput{}))))
	*cPtr = C.ml_VlmGenerateInput{}

	cPtr.prompt_utf8 = C.CString(vgi.PromptUTF8)
	if vgi.Config != nil {
		cPtr.config = vgi.Config.toCPtr()
	} else {
		cPtr.config = nil
	}

	// Note: on_token and user_data should be set by the caller
	cPtr.on_token = nil
	cPtr.user_data = nil

	return cPtr
}

func freeVlmGenerateInput(cPtr *C.ml_VlmGenerateInput) {
	if cPtr != nil {
		if cPtr.prompt_utf8 != nil {
			C.free(unsafe.Pointer(cPtr.prompt_utf8))
		}

		if cPtr.config != nil {
			freeGenerationConfig(cPtr.config)
		}

		C.free(unsafe.Pointer(cPtr))
	}
}

// VlmGenerateOutput represents output from VLM text generation
type VlmGenerateOutput struct {
	FullText    string
	ProfileData ProfileData
}

func newVlmGenerateOutputFromCPtr(c *C.ml_VlmGenerateOutput) VlmGenerateOutput {
	output := VlmGenerateOutput{}

	if c == nil {
		return output
	}

	if c.full_text != nil {
		output.FullText = C.GoString(c.full_text)
	}
	output.ProfileData = newProfileDataFromCPtr(c.profile_data)
	return output
}

func freeVlmGenerateOutput(ptr *C.ml_VlmGenerateOutput) {
	if ptr == nil {
		return
	}
	if ptr.full_text != nil {
		mlFree(unsafe.Pointer(ptr.full_text))
	}
}

// VLM represents a VLM instance
type VLM struct {
	ptr *C.ml_VLM
}

// NewVLM creates a new VLM instance
func NewVLM(input VlmCreateInput) (*VLM, error) {
	slog.Debug("NewVLM called", "input", input)

	cInput := input.toCPtr()
	defer freeVlmCreateInput(cInput)

	var cHandle *C.ml_VLM
	res := C.ml_vlm_create(cInput, &cHandle)
	if res < 0 {
		return nil, SDKError(res)
	}

	return &VLM{ptr: cHandle}, nil
}

// Destroy destroys the VLM instance and frees associated resources
func (v *VLM) Destroy() error {
	slog.Debug("Destroy called", "ptr", v.ptr)

	if v.ptr == nil {
		return nil
	}

	res := C.ml_vlm_destroy(v.ptr)
	if res < 0 {
		return SDKError(res)
	}
	v.ptr = nil
	return nil
}

// Reset resets the VLM internal state (clear KV cache, reset sampling)
func (v *VLM) Reset() error {
	slog.Debug("Reset called", "ptr", v.ptr)

	if v.ptr == nil {
		return SDKError(C.ML_ERROR_COMMON_INVALID_INPUT)
	}

	res := C.ml_vlm_reset(v.ptr)
	if res < 0 {
		return SDKError(res)
	}
	return nil
}

// ApplyChatTemplate applies chat template to messages
func (v *VLM) ApplyChatTemplate(input VlmApplyChatTemplateInput) (*VlmApplyChatTemplateOutput, error) {
	slog.Debug("ApplyChatTemplate called", "input", input)

	cinput := input.toCPtr()
	defer freeVlmApplyChatTemplateInput(cinput)

	var cOutput C.ml_VlmApplyChatTemplateOutput
	defer freeVlmApplyChatTemplateOutput(&cOutput)

	res := C.ml_vlm_apply_chat_template(v.ptr, cinput, &cOutput)
	if res < 0 {
		return nil, SDKError(res)
	}

	output := newVlmApplyChatTemplateOutputFromCPtr(&cOutput)

	return &output, nil
}

// Generate generates text with streaming token callback
func (v *VLM) Generate(input VlmGenerateInput) (*VlmGenerateOutput, error) {
	slog.Debug("Generate called", "input", input, "config", input.Config, "sampler", input.Config.SamplerConfig)

	cInput := input.toCPtr()
	defer freeVlmGenerateInput(cInput)

	// set the callback
	onToken = input.OnToken
	defer func() {
		onToken = nil // reset to default
	}()
	cInput.on_token = C.ml_token_callback(C.go_generate_stream_on_token)

	var cOutput C.ml_VlmGenerateOutput
	defer freeVlmGenerateOutput(&cOutput)

	res := C.ml_vlm_generate(v.ptr, cInput, &cOutput)
	if res < 0 {
		return nil, SDKError(res)
	}

	output := newVlmGenerateOutputFromCPtr(&cOutput)

	return &output, nil
}
