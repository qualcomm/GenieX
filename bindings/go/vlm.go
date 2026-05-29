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

extern bool go_generate_stream_on_token(char*, void*);
*/
import "C"

import (
	"log/slog"
	"runtime/cgo"
	"unsafe"
)

// LCOV_EXCL_START

type VlmCreateInput struct {
	ModelName     string
	ModelPath     string
	MmprojPath    string
	TokenizerPath string
	Config        ModelConfig
	PluginID      string
	DeviceID      string
}

func (vci VlmCreateInput) toCPtr() *C.geniex_VlmCreateInput {
	cPtr := (*C.geniex_VlmCreateInput)(cMalloc(C.sizeof_geniex_VlmCreateInput))
	*cPtr = C.geniex_VlmCreateInput{
		model_name:     cStringIfSet(vci.ModelName),
		model_path:     cStringIfSet(vci.ModelPath),
		mmproj_path:    cStringIfSet(vci.MmprojPath),
		tokenizer_path: cStringIfSet(vci.TokenizerPath),
		plugin_id:      cStringIfSet(vci.PluginID),
		device_id:      cStringIfSet(vci.DeviceID),
	}
	vci.Config.fillC(&cPtr.config)
	return cPtr
}

func freeVlmCreateInput(cPtr *C.geniex_VlmCreateInput) {
	if cPtr == nil {
		return
	}
	cFreeIfSet(unsafe.Pointer(cPtr.model_name))
	cFreeIfSet(unsafe.Pointer(cPtr.model_path))
	cFreeIfSet(unsafe.Pointer(cPtr.mmproj_path))
	cFreeIfSet(unsafe.Pointer(cPtr.tokenizer_path))
	cFreeIfSet(unsafe.Pointer(cPtr.plugin_id))
	cFreeIfSet(unsafe.Pointer(cPtr.device_id))
	freeCModelConfig(&cPtr.config)
	C.free(unsafe.Pointer(cPtr))
}

type VlmContentType string

const (
	VlmContentTypeText  VlmContentType = "text"
	VlmContentTypeImage VlmContentType = "image"
	VlmContentTypeAudio VlmContentType = "audio"
)

type VlmContent struct {
	Type VlmContentType
	Text string
}

type vlmContents []VlmContent

func (vcs vlmContents) toCPtr() (*C.geniex_VlmContent, C.int64_t) {
	if len(vcs) == 0 {
		return nil, 0
	}
	count := len(vcs)
	raw := cMalloc(C.size_t(count) * C.sizeof_geniex_VlmContent)
	cContents := unsafe.Slice((*C.geniex_VlmContent)(raw), count)
	for i, vc := range vcs {
		cContents[i] = C.geniex_VlmContent{
			_type: cStringIfSet(string(vc.Type)),
			text:  cStringIfSet(vc.Text),
		}
	}
	return (*C.geniex_VlmContent)(raw), C.int64_t(count)
}

func freeVlmContents(cPtr *C.geniex_VlmContent, count C.int64_t) {
	if cPtr == nil || count == 0 {
		return
	}
	cContents := unsafe.Slice(cPtr, int(count))
	for i := range cContents {
		cFreeIfSet(unsafe.Pointer(cContents[i]._type))
		cFreeIfSet(unsafe.Pointer(cContents[i].text))
	}
	C.free(unsafe.Pointer(cPtr))
}

type VlmRole string

const (
	VlmRoleUser      VlmRole = "user"
	VlmRoleAssistant VlmRole = "assistant"
	VlmRoleSystem    VlmRole = "system"
)

type VlmChatMessage struct {
	Role     VlmRole
	Contents []VlmContent
}

type vlmChatMessages []VlmChatMessage

func (vcms vlmChatMessages) toCPtr() (*C.geniex_VlmChatMessage, C.int32_t) {
	if len(vcms) == 0 {
		return nil, 0
	}
	count := len(vcms)
	raw := cMalloc(C.size_t(count) * C.sizeof_geniex_VlmChatMessage)
	cMessages := unsafe.Slice((*C.geniex_VlmChatMessage)(raw), count)
	for i, vcm := range vcms {
		contents, contentCount := vlmContents(vcm.Contents).toCPtr()
		cMessages[i] = C.geniex_VlmChatMessage{
			role:          cStringIfSet(string(vcm.Role)),
			contents:      contents,
			content_count: contentCount,
		}
	}
	return (*C.geniex_VlmChatMessage)(raw), C.int32_t(count)
}

func freeVlmChatMessages(cPtr *C.geniex_VlmChatMessage, count C.int32_t) {
	if cPtr == nil || count == 0 {
		return
	}
	cMessages := unsafe.Slice(cPtr, int(count))
	for i := range cMessages {
		cFreeIfSet(unsafe.Pointer(cMessages[i].role))
		freeVlmContents(cMessages[i].contents, cMessages[i].content_count)
	}
	C.free(unsafe.Pointer(cPtr))
}

type VlmApplyChatTemplateInput struct {
	Messages    []VlmChatMessage
	Tools       string
	EnableThink bool
}

func (vati VlmApplyChatTemplateInput) toCPtr() *C.geniex_VlmApplyChatTemplateInput {
	cPtr := (*C.geniex_VlmApplyChatTemplateInput)(cMalloc(C.sizeof_geniex_VlmApplyChatTemplateInput))
	*cPtr = C.geniex_VlmApplyChatTemplateInput{
		tools:           cStringIfSet(vati.Tools),
		enable_thinking: C.bool(vati.EnableThink),
	}
	cPtr.messages, cPtr.message_count = vlmChatMessages(vati.Messages).toCPtr()
	return cPtr
}

func freeVlmApplyChatTemplateInput(cPtr *C.geniex_VlmApplyChatTemplateInput) {
	if cPtr == nil {
		return
	}
	freeVlmChatMessages(cPtr.messages, cPtr.message_count)
	cFreeIfSet(unsafe.Pointer(cPtr.tools))
	C.free(unsafe.Pointer(cPtr))
}

type VlmApplyChatTemplateOutput struct {
	FormattedText string
}

func newVlmApplyChatTemplateOutputFromCPtr(c *C.geniex_VlmApplyChatTemplateOutput) VlmApplyChatTemplateOutput {
	if c == nil {
		return VlmApplyChatTemplateOutput{}
	}
	return VlmApplyChatTemplateOutput{FormattedText: C.GoString(c.formatted_text)}
}

func freeVlmApplyChatTemplateOutput(cPtr *C.geniex_VlmApplyChatTemplateOutput) {
	if cPtr == nil {
		return
	}
	free(unsafe.Pointer(cPtr.formatted_text))
}

type VlmGenerateInput struct {
	PromptUTF8 string
	Config     *GenerationConfig
	OnToken    OnTokenCallback
}

func (vgi VlmGenerateInput) toCPtr() *C.geniex_VlmGenerateInput {
	cPtr := (*C.geniex_VlmGenerateInput)(cMalloc(C.sizeof_geniex_VlmGenerateInput))
	*cPtr = C.geniex_VlmGenerateInput{
		prompt_utf8: cStringIfSet(vgi.PromptUTF8),
	}
	if vgi.Config != nil {
		cPtr.config = vgi.Config.toCPtr()
	}
	return cPtr
}

func freeVlmGenerateInput(cPtr *C.geniex_VlmGenerateInput) {
	if cPtr == nil {
		return
	}
	cFreeIfSet(unsafe.Pointer(cPtr.prompt_utf8))
	freeGenerationConfig(cPtr.config)
	C.free(unsafe.Pointer(cPtr))
}

type VlmGenerateOutput struct {
	FullText    string
	ProfileData ProfileData
}

func newVlmGenerateOutputFromCPtr(c *C.geniex_VlmGenerateOutput) VlmGenerateOutput {
	if c == nil {
		return VlmGenerateOutput{}
	}
	return VlmGenerateOutput{
		FullText:    C.GoString(c.full_text),
		ProfileData: newProfileDataFromCPtr(c.profile_data),
	}
}

func freeVlmGenerateOutput(ptr *C.geniex_VlmGenerateOutput) {
	if ptr == nil {
		return
	}
	free(unsafe.Pointer(ptr.full_text))
}

type VLM struct {
	ptr *C.geniex_VLM
}

func NewVLM(input VlmCreateInput) (*VLM, error) {
	slog.Debug("NewVLM called", "input", input)

	cInput := input.toCPtr()
	defer freeVlmCreateInput(cInput)

	var cHandle *C.geniex_VLM
	res := C.geniex_vlm_create(cInput, &cHandle)
	if res < 0 {
		return nil, SDKError(res)
	}
	return &VLM{ptr: cHandle}, nil
}

func (v *VLM) Destroy() error {
	slog.Debug("Destroy called", "ptr", v.ptr)
	if v.ptr == nil {
		return nil
	}
	res := C.geniex_vlm_destroy(v.ptr)
	if res < 0 {
		return SDKError(res)
	}
	v.ptr = nil
	return nil
}

type VlmCapabilities struct {
	SupportsVision bool
	SupportsAudio  bool
}

// Capabilities reports the mmproj's supported modalities. Plugins without a
// modality probe (e.g. QAIRT) return both flags false.
func (v *VLM) Capabilities() (VlmCapabilities, error) {
	if v.ptr == nil {
		return VlmCapabilities{}, SDKError(C.GENIEX_ERROR_COMMON_INVALID_INPUT)
	}
	var cOut C.geniex_VlmCapabilities
	res := C.geniex_vlm_get_capabilities(v.ptr, &cOut)
	if res < 0 {
		return VlmCapabilities{}, SDKError(res)
	}
	return VlmCapabilities{
		SupportsVision: bool(cOut.supports_vision),
		SupportsAudio:  bool(cOut.supports_audio),
	}, nil
}

func (v *VLM) Reset() error {
	slog.Debug("Reset called", "ptr", v.ptr)
	if v.ptr == nil {
		return SDKError(C.GENIEX_ERROR_COMMON_INVALID_INPUT)
	}
	res := C.geniex_vlm_reset(v.ptr)
	if res < 0 {
		return SDKError(res)
	}
	return nil
}

func (v *VLM) ApplyChatTemplate(input VlmApplyChatTemplateInput) (*VlmApplyChatTemplateOutput, error) {
	slog.Debug("ApplyChatTemplate called", "input", input)

	cInput := input.toCPtr()
	defer freeVlmApplyChatTemplateInput(cInput)

	var cOutput C.geniex_VlmApplyChatTemplateOutput
	defer freeVlmApplyChatTemplateOutput(&cOutput)

	res := C.geniex_vlm_apply_chat_template(v.ptr, cInput, &cOutput)
	if res < 0 {
		return nil, SDKError(res)
	}
	output := newVlmApplyChatTemplateOutputFromCPtr(&cOutput)
	return &output, nil
}

func (v *VLM) Generate(input VlmGenerateInput) (*VlmGenerateOutput, error) {
	slog.Debug("Generate called", "promptLen", len(input.PromptUTF8))

	cInput := input.toCPtr()
	defer freeVlmGenerateInput(cInput)

	if input.OnToken != nil {
		h := cgo.NewHandle(input.OnToken)
		defer h.Delete()
		cInput.on_token = C.geniex_token_callback(C.go_generate_stream_on_token)
		cInput.user_data = handleToUserData(h)
	}

	var cOutput C.geniex_VlmGenerateOutput
	defer freeVlmGenerateOutput(&cOutput)

	res := C.geniex_vlm_generate(v.ptr, cInput, &cOutput)
	// On context-length errors the SDK still populates whatever was generated
	// before the cutoff; surface that to the caller alongside the error.
	if res < 0 && res != C.GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH {
		return nil, SDKError(res)
	}
	output := newVlmGenerateOutputFromCPtr(&cOutput)
	if res < 0 {
		return &output, SDKError(res)
	}
	return &output, nil
}

// LCOV_EXCL_STOP
