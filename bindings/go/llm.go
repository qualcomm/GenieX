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

type LlmRole string

const (
	LlmRoleSystem    LlmRole = "system"
	LlmRoleUser      LlmRole = "user"
	LlmRoleAssistant LlmRole = "assistant"
)

type LlmCreateInput struct {
	ModelName     string
	ModelPath     string
	TokenizerPath string
	Config        ModelConfig
	PluginID      string
	DeviceID      string
}

func (lci LlmCreateInput) toCPtr() *C.geniex_LlmCreateInput {
	cPtr := (*C.geniex_LlmCreateInput)(cMalloc(C.sizeof_geniex_LlmCreateInput))
	*cPtr = C.geniex_LlmCreateInput{
		model_name:     cStringIfSet(lci.ModelName),
		model_path:     cStringIfSet(lci.ModelPath),
		tokenizer_path: cStringIfSet(lci.TokenizerPath),
		plugin_id:      cStringIfSet(lci.PluginID),
		device_id:      cStringIfSet(lci.DeviceID),
	}
	lci.Config.fillC(&cPtr.config)
	return cPtr
}

func freeLlmCreateInput(cPtr *C.geniex_LlmCreateInput) {
	if cPtr == nil {
		return
	}
	cFreeIfSet(unsafe.Pointer(cPtr.model_name))
	cFreeIfSet(unsafe.Pointer(cPtr.model_path))
	cFreeIfSet(unsafe.Pointer(cPtr.tokenizer_path))
	cFreeIfSet(unsafe.Pointer(cPtr.plugin_id))
	cFreeIfSet(unsafe.Pointer(cPtr.device_id))
	freeCModelConfig(&cPtr.config)
	C.free(unsafe.Pointer(cPtr))
}

type LlmGenerateInput struct {
	PromptUTF8 string
	InputIDs   []int32
	Config     *GenerationConfig
	OnToken    OnTokenCallback
}

func (lgi LlmGenerateInput) toCPtr() *C.geniex_LlmGenerateInput {
	cPtr := (*C.geniex_LlmGenerateInput)(cMalloc(C.sizeof_geniex_LlmGenerateInput))
	*cPtr = C.geniex_LlmGenerateInput{
		prompt_utf8: cStringIfSet(lgi.PromptUTF8),
	}

	if n := len(lgi.InputIDs); n > 0 {
		raw := cMalloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.int32_t(0))))
		ids := unsafe.Slice((*C.int32_t)(raw), n)
		for i, id := range lgi.InputIDs {
			ids[i] = C.int32_t(id)
		}
		cPtr.input_ids = (*C.int32_t)(raw)
		cPtr.input_ids_count = C.int32_t(n)
	}

	if lgi.Config != nil {
		cPtr.config = lgi.Config.toCPtr()
	}
	return cPtr
}

func freeLlmGenerateInput(cPtr *C.geniex_LlmGenerateInput) {
	if cPtr == nil {
		return
	}
	cFreeIfSet(unsafe.Pointer(cPtr.prompt_utf8))
	cFreeIfSet(unsafe.Pointer(cPtr.input_ids))
	freeGenerationConfig(cPtr.config)
	C.free(unsafe.Pointer(cPtr))
}

type LlmGenerateOutput struct {
	FullText    string
	ProfileData ProfileData
}

func newLlmGenerateOutputFromCPtr(c *C.geniex_LlmGenerateOutput) LlmGenerateOutput {
	if c == nil {
		return LlmGenerateOutput{}
	}
	return LlmGenerateOutput{
		FullText:    C.GoString(c.full_text),
		ProfileData: newProfileDataFromCPtr(c.profile_data),
	}
}

func freeLlmGenerateOutput(ptr *C.geniex_LlmGenerateOutput) {
	if ptr == nil {
		return
	}
	mlFree(unsafe.Pointer(ptr.full_text))
}

type LlmChatMessage struct {
	Role    LlmRole
	Content string
}

type llmChatMessages []LlmChatMessage

func (lcm llmChatMessages) toCPtr() (*C.geniex_LlmChatMessage, C.int32_t) {
	if len(lcm) == 0 {
		return nil, 0
	}
	count := len(lcm)
	raw := cMalloc(C.size_t(count) * C.sizeof_geniex_LlmChatMessage)
	cMessages := unsafe.Slice((*C.geniex_LlmChatMessage)(raw), count)
	for i, msg := range lcm {
		cMessages[i] = C.geniex_LlmChatMessage{
			role:    cStringIfSet(string(msg.Role)),
			content: cStringIfSet(msg.Content),
		}
	}
	return (*C.geniex_LlmChatMessage)(raw), C.int32_t(count)
}

func freeLlmChatMessages(cPtr *C.geniex_LlmChatMessage, count C.int32_t) {
	if cPtr == nil || count == 0 {
		return
	}
	cMessages := unsafe.Slice(cPtr, int(count))
	for i := range cMessages {
		cFreeIfSet(unsafe.Pointer(cMessages[i].role))
		cFreeIfSet(unsafe.Pointer(cMessages[i].content))
	}
	C.free(unsafe.Pointer(cPtr))
}

type LlmApplyChatTemplateInput struct {
	Messages            []LlmChatMessage
	Tools               string
	EnableThink         bool
	AddGenerationPrompt bool
}

func (lati LlmApplyChatTemplateInput) toCPtr() *C.geniex_LlmApplyChatTemplateInput {
	cPtr := (*C.geniex_LlmApplyChatTemplateInput)(cMalloc(C.sizeof_geniex_LlmApplyChatTemplateInput))
	*cPtr = C.geniex_LlmApplyChatTemplateInput{
		tools:                 cStringIfSet(lati.Tools),
		enable_thinking:       C.bool(lati.EnableThink),
		add_generation_prompt: C.bool(lati.AddGenerationPrompt),
	}
	cPtr.messages, cPtr.message_count = llmChatMessages(lati.Messages).toCPtr()
	return cPtr
}

func freeLlmApplyChatTemplateInput(cPtr *C.geniex_LlmApplyChatTemplateInput) {
	if cPtr == nil {
		return
	}
	freeLlmChatMessages(cPtr.messages, cPtr.message_count)
	cFreeIfSet(unsafe.Pointer(cPtr.tools))
	C.free(unsafe.Pointer(cPtr))
}

type LlmApplyChatTemplateOutput struct {
	FormattedText string
}

func newLlmApplyChatTemplateOutputFromCPtr(c *C.geniex_LlmApplyChatTemplateOutput) LlmApplyChatTemplateOutput {
	if c == nil {
		return LlmApplyChatTemplateOutput{}
	}
	return LlmApplyChatTemplateOutput{FormattedText: C.GoString(c.formatted_text)}
}

func freeLlmApplyChatTemplateOutput(cPtr *C.geniex_LlmApplyChatTemplateOutput) {
	if cPtr == nil {
		return
	}
	mlFree(unsafe.Pointer(cPtr.formatted_text))
}

type LLM struct {
	ptr *C.geniex_LLM
}

func NewLLM(input LlmCreateInput) (*LLM, error) {
	slog.Debug("NewLLM called", "input", input)

	cInput := input.toCPtr()
	defer freeLlmCreateInput(cInput)

	var cHandle *C.geniex_LLM
	res := C.geniex_llm_create(cInput, &cHandle)
	if res < 0 {
		return nil, SDKError(res)
	}
	return &LLM{ptr: cHandle}, nil
}

func (l *LLM) Destroy() error {
	slog.Debug("Destroy called", "ptr", l.ptr)
	if l.ptr == nil {
		return nil
	}
	res := C.geniex_llm_destroy(l.ptr)
	if res < 0 {
		return SDKError(res)
	}
	l.ptr = nil
	return nil
}

func (l *LLM) Reset() error {
	slog.Debug("Reset called", "ptr", l.ptr)
	if l.ptr == nil {
		return nil
	}
	res := C.geniex_llm_reset(l.ptr)
	if res < 0 {
		return SDKError(res)
	}
	return nil
}

type LlmSaveKVCacheInput struct {
	Path string
}

func (lsci LlmSaveKVCacheInput) toCPtr() *C.geniex_KvCacheSaveInput {
	cPtr := (*C.geniex_KvCacheSaveInput)(cMalloc(C.sizeof_geniex_KvCacheSaveInput))
	*cPtr = C.geniex_KvCacheSaveInput{path: cStringIfSet(lsci.Path)}
	return cPtr
}

func freeLlmSaveKVCacheInput(cPtr *C.geniex_KvCacheSaveInput) {
	if cPtr == nil {
		return
	}
	cFreeIfSet(unsafe.Pointer(cPtr.path))
	C.free(unsafe.Pointer(cPtr))
}

type LlmLoadKVCacheInput struct {
	Path string
}

func (llci LlmLoadKVCacheInput) toCPtr() *C.geniex_KvCacheLoadInput {
	cPtr := (*C.geniex_KvCacheLoadInput)(cMalloc(C.sizeof_geniex_KvCacheLoadInput))
	*cPtr = C.geniex_KvCacheLoadInput{path: cStringIfSet(llci.Path)}
	return cPtr
}

func freeLlmLoadKVCacheInput(cPtr *C.geniex_KvCacheLoadInput) {
	if cPtr == nil {
		return
	}
	cFreeIfSet(unsafe.Pointer(cPtr.path))
	C.free(unsafe.Pointer(cPtr))
}

func (l *LLM) ApplyChatTemplate(input LlmApplyChatTemplateInput) (*LlmApplyChatTemplateOutput, error) {
	slog.Debug("ApplyChatTemplate called", "input", input)

	cInput := input.toCPtr()
	defer freeLlmApplyChatTemplateInput(cInput)

	var cOutput C.geniex_LlmApplyChatTemplateOutput
	defer freeLlmApplyChatTemplateOutput(&cOutput)

	res := C.geniex_llm_apply_chat_template(l.ptr, cInput, &cOutput)
	if res < 0 {
		return nil, SDKError(res)
	}
	output := newLlmApplyChatTemplateOutputFromCPtr(&cOutput)
	return &output, nil
}

func (l *LLM) SaveKVCache(input LlmSaveKVCacheInput) error {
	slog.Debug("SaveKVCache called", "input", input)

	cInput := input.toCPtr()
	defer freeLlmSaveKVCacheInput(cInput)

	var cOutput C.geniex_KvCacheSaveOutput
	res := C.geniex_llm_save_kv_cache(l.ptr, cInput, &cOutput)
	if res < 0 {
		return SDKError(res)
	}
	return nil
}

func (l *LLM) LoadKVCache(input LlmLoadKVCacheInput) error {
	slog.Debug("LoadKVCache called", "input", input)

	cInput := input.toCPtr()
	defer freeLlmLoadKVCacheInput(cInput)

	var cOutput C.geniex_KvCacheLoadOutput
	res := C.geniex_llm_load_kv_cache(l.ptr, cInput, &cOutput)
	if res < 0 {
		return SDKError(res)
	}
	return nil
}

func (l *LLM) Generate(input LlmGenerateInput) (*LlmGenerateOutput, error) {
	slog.Debug("Generate called", "promptLen", len(input.PromptUTF8), "inputIDsLen", len(input.InputIDs))

	cInput := input.toCPtr()
	defer freeLlmGenerateInput(cInput)

	if input.OnToken != nil {
		h := cgo.NewHandle(input.OnToken)
		defer h.Delete()
		cInput.on_token = C.geniex_token_callback(C.go_generate_stream_on_token)
		cInput.user_data = handleToUserData(h)
	}

	var cOutput C.geniex_LlmGenerateOutput
	defer freeLlmGenerateOutput(&cOutput)

	res := C.geniex_llm_generate(l.ptr, cInput, &cOutput)
	// On context-length errors the SDK still populates whatever was generated
	// before the cutoff; surface that to the caller alongside the error.
	if res < 0 && res != C.GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH {
		return nil, SDKError(res)
	}
	output := newLlmGenerateOutputFromCPtr(&cOutput)
	if res < 0 {
		return &output, SDKError(res)
	}
	return &output, nil
}
