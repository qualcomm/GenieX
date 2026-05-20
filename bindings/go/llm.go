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
	"unsafe"
)

// TODO: use same role const with VLM and others
// LLMRole represents different roles in a chat conversation
type LLMRole string

const (
	LLMRoleSystem    LLMRole = "system"    // System role for instructions
	LLMRoleUser      LLMRole = "user"      // User role for queries
	LLMRoleAssistant LLMRole = "assistant" // Assistant role for responses
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
	cPtr := (*C.geniex_LlmCreateInput)(C.malloc(C.size_t(unsafe.Sizeof(C.geniex_LlmCreateInput{}))))
	*cPtr = C.geniex_LlmCreateInput{}

	if lci.ModelName != "" {
		cPtr.model_name = C.CString(lci.ModelName)
	}
	if lci.ModelPath != "" {
		cPtr.model_path = C.CString(lci.ModelPath)
	}
	if lci.TokenizerPath != "" {
		cPtr.tokenizer_path = C.CString(lci.TokenizerPath)
	}
	if lci.PluginID != "" {
		cPtr.plugin_id = C.CString(lci.PluginID)
	}
	if lci.DeviceID != "" {
		cPtr.device_id = C.CString(lci.DeviceID)
	}

	// no need to use toCPtr() here, because we are not using the config pointer
	cPtr.config = C.geniex_ModelConfig{
		n_ctx:           C.int32_t(lci.Config.NCtx),
		n_threads:       C.int32_t(lci.Config.NThreads),
		n_threads_batch: C.int32_t(lci.Config.NThreadsBatch),
		n_batch:         C.int32_t(lci.Config.NBatch),
		n_ubatch:        C.int32_t(lci.Config.NUbatch),
		n_seq_max:       C.int32_t(lci.Config.NSeqMax),
		n_gpu_layers:    C.int32_t(lci.Config.NGpuLayers),
	}
	if lci.Config.ChatTemplatePath != "" {
		cPtr.config.chat_template_path = C.CString(lci.Config.ChatTemplatePath)
	}
	if lci.Config.ChatTemplateContent != "" {
		cPtr.config.chat_template_content = C.CString(lci.Config.ChatTemplateContent)
	}

	return cPtr
}

func freeLlmCreateInput(cPtr *C.geniex_LlmCreateInput) {
	if cPtr != nil {
		// Free string fields
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

		// Free nested ModelConfig - config is a value, not a pointer
		// We need to free the string fields manually
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

type LlmGenerateInput struct {
	PromptUTF8 string
	InputIDs   []int32
	Config     *GenerationConfig
	OnToken    OnTokenCallback
	// UserData   unsafe.Pointer
}

func (lgi LlmGenerateInput) toCPtr() *C.geniex_LlmGenerateInput {
	cPtr := (*C.geniex_LlmGenerateInput)(C.malloc(C.size_t(unsafe.Sizeof(C.geniex_LlmGenerateInput{}))))
	*cPtr = C.geniex_LlmGenerateInput{}

	if lgi.PromptUTF8 != "" {
		cPtr.prompt_utf8 = C.CString(lgi.PromptUTF8)
	}

	if len(lgi.InputIDs) > 0 {
		cPtr.input_ids_count = C.int32_t(len(lgi.InputIDs))
		cInputIDs := (*C.int32_t)(C.malloc(C.size_t(len(lgi.InputIDs)) * C.size_t(unsafe.Sizeof(C.int32_t(0)))))
		cInputIDsSlice := unsafe.Slice(cInputIDs, len(lgi.InputIDs))
		for i, id := range lgi.InputIDs {
			cInputIDsSlice[i] = C.int32_t(id)
		}
		cPtr.input_ids = cInputIDs
	} else {
		cPtr.input_ids = nil
		cPtr.input_ids_count = 0
	}

	if lgi.Config != nil {
		cPtr.config = lgi.Config.toCPtr()
	} else {
		cPtr.config = nil
	}

	// Note: on_token and user_data should be set by the caller
	cPtr.on_token = nil
	cPtr.user_data = nil

	return cPtr
}

func freeLlmGenerateInput(cPtr *C.geniex_LlmGenerateInput) {
	if cPtr != nil {
		if cPtr.prompt_utf8 != nil {
			C.free(unsafe.Pointer(cPtr.prompt_utf8))
		}

		if cPtr.input_ids != nil {
			C.free(unsafe.Pointer(cPtr.input_ids))
		}

		if cPtr.config != nil {
			freeGenerationConfig(cPtr.config)
		}

		C.free(unsafe.Pointer(cPtr))
	}
}

type LlmGenerateOutput struct {
	FullText    string
	ProfileData ProfileData
}

func newLlmGenerateOutputFromCPtr(c *C.geniex_LlmGenerateOutput) LlmGenerateOutput {
	output := LlmGenerateOutput{}

	if c == nil {
		return output
	}

	if c.full_text != nil {
		output.FullText = C.GoString(c.full_text)
	}
	output.ProfileData = newProfileDataFromCPtr(c.profile_data)
	return output
}

func freeLlmGenerateOutput(ptr *C.geniex_LlmGenerateOutput) {
	if ptr == nil {
		return
	}
	if ptr.full_text != nil {
		mlFree(unsafe.Pointer(ptr.full_text))
	}
}

type LlmChatMessage struct {
	Role    LLMRole
	Content string
}

type llmChatMessages []LlmChatMessage

func (lcm llmChatMessages) toCPtr() (*C.geniex_LlmChatMessage, C.int32_t) {
	if len(lcm) == 0 {
		return nil, 0
	}

	count := len(lcm)
	raw := C.malloc(C.size_t(count * C.sizeof_geniex_LlmChatMessage))
	cMessages := unsafe.Slice((*C.geniex_LlmChatMessage)(raw), count)

	for i, msg := range lcm {
		if msg.Role != "" {
			cMessages[i].role = C.CString(string(msg.Role))
			cMessages[i].content = C.CString(msg.Content)
		}
	}

	return (*C.geniex_LlmChatMessage)(raw), C.int32_t(count)
}

func freeLlmChatMessages(cPtr *C.geniex_LlmChatMessage, count C.int32_t) {
	if cPtr == nil || count == 0 {
		return
	}

	cMessages := unsafe.Slice(cPtr, int(count))
	for i := range count {
		if cMessages[i].role != nil {
			C.free(unsafe.Pointer(cMessages[i].role))
		}
		if cMessages[i].content != nil {
			C.free(unsafe.Pointer(cMessages[i].content))
		}
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
	cPtr := (*C.geniex_LlmApplyChatTemplateInput)(C.malloc(C.size_t(unsafe.Sizeof(C.geniex_LlmApplyChatTemplateInput{}))))
	*cPtr = C.geniex_LlmApplyChatTemplateInput{}

	if len(lati.Messages) > 0 {
		cMessages, messageCount := llmChatMessages(lati.Messages).toCPtr()
		cPtr.messages = cMessages
		cPtr.message_count = C.int32_t(messageCount)
	}

	if lati.Tools != "" {
		cPtr.tools = C.CString(lati.Tools)
	}

	cPtr.enable_thinking = C.bool(lati.EnableThink)
	cPtr.add_generation_prompt = C.bool(lati.AddGenerationPrompt)

	return cPtr
}

func freeLlmApplyChatTemplateInput(cPtr *C.geniex_LlmApplyChatTemplateInput) {
	if cPtr == nil {
		return
	}
	freeLlmChatMessages(cPtr.messages, cPtr.message_count)
	if cPtr.tools != nil {
		C.free(unsafe.Pointer(cPtr.tools))
	}
	C.free(unsafe.Pointer(cPtr))
}

type LlmApplyChatTemplateOutput struct {
	FormattedText string
}

func newLlmApplyChatTemplateOutputFromCPtr(c *C.geniex_LlmApplyChatTemplateOutput) LlmApplyChatTemplateOutput {
	output := LlmApplyChatTemplateOutput{}

	if c == nil {
		return output
	}

	if c.formatted_text != nil {
		output.FormattedText = C.GoString(c.formatted_text)
	}

	return output
}

func freeLlmApplyChatTemplateOutput(cPtr *C.geniex_LlmApplyChatTemplateOutput) {
	if cPtr == nil {
		return
	}
	if cPtr.formatted_text != nil {
		mlFree(unsafe.Pointer(cPtr.formatted_text))
	}
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

// LlmSaveKVCacheInput represents input for saving LLM KV cache
type LlmSaveKVCacheInput struct {
	Path string
}

func (lsci LlmSaveKVCacheInput) toCPtr() *C.geniex_KvCacheSaveInput {
	cPtr := (*C.geniex_KvCacheSaveInput)(C.malloc(C.size_t(unsafe.Sizeof(C.geniex_KvCacheSaveInput{}))))
	*cPtr = C.geniex_KvCacheSaveInput{}

	if lsci.Path != "" {
		cPtr.path = C.CString(lsci.Path)
	}

	return cPtr
}

func freeLlmSaveKVCacheInput(cPtr *C.geniex_KvCacheSaveInput) {
	if cPtr != nil {
		if cPtr.path != nil {
			C.free(unsafe.Pointer(cPtr.path))
		}
		C.free(unsafe.Pointer(cPtr))
	}
}

// LlmSaveKVCacheOutput represents output from saving LLM KV cache
type LlmSaveKVCacheOutput struct {
	Reserved any
}

func newLlmSaveKVCacheOutputFromCPtr(c *C.geniex_KvCacheSaveOutput) LlmSaveKVCacheOutput {
	output := LlmSaveKVCacheOutput{}

	if c == nil {
		return output
	}

	// Currently no fields to copy from C structure
	return output
}

func freeLlmSaveKVCacheOutput(cPtr *C.geniex_KvCacheSaveOutput) {
	if cPtr == nil {
		return
	}
	// Currently no fields to free
}

// LlmLoadKVCacheInput represents input for loading LLM KV cache
type LlmLoadKVCacheInput struct {
	Path string
}

func (llci LlmLoadKVCacheInput) toCPtr() *C.geniex_KvCacheLoadInput {
	cPtr := (*C.geniex_KvCacheLoadInput)(C.malloc(C.size_t(unsafe.Sizeof(C.geniex_KvCacheLoadInput{}))))
	*cPtr = C.geniex_KvCacheLoadInput{}

	if llci.Path != "" {
		cPtr.path = C.CString(llci.Path)
	}

	return cPtr
}

func freeLlmLoadKVCacheInput(cPtr *C.geniex_KvCacheLoadInput) {
	if cPtr != nil {
		if cPtr.path != nil {
			C.free(unsafe.Pointer(cPtr.path))
		}
		C.free(unsafe.Pointer(cPtr))
	}
}

// LlmLoadKVCacheOutput represents output from loading LLM KV cache
type LlmLoadKVCacheOutput struct {
	Reserved any
}

func newLlmLoadKVCacheOutputFromCPtr(c *C.geniex_KvCacheLoadOutput) LlmLoadKVCacheOutput {
	output := LlmLoadKVCacheOutput{}

	if c == nil {
		return output
	}

	// Currently no fields to copy from C structure
	return output
}

func freeLlmLoadKVCacheOutput(cPtr *C.geniex_KvCacheLoadOutput) {
	if cPtr == nil {
		return
	}
	// Currently no fields to free
}

func (l *LLM) ApplyChatTemplate(input LlmApplyChatTemplateInput) (*LlmApplyChatTemplateOutput, error) {
	slog.Debug("ApplyChatTemplate called", "input", input)

	cinput := input.toCPtr()
	defer freeLlmApplyChatTemplateInput(cinput)

	var cOutput C.geniex_LlmApplyChatTemplateOutput
	defer freeLlmApplyChatTemplateOutput(&cOutput)

	res := C.geniex_llm_apply_chat_template(l.ptr, cinput, &cOutput)
	if res < 0 {
		return nil, SDKError(res)
	}

	output := newLlmApplyChatTemplateOutputFromCPtr(&cOutput)

	return &output, nil
}

func (l *LLM) SaveKVCache(input LlmSaveKVCacheInput) (*LlmSaveKVCacheOutput, error) {
	slog.Debug("SaveKVCache called", "input", input)

	cInput := input.toCPtr()
	defer freeLlmSaveKVCacheInput(cInput)

	var cOutput C.geniex_KvCacheSaveOutput
	defer freeLlmSaveKVCacheOutput(&cOutput)

	res := C.geniex_llm_save_kv_cache(l.ptr, cInput, &cOutput)
	if res < 0 {
		return nil, SDKError(res)
	}

	output := newLlmSaveKVCacheOutputFromCPtr(&cOutput)

	return &output, nil
}

func (l *LLM) LoadKVCache(input LlmLoadKVCacheInput) (*LlmLoadKVCacheOutput, error) {
	slog.Debug("LoadKVCache called", "input", input)

	cInput := input.toCPtr()
	defer freeLlmLoadKVCacheInput(cInput)

	var cOutput C.geniex_KvCacheLoadOutput
	defer freeLlmLoadKVCacheOutput(&cOutput)

	res := C.geniex_llm_load_kv_cache(l.ptr, cInput, &cOutput)
	if res < 0 {
		return nil, SDKError(res)
	}

	output := newLlmLoadKVCacheOutputFromCPtr(&cOutput)

	return &output, nil
}

func (l *LLM) Generate(input LlmGenerateInput) (LlmGenerateOutput, error) {
	slog.Debug("Generate called", "input", input, "config", input.Config, "sampler", input.Config.SamplerConfig)

	cInput := input.toCPtr()
	defer freeLlmGenerateInput(cInput)

	// set the callback
	onToken = input.OnToken
	defer func() {
		onToken = nil // reset to default
	}()
	cInput.on_token = C.geniex_token_callback(C.go_generate_stream_on_token)

	var cOutput C.geniex_LlmGenerateOutput
	defer freeLlmGenerateOutput(&cOutput)

	res := C.geniex_llm_generate(l.ptr, cInput, &cOutput)
	if res < 0 && res != C.GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH {
		return LlmGenerateOutput{}, SDKError(res)
	}
	output := newLlmGenerateOutputFromCPtr(&cOutput)
	if res < 0 {
		return output, SDKError(res)
	}
	return output, nil
}
