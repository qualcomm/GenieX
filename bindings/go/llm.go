// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

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

type LlmRole string

const (
	LlmRoleSystem    LlmRole = "system"
	LlmRoleUser      LlmRole = "user"
	LlmRoleAssistant LlmRole = "assistant"
)

type LlmCreateInput struct {
	ModelPath     string
	TokenizerPath string
	Config        ModelConfig
	RuntimeID     string
	DeviceID      string
}

func (lci LlmCreateInput) toCPtr() *C.geniex_LlmCreateInput {
	cPtr := (*C.geniex_LlmCreateInput)(cMalloc(C.sizeof_geniex_LlmCreateInput))
	*cPtr = C.geniex_LlmCreateInput{
		model_path:     cStringIfSet(lci.ModelPath),
		tokenizer_path: cStringIfSet(lci.TokenizerPath),
		plugin_id:      cStringIfSet(lci.RuntimeID),
		device_id:      cStringIfSet(lci.DeviceID),
	}
	lci.Config.fillC(&cPtr.config)
	return cPtr
}

func freeLlmCreateInput(cPtr *C.geniex_LlmCreateInput) {
	if cPtr == nil {
		return
	}
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
	free(unsafe.Pointer(ptr.full_text))
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
	free(unsafe.Pointer(cPtr.formatted_text))
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

type LlmModelInfo struct {
	VocabSize int32
	BosToken  int32
	AddBos    bool
}

func (l *LLM) GetModelInfo() (LlmModelInfo, error) {
	if l.ptr == nil {
		return LlmModelInfo{}, SDKError(-1)
	}
	var cInfo C.geniex_LlmModelInfo
	res := C.geniex_llm_get_model_info(l.ptr, &cInfo)
	if res < 0 {
		return LlmModelInfo{}, SDKError(res)
	}
	return LlmModelInfo{
		VocabSize: int32(cInfo.vocab_size),
		BosToken:  int32(cInfo.bos_token),
		AddBos:    cInfo.add_bos != 0,
	}, nil
}

// ForwardLogitsResult holds one prefill-only forward pass. Logits is row-major
// [NRows, RowWidth]. When TopN is 0, RowWidth == VocabSize and TokenIDs is nil
// (column index is the token id). When TopN > 0, RowWidth == min(TopN, VocabSize),
// each row holds its top logits sorted descending, and TokenIDs is the matching
// [NRows, RowWidth] token ids.
type ForwardLogitsResult struct {
	Logits    []float32
	TokenIDs  []int32
	NRows     int
	RowWidth  int
	VocabSize int
}

// ForwardLogits runs a single non-autoregressive forward pass over inputIDs.
// When allPositions is false NRows is 1 (the last token's row); when true it is
// len(inputIDs). topN > 0 reduces each row to its top-N logits (with matching
// token ids); topN == 0 returns the full vocabulary per row. The caller owns any
// special tokens; none are added.
func (l *LLM) ForwardLogits(inputIDs []int32, allPositions bool, topN int) (ForwardLogitsResult, error) {
	if l.ptr == nil {
		return ForwardLogitsResult{}, SDKError(-1)
	}
	if len(inputIDs) == 0 {
		return ForwardLogitsResult{}, SDKError(-100001) // GENIEX_ERROR_COMMON_INVALID_INPUT
	}

	n := len(inputIDs)
	raw := cMalloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.int32_t(0))))
	defer C.free(raw)
	ids := unsafe.Slice((*C.int32_t)(raw), n)
	for i, id := range inputIDs {
		ids[i] = C.int32_t(id)
	}

	cInput := C.geniex_LlmForwardLogitsInput{
		input_ids:       (*C.int32_t)(raw),
		input_ids_count: C.int32_t(n),
		all_positions:   C.bool(allPositions),
		top_n:           C.int32_t(topN),
	}

	var cOutput C.geniex_LlmForwardLogitsOutput
	res := C.geniex_llm_forward_logits(l.ptr, &cInput, &cOutput)
	if res < 0 {
		return ForwardLogitsResult{}, SDKError(res)
	}
	defer C.geniex_free(unsafe.Pointer(cOutput.logits))
	defer C.geniex_free(unsafe.Pointer(cOutput.token_ids))

	out := ForwardLogitsResult{
		NRows:     int(cOutput.n_rows),
		RowWidth:  int(cOutput.row_width),
		VocabSize: int(cOutput.vocab_size),
	}
	total := out.NRows * out.RowWidth
	if total > 0 && cOutput.logits != nil {
		out.Logits = make([]float32, total)
		src := unsafe.Slice((*C.float)(cOutput.logits), total)
		for i := 0; i < total; i++ {
			out.Logits[i] = float32(src[i])
		}
	}
	if total > 0 && cOutput.token_ids != nil {
		out.TokenIDs = make([]int32, total)
		src := unsafe.Slice((*C.int32_t)(cOutput.token_ids), total)
		for i := 0; i < total; i++ {
			out.TokenIDs[i] = int32(src[i])
		}
	}
	return out, nil
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
	// The SDK populates whatever was generated before any cutoff, so surface the
	// partial output to the caller even on error (e.g. truncation or a mid-run
	// decode failure).
	output := newLlmGenerateOutputFromCPtr(&cOutput)
	if res < 0 {
		return &output, SDKError(res)
	}
	return &output, nil
}

// LCOV_EXCL_STOP
