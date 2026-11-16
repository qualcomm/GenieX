// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

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
	TTFT              int64
	MediaTime         int64
	PromptTime        int64
	DecodeTime        int64
	PromptTokens      int64 // Evaluated in prefill.
	TotalPromptTokens int64 // Full requested input, including tokens reused from the KV cache.
	GeneratedTokens   int64
	PrefillSpeed      float64
	DecodingSpeed     float64
	DraftNTotal       int64
	DraftNAccepted    int64
	StopReason        string
}

func (p ProfileData) CachedPromptTokens() int64 {
	return max(p.TotalPromptTokens-p.PromptTokens, 0)
}

func (p ProfileData) TotalTokens() int64 {
	return p.PromptTokens + p.GeneratedTokens
}

func (p ProfileData) TotalTimeUs() int64 {
	return p.MediaTime + p.PromptTime + p.DecodeTime
}

// LCOV_EXCL_START
func newProfileDataFromCPtr(c C.geniex_ProfileData) ProfileData {
	return ProfileData{
		TTFT:              int64(c.ttft),
		MediaTime:         int64(c.media_time),
		PromptTime:        int64(c.prompt_time),
		DecodeTime:        int64(c.decode_time),
		PromptTokens:      int64(c.prompt_tokens),
		TotalPromptTokens: int64(c.total_prompt_tokens),
		GeneratedTokens:   int64(c.generated_tokens),
		PrefillSpeed:      float64(c.prefill_speed),
		DecodingSpeed:     float64(c.decoding_speed),
		DraftNTotal:       int64(c.draft_n_total),
		DraftNAccepted:    int64(c.draft_n_accepted),
		StopReason:        C.GoString(c.stop_reason),
	}
}

// LCOV_EXCL_STOP

// DecideOption preserves boolean choice values without serializing the question as JSON.
// A nil Description is absent; a pointer to "" is an explicit empty description.
type DecideOption struct {
	Key         string
	KeyIsBool   bool
	Description *string
}

type DecideQuestion struct {
	ID           string
	Type         string
	Instructions string
	Options      []DecideOption
}

type DecideInput struct {
	State         string
	Questions     []DecideQuestion
	QuestionIndex int
	ImagePaths    []string
	AudioPaths    []string
}

type DecideOutput struct {
	Logits        []float32
	Probabilities []float32
	InputTokens   int
	Temperature   float32
	DecisionType  string
}

// LCOV_EXCL_START
// Allocate nested C-owned copies so no Go pointers cross the cgo call.
func (input DecideInput) toCPtr() *C.geniex_DecideInput {
	cPtr := (*C.geniex_DecideInput)(cMalloc(C.sizeof_geniex_DecideInput))
	*cPtr = C.geniex_DecideInput{
		state:          C.CString(input.State),
		question_index: C.int32_t(input.QuestionIndex),
	}
	if len(input.Questions) > 0 {
		raw := cMalloc(C.size_t(len(input.Questions)) * C.sizeof_geniex_DecideQuestion)
		questions := unsafe.Slice((*C.geniex_DecideQuestion)(raw), len(input.Questions))
		for i, question := range input.Questions {
			questions[i] = C.geniex_DecideQuestion{
				id:            cStringIfSet(question.ID),
				question_type: C.CString(question.Type),
				instructions:  C.CString(question.Instructions),
			}
			if len(question.Options) > 0 {
				optionsRaw := cMalloc(C.size_t(len(question.Options)) * C.sizeof_geniex_DecideOption)
				options := unsafe.Slice((*C.geniex_DecideOption)(optionsRaw), len(question.Options))
				for j, option := range question.Options {
					options[j] = C.geniex_DecideOption{
						key:         C.CString(option.Key),
						key_is_bool: C.bool(option.KeyIsBool),
					}
					if option.Description != nil {
						options[j].description = C.CString(*option.Description)
					}
				}
				questions[i].options = (*C.geniex_DecideOption)(optionsRaw)
				questions[i].option_count = C.int32_t(len(options))
			}
		}
		cPtr.questions = (*C.geniex_DecideQuestion)(raw)
		cPtr.question_count = C.int32_t(len(questions))
	}
	if len(input.ImagePaths) > 0 {
		paths, count := sliceToCCharArray(input.ImagePaths)
		cPtr.image_paths = (**C.char)(unsafe.Pointer(paths))
		cPtr.image_count = count
	}
	if len(input.AudioPaths) > 0 {
		paths, count := sliceToCCharArray(input.AudioPaths)
		cPtr.audio_paths = (**C.char)(unsafe.Pointer(paths))
		cPtr.audio_count = count
	}
	return cPtr
}

// Free nested allocations in the reverse ownership direction after the SDK call.
func freeDecideInput(cPtr *C.geniex_DecideInput) {
	if cPtr == nil {
		return
	}
	cFreeIfSet(unsafe.Pointer(cPtr.state))
	if cPtr.questions != nil {
		questions := unsafe.Slice((*C.geniex_DecideQuestion)(unsafe.Pointer(cPtr.questions)), int(cPtr.question_count))
		for _, question := range questions {
			cFreeIfSet(unsafe.Pointer(question.id))
			cFreeIfSet(unsafe.Pointer(question.question_type))
			cFreeIfSet(unsafe.Pointer(question.instructions))
			if question.options != nil {
				options := unsafe.Slice((*C.geniex_DecideOption)(unsafe.Pointer(question.options)), int(question.option_count))
				for _, option := range options {
					cFreeIfSet(unsafe.Pointer(option.key))
					cFreeIfSet(unsafe.Pointer(option.description))
				}
				C.free(unsafe.Pointer(question.options))
			}
		}
		C.free(unsafe.Pointer(cPtr.questions))
	}
	freeCCharArray((**C.char)(unsafe.Pointer(cPtr.image_paths)), cPtr.image_count)
	freeCCharArray((**C.char)(unsafe.Pointer(cPtr.audio_paths)), cPtr.audio_count)
	C.free(unsafe.Pointer(cPtr))
}

// Multi-pass models return calibrated probabilities instead of raw logits.
func decideOutput(cOutput *C.geniex_DecideOutput) DecideOutput {
	output := DecideOutput{
		InputTokens:  int(cOutput.input_tokens),
		Temperature:  float32(cOutput.temperature),
		DecisionType: C.GoString(cOutput.decision_type),
	}
	if cOutput.option_count > 0 {
		if cOutput.logits != nil {
			values := unsafe.Slice((*C.float)(cOutput.logits), int(cOutput.option_count))
			output.Logits = make([]float32, len(values))
			for i, value := range values {
				output.Logits[i] = float32(value)
			}
		}
		if cOutput.probabilities != nil {
			values := unsafe.Slice((*C.float)(cOutput.probabilities), int(cOutput.option_count))
			output.Probabilities = make([]float32, len(values))
			for i, value := range values {
				output.Probabilities[i] = float32(value)
			}
		}
	}
	return output
}

func freeDecideOutput(cOutput *C.geniex_DecideOutput) {
	if cOutput == nil {
		return
	}
	free(unsafe.Pointer(cOutput.logits))
	free(unsafe.Pointer(cOutput.probabilities))
	free(unsafe.Pointer(cOutput.decision_type))
}

// LCOV_EXCL_STOP

type GrammarType int32

type GrammarTriggerType int32

const (
	GrammarTypeNone GrammarType = iota
	GrammarTypeUser
	GrammarTypeOutputFormat
	GrammarTypeToolCalls
)

const (
	GrammarTriggerToken GrammarTriggerType = iota
	GrammarTriggerWord
	GrammarTriggerPattern
	GrammarTriggerPatternFull
)

type GrammarTrigger struct {
	Type  GrammarTriggerType
	Value string
	Token int32
}

type ChatTemplateGrammar struct {
	Type             GrammarType
	Grammar          string
	Lazy             bool
	GenerationPrompt string
	Triggers         []GrammarTrigger
	PreservedTokens  []int32
}

type SamplerConfig struct {
	Temperature         float32
	TopP                float32
	TopK                int32
	MinP                float32
	RepetitionPenalty   float32
	PresencePenalty     float32
	FrequencyPenalty    float32
	Seed                int32
	GrammarPath         string
	GrammarString       string
	ChatTemplateGrammar ChatTemplateGrammar
}

func chatTemplateGrammarFromCPtr(src *C.geniex_ChatTemplateGrammar) ChatTemplateGrammar {
	if src == nil {
		return ChatTemplateGrammar{}
	}
	out := ChatTemplateGrammar{
		Type:             GrammarType(src.grammar_type),
		Grammar:          C.GoString(src.grammar),
		Lazy:             bool(src.lazy),
		GenerationPrompt: C.GoString(src.generation_prompt),
	}
	if src.trigger_count > 0 && src.triggers != nil {
		triggers := unsafe.Slice(src.triggers, int(src.trigger_count))
		out.Triggers = make([]GrammarTrigger, len(triggers))
		for i, trigger := range triggers {
			out.Triggers[i] = GrammarTrigger{
				Type:  GrammarTriggerType(trigger.trigger_type),
				Value: C.GoString(trigger.value),
				Token: int32(trigger.token),
			}
		}
	}
	if src.preserved_token_count > 0 && src.preserved_tokens != nil {
		tokens := unsafe.Slice(src.preserved_tokens, int(src.preserved_token_count))
		out.PreservedTokens = make([]int32, len(tokens))
		for i, token := range tokens {
			out.PreservedTokens[i] = int32(token)
		}
	}
	return out
}

func fillChatTemplateGrammar(dst *C.geniex_ChatTemplateGrammar, src ChatTemplateGrammar) {
	if dst == nil || src.Type == GrammarTypeNone {
		return
	}
	dst.grammar_type = C.geniex_GrammarType(src.Type)
	dst.grammar = cStringIfSet(src.Grammar)
	dst.lazy = C.bool(src.Lazy)
	dst.generation_prompt = cStringIfSet(src.GenerationPrompt)
	if len(src.Triggers) > 0 {
		raw := cMalloc(C.size_t(len(src.Triggers)) * C.sizeof_geniex_GrammarTrigger)
		triggers := unsafe.Slice((*C.geniex_GrammarTrigger)(raw), len(src.Triggers))
		for i, trigger := range src.Triggers {
			triggers[i] = C.geniex_GrammarTrigger{
				trigger_type: C.geniex_GrammarTriggerType(trigger.Type),
				value:        cStringIfSet(trigger.Value),
				token:        C.int32_t(trigger.Token),
			}
		}
		dst.triggers = (*C.geniex_GrammarTrigger)(raw)
		dst.trigger_count = C.int32_t(len(src.Triggers))
	}
	if len(src.PreservedTokens) > 0 {
		raw := cMalloc(C.size_t(len(src.PreservedTokens)) * C.sizeof_int32_t)
		tokens := unsafe.Slice((*C.int32_t)(raw), len(src.PreservedTokens))
		for i, token := range src.PreservedTokens {
			tokens[i] = C.int32_t(token)
		}
		dst.preserved_tokens = (*C.int32_t)(raw)
		dst.preserved_token_count = C.int32_t(len(src.PreservedTokens))
	}
}

func freeChatTemplateGrammarInput(grammar *C.geniex_ChatTemplateGrammar) {
	if grammar == nil {
		return
	}
	cFreeIfSet(unsafe.Pointer(grammar.grammar))
	cFreeIfSet(unsafe.Pointer(grammar.generation_prompt))
	if grammar.triggers != nil {
		triggers := unsafe.Slice(grammar.triggers, int(grammar.trigger_count))
		for _, trigger := range triggers {
			cFreeIfSet(unsafe.Pointer(trigger.value))
		}
		cFreeIfSet(unsafe.Pointer(grammar.triggers))
	}
	cFreeIfSet(unsafe.Pointer(grammar.preserved_tokens))
}

func freeChatTemplateGrammarOutput(grammar *C.geniex_ChatTemplateGrammar) {
	if grammar == nil {
		return
	}
	free(unsafe.Pointer(grammar.grammar))
	free(unsafe.Pointer(grammar.generation_prompt))
	if grammar.triggers != nil {
		triggers := unsafe.Slice(grammar.triggers, int(grammar.trigger_count))
		for _, trigger := range triggers {
			free(unsafe.Pointer(trigger.value))
		}
		free(unsafe.Pointer(grammar.triggers))
	}
	free(unsafe.Pointer(grammar.preserved_tokens))
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
	}
	fillChatTemplateGrammar(&cPtr.chat_template_grammar, sc.ChatTemplateGrammar)
	return cPtr
}

func freeSamplerConfig(cPtr *C.geniex_SamplerConfig) {
	if cPtr == nil {
		return
	}
	cFreeIfSet(unsafe.Pointer(cPtr.grammar_path))
	cFreeIfSet(unsafe.Pointer(cPtr.grammar_string))
	freeChatTemplateGrammarInput(&cPtr.chat_template_grammar)
	C.free(unsafe.Pointer(cPtr))
}

// LCOV_EXCL_STOP

type GenerationConfig struct {
	MaxTokens     int32
	Stop          []string
	SamplerConfig *SamplerConfig
	ImagePaths    []string
	AudioPaths    []string

	// Opt-in ring-buffer context eviction (qairt only).
	SlidingWindow      bool
	SlidingWindowNKeep int32 // 0 = plugin default (4)
}

// LCOV_EXCL_START
func (gc GenerationConfig) toCPtr() *C.geniex_GenerationConfig {
	cPtr := (*C.geniex_GenerationConfig)(cMalloc(C.sizeof_geniex_GenerationConfig))
	*cPtr = C.geniex_GenerationConfig{
		max_tokens:            C.int32_t(gc.MaxTokens),
		sliding_window:        C.bool(gc.SlidingWindow),
		sliding_window_n_keep: C.int32_t(gc.SlidingWindowNKeep),
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
	SpecType            string
	SpecDraftModel      string
	SpecNMax            int32
	SpecNMin            int32
	SpecPMin            float32
	PowerMode           PowerMode
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
		spec_type:             cStringIfSet(mc.SpecType),
		spec_draft_model:      cStringIfSet(mc.SpecDraftModel),
		spec_n_max:            C.int32_t(mc.SpecNMax),
		spec_n_min:            C.int32_t(mc.SpecNMin),
		spec_p_min:            C.float(mc.SpecPMin),
		power_mode:            C.geniex_PowerMode(mc.PowerMode),
	}
}

func freeCModelConfig(c *C.geniex_ModelConfig) {
	if c == nil {
		return
	}
	cFreeIfSet(unsafe.Pointer(c.chat_template_path))
	cFreeIfSet(unsafe.Pointer(c.chat_template_content))
	cFreeIfSet(unsafe.Pointer(c.spec_type))
	cFreeIfSet(unsafe.Pointer(c.spec_draft_model))
}

// LCOV_EXCL_STOP
