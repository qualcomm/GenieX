// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import (
	"encoding/binary"
	"hash"
	"hash/fnv"

	"github.com/openai/openai-go/v3"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

// SessionKey identifies a conversation as one hash per turn (chat: one per
// message; completions/logits: a single turn). A cached model is reset when
// the new request's key isn't a continuation of the last one served.
type SessionKey []uint64

// IsContinuation reports whether next carries on from last: the same
// sequence of turns, with zero or more new ones appended.
func IsContinuation(last, next SessionKey) bool {
	if len(last) > len(next) {
		return false
	}
	for i, v := range last {
		if next[i] != v {
			return false
		}
	}
	return true
}

// hashFields folds a turn's fields into one SessionKey element, length-prefixed
// so ("ab","c") and ("a","bc") can't collide.
func hashFields(fields ...string) uint64 {
	h := fnv.New64a()
	var lenBuf [8]byte
	for _, f := range fields {
		binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(f)))
		h.Write(lenBuf[:])
		h.Write([]byte(f))
	}
	return h.Sum64()
}

// HashText returns the single-turn SessionKey for a plain-text prompt (the
// /v1/completions endpoint, which has no message history).
func HashText(text string) SessionKey {
	return SessionKey{hashFields(text)}
}

// HashTokens returns the single-turn SessionKey for a token-ID prompt (the
// /v1/logits endpoint, which has no text form).
func HashTokens(ids []int32) SessionKey {
	h := fnv.New64a()
	writeInt32s(h, ids)
	return SessionKey{h.Sum64()}
}

func writeInt32s(h hash.Hash, ids []int32) {
	var buf [4]byte
	for _, id := range ids {
		binary.LittleEndian.PutUint32(buf[:], uint32(id))
		h.Write(buf[:])
	}
}

// ToolCallFields flattens tool calls into a field list for HashMessages.
func ToolCallFields(tcs []geniex_sdk.ToolCall) []string {
	fields := make([]string, 0, len(tcs)*3)
	for _, tc := range tcs {
		fields = append(fields, tc.ID, tc.Name, tc.Arguments)
	}
	return fields
}

// HashMessages builds a SessionKey by hashing each message's fields, as
// produced by fields. Hash the client-sent content, not a server-derived
// artifact (e.g. a per-request temp file path) — that would never repeat
// across requests and so would never be recognized as a continuation.
func HashMessages[M any](msgs []M, fields func(M) []string) SessionKey {
	key := make(SessionKey, len(msgs))
	for i, m := range msgs {
		key[i] = hashFields(fields(m)...)
	}
	return key
}

// SessionKeyOf hashes the model-ready LLM messages GenieX is about to feed
// the plugin, keyed stably regardless of the request's raw JSON formatting.
func SessionKeyOf(messages []geniex_sdk.LlmChatMessage) SessionKey {
	return HashMessages(messages, func(m geniex_sdk.LlmChatMessage) []string {
		return append([]string{string(m.Role), m.Content, m.ToolCallID, m.ToolName}, ToolCallFields(m.ToolCalls)...)
	})
}

// SessionKeyOfVLMRequest hashes the raw request messages, not the
// model-ready ones buildVLMMessages produces — those carry a per-request
// temp file path for each image/audio part instead of its source.
func SessionKeyOfVLMRequest(msgs []openai.ChatCompletionMessageParamUnion) SessionKey {
	return HashMessages(msgs, func(msg openai.ChatCompletionMessageParamUnion) []string {
		fields := []string{}
		if role := msg.GetRole(); role != nil {
			fields = append(fields, *role)
		}
		switch content := msg.GetContent().AsAny().(type) {
		case *string:
			fields = append(fields, "text", *content)
		case *[]openai.ChatCompletionContentPartTextParam:
			for _, ct := range *content {
				fields = append(fields, "text", ct.Text)
			}
		case *[]openai.ChatCompletionContentPartUnionParam:
			for _, ct := range *content {
				switch *ct.GetType() {
				case "text":
					fields = append(fields, "text", *ct.GetText())
				case "image_url":
					fields = append(fields, "image", ct.GetImageURL().URL)
				case "input_audio":
					fields = append(fields, "audio", ct.GetInputAudio().Data)
				}
			}
		case *[]openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion:
			for _, ct := range *content {
				if *ct.GetType() == "text" {
					fields = append(fields, "text", *ct.GetText())
				}
			}
		}
		if toolCallID := msg.GetToolCallID(); toolCallID != nil {
			fields = append(fields, *toolCallID)
		}
		for _, tc := range msg.GetToolCalls() {
			if fn := tc.GetFunction(); fn != nil {
				fields = append(fields, fn.Name, fn.Arguments)
			}
			if id := tc.GetID(); id != nil {
				fields = append(fields, *id)
			}
		}
		return fields
	})
}
