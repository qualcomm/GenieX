// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"testing"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

func TestPackageBuilds(t *testing.T) {}

func TestProfile2TimingsUsesSDKMicroseconds(t *testing.T) {
	got := profile2Timings(geniex_sdk.ProfileData{
		PromptTokens:    128,
		GeneratedTokens: 32,
		PromptTime:      2_500_000,
		DecodeTime:      1_600_000,
		TTFT:            2_550_000,
		PrefillSpeed:    51.2,
		DecodingSpeed:   20,
	})
	if got.PromptN != 128 || got.PromptMS != 2500 || got.PromptPerSecond != 51.2 ||
		got.PredictedN != 32 || got.PredictedMS != 1600 || got.PredictedPerSecond != 20 || got.TTFTMS != 2550 {
		t.Fatalf("unexpected timings: %+v", got)
	}
}

func TestFinishChunkCarriesAuthoritativeTimings(t *testing.T) {
	timings := llamaTimings{PromptN: 10, PromptMS: 20, PredictedN: 30, PredictedMS: 40}
	data, err := json.Marshal(finishChunk("stop", timings, "local/test:Q4_0"))
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got["timings"] == nil {
		t.Fatalf("final chunk omitted timings: %s", data)
	}
	if got["model"] != "local/test:Q4_0" {
		t.Fatalf("final chunk omitted model: %s", data)
	}
}

func TestNormalizeLlamaWebUIRequest(t *testing.T) {
	t.Setenv("GENIEX_MODEL", "local/gemma-4-E2B-it:Q4_0")
	param := defaultChatCompletionRequest()
	if err := json.Unmarshal([]byte(`{"max_tokens":64,"repeat_penalty":1.25,"chat_template_kwargs":{"enable_thinking":false}}`), &param); err != nil {
		t.Fatal(err)
	}
	normalizeLlamaWebUIRequest(&param)
	if param.MaxCompletionTokens.Value != 64 {
		t.Fatalf("max_tokens alias not applied: %d", param.MaxCompletionTokens.Value)
	}
	if param.RepetitionPenalty != 1.25 {
		t.Fatalf("repeat_penalty alias not applied: %f", param.RepetitionPenalty)
	}
	if param.EnableThink {
		t.Fatal("enable_thinking alias not applied")
	}
	if param.Model != "local/gemma-4-E2B-it:Q4_0" {
		t.Fatalf("server default model not applied: %q", param.Model)
	}
}

func TestGemma4ReasoningParsingMatchesLlamaWebUIFields(t *testing.T) {
	full := "<|channel>thought\ncheck the answer<channel|>final answer"
	reasoning, content := splitGemma4Response(full)
	if reasoning != "check the answer" || content != "final answer" {
		t.Fatalf("unexpected blocking parse: reasoning=%q content=%q", reasoning, content)
	}

	parser := gemma4StreamParser{}
	pieces := []string{"<|channel>", "thought", "\n", "check", " the answer", "<channel|>", "final", " answer"}
	var gotReasoning, gotContent string
	for _, piece := range pieces {
		chunk, emit := parser.parse(piece)
		if !emit {
			continue
		}
		gotReasoning += chunk.Choices[0].Delta.ReasoningContent
		gotContent += chunk.Choices[0].Delta.Content
	}
	if gotReasoning != "check the answer" || gotContent != "final answer" {
		t.Fatalf("unexpected stream parse: reasoning=%q content=%q", gotReasoning, gotContent)
	}
}
