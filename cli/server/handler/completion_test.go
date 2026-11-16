// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"errors"
	"mime"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/server/types"
)

func TestCompletionPrompt(t *testing.T) {
	tests := []struct {
		name    string
		prompt  openai.CompletionNewParamsPromptUnion
		want    string
		wantErr bool
	}{
		{"string", openai.CompletionNewParamsPromptUnion{OfString: param.NewOpt("<|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>")}, "<|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>", false},
		{"empty string", openai.CompletionNewParamsPromptUnion{OfString: param.NewOpt("")}, "", true},
		{"single-element array", openai.CompletionNewParamsPromptUnion{OfArrayOfStrings: []string{"abc"}}, "abc", false},
		{"multi-element array", openai.CompletionNewParamsPromptUnion{OfArrayOfStrings: []string{"a", "b"}}, "", true},
		{"empty array", openai.CompletionNewParamsPromptUnion{OfArrayOfStrings: []string{}}, "", true},
		{"token array", openai.CompletionNewParamsPromptUnion{OfArrayOfTokens: []int64{1, 2}}, "", true},
		{"omitted", openai.CompletionNewParamsPromptUnion{}, "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := completionPrompt(tt.prompt)
			if (err != nil) != tt.wantErr {
				t.Fatalf("completionPrompt() error = %v, wantErr %v", err, tt.wantErr)
			}
			if got != tt.want {
				t.Errorf("completionPrompt() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCompletionStop(t *testing.T) {
	tests := []struct {
		name string
		stop openai.CompletionNewParamsStopUnion
		want []string
	}{
		{"omitted", openai.CompletionNewParamsStopUnion{}, nil},
		{"string", openai.CompletionNewParamsStopUnion{OfString: param.NewOpt("\n\n")}, []string{"\n\n"}},
		{"empty string", openai.CompletionNewParamsStopUnion{OfString: param.NewOpt("")}, nil},
		{"array", openai.CompletionNewParamsStopUnion{OfStringArray: []string{"<|endoftext|>", "\n"}}, []string{"<|endoftext|>", "\n"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := completionStop(tt.stop); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("completionStop() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestCompletionUnsupported(t *testing.T) {
	tests := []struct {
		name    string
		params  CompletionNewParams
		wantErr bool
	}{
		{"defaults", CompletionNewParams{}, false},
		{"n 1", CompletionNewParams{N: param.NewOpt[int64](1)}, false},
		{"best_of 1", CompletionNewParams{BestOf: param.NewOpt[int64](1)}, false},
		{"empty suffix", CompletionNewParams{Suffix: param.NewOpt("")}, false},
		{"suffix", CompletionNewParams{Suffix: param.NewOpt("tail")}, true},
		{"n 2", CompletionNewParams{N: param.NewOpt[int64](2)}, true},
		{"best_of 2", CompletionNewParams{BestOf: param.NewOpt[int64](2)}, true},
		{"logprobs", CompletionNewParams{Logprobs: param.NewOpt[int64](0)}, true},
		{"logit_bias", CompletionNewParams{LogitBias: map[string]int64{"50256": -100}}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := completionUnsupported(tt.params); (err != nil) != tt.wantErr {
				t.Errorf("completionUnsupported() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCompletionRequestFieldsAfterSDKUpgrade(t *testing.T) {
	req := defaultCompletionRequest()
	body := `{
		"model":"local/example:Q4_K_M", "prompt":["Hello"], "max_tokens":12,
		"suffix":"", "echo":true, "stop":["STOP","END"],
		"temperature":0.3, "top_p":0.9, "seed":31,
		"presence_penalty":0.4, "frequency_penalty":0.5,
		"stream":true, "stream_options":{"include_usage":true},
		"top_k":7, "min_p":0.1, "repetition_penalty":1.2,
		"nctx":2048, "ngl":-1, "compute":"hybrid", "vit_compute":"CPU", "power_mode":"burst"
	}`
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	prompt, err := completionPrompt(req.Prompt)
	if err != nil || prompt != "Hello" || !reflect.DeepEqual(completionStop(req.Stop), []string{"STOP", "END"}) || completionUnsupported(req.CompletionNewParams) != nil {
		t.Fatalf("prompt/stop/support fields changed: %+v, %v", req, err)
	}
	if req.Model != "local/example:Q4_K_M" || req.MaxTokens.Value != 12 || req.Suffix.Value != "" || !req.Echo.Value || !req.Stream || !req.StreamOptions.IncludeUsage.Value {
		t.Fatalf("core completion fields changed: %+v", req)
	}
	if req.Temperature.Value != 0.3 || req.TopP.Value != 0.9 || req.Seed.Value != 31 || req.PresencePenalty.Value != 0.4 || req.FrequencyPenalty.Value != 0.5 || req.TopK != 7 || req.MinP != 0.1 || req.RepetitionPenalty != 1.2 {
		t.Fatalf("sampler fields changed: %+v", req)
	}
	if req.NCtx != 2048 || req.Ngl != -1 || req.Compute != "hybrid" || req.VitCompute != "CPU" || req.PowerMode != "burst" {
		t.Fatalf("device fields changed: %+v", req)
	}
}

func TestCompletionResponseProtocol(t *testing.T) {
	gin.SetMode(gin.TestMode)
	seen := map[string]bool{}
	for _, text := range []string{"hello", ""} {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		profile := geniex_sdk.ProfileData{PromptTokens: 3, TotalPromptTokens: 3, GeneratedTokens: 2, StopReason: "length"}
		writeCompletionResponse(c, text, profile, "test-model")
		meta := checkResponseMetadata(t, w.Body.Bytes(), "cmpl", "test-model")
		if seen[meta.ID] {
			t.Fatalf("reused response ID %q", meta.ID)
		}
		seen[meta.ID] = true
		var got completionResponse
		if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		if w.Code != 200 || got.Object != "text_completion" || len(got.Choices) != 1 || got.Choices[0].Text != text || got.Choices[0].FinishReason != "length" || got.Usage.TotalTokens != 5 {
			t.Fatalf("response: %s", w.Body.Bytes())
		}
	}
}

func TestCompletionSSEProtocol(t *testing.T) {
	gin.SetMode(gin.TestMode)
	seen := map[string]bool{}
	for _, tc := range []struct {
		name   string
		tokens []string
		echo   string
		usage  bool
		err    error
	}{
		{name: "text and usage", tokens: []string{"hello", " world"}, usage: true},
		{name: "echo", tokens: []string{" world"}, echo: "hello"},
		{name: "empty", usage: true},
		{name: "context limit", tokens: []string{"partial"}, err: geniex_sdk.ErrLlmTokenizationContextLength, usage: true},
		{name: "error", err: errors.New("generation failed"), usage: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := &streamRecorder{httptest.NewRecorder(), make(chan bool)}
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest("POST", "/v1/completions", nil)
			ch := make(chan string, len(tc.tokens))
			for _, token := range tc.tokens {
				ch <- token
			}
			close(ch)
			profile := geniex_sdk.ProfileData{PromptTokens: 3, TotalPromptTokens: 3, GeneratedTokens: 2, StopReason: "eos"}
			if tc.err == geniex_sdk.ErrLlmTokenizationContextLength {
				profile.StopReason = "length"
			}
			streamCompletion(c, ch, func() error { return tc.err }, tc.usage, tc.echo, &profile, "test-model")
			result := w.Result()
			mediaType, _, mediaTypeErr := mime.ParseMediaType(result.Header.Get("Content-Type"))
			if mediaTypeErr != nil || mediaType != "text/event-stream" || result.Header.Get("Cache-Control") != "no-cache" {
				t.Fatalf("committed headers: %v", result.Header)
			}
			if result.StatusCode != 200 {
				t.Fatalf("committed status: %d", result.StatusCode)
			}
			var meta types.ResponseMetadata
			var text strings.Builder
			chunks, textChunks, finishes, usage, done, failures := 0, 0, 0, 0, 0, 0
			for line := range strings.SplitSeq(w.Body.String(), "\n") {
				if !strings.HasPrefix(line, "data:") {
					continue
				}
				body := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
				if body == "[DONE]" {
					done++
					continue
				}
				var errorBody struct {
					Error string `json:"error"`
				}
				if err := json.Unmarshal([]byte(body), &errorBody); err != nil {
					t.Fatal(err)
				}
				if errorBody.Error != "" {
					failures++
					continue
				}
				gotMeta := checkResponseMetadata(t, []byte(body), "cmpl", "test-model")
				if chunks == 0 {
					if seen[gotMeta.ID] {
						t.Fatalf("reused response ID %q", gotMeta.ID)
					}
					seen[gotMeta.ID] = true
					meta = gotMeta
				} else if gotMeta != meta {
					t.Fatalf("unstable metadata: %s", body)
				}
				chunks++
				var got completionStreamChunk
				if err := json.Unmarshal([]byte(body), &got); err != nil {
					t.Fatal(err)
				}
				if got.Object != "text_completion" {
					t.Fatalf("object: %s", body)
				}
				if got.Usage != nil {
					usage++
					if len(got.Choices) != 0 || got.Usage.TotalTokens != 5 {
						t.Fatalf("usage: %s", body)
					}
				}
				for _, choice := range got.Choices {
					text.WriteString(choice.Text)
					if choice.FinishReason != nil {
						finishes++
						if *choice.FinishReason != mapFinishReason(profile.StopReason) {
							t.Fatalf("finish: %s", body)
						}
					} else {
						textChunks++
					}
				}
			}
			if text.String() != tc.echo+strings.Join(tc.tokens, "") {
				t.Fatalf("text=%q", text.String())
			}
			if tc.err != nil && tc.err != geniex_sdk.ErrLlmTokenizationContextLength {
				if failures != 1 || done != 0 || usage != 0 || finishes != 0 {
					t.Fatalf("error stream: %s", w.Body.String())
				}
			} else {
				wantUsage := 0
				if tc.usage {
					wantUsage = 1
				}
				wantTextChunks := len(tc.tokens)
				if tc.echo != "" {
					wantTextChunks++
				}
				if failures != 0 || done != 1 || finishes != 1 || usage != wantUsage || textChunks != wantTextChunks {
					t.Fatalf("terminal stream: text=%d finish=%d usage=%d done=%d body=%s", textChunks, finishes, usage, done, w.Body.String())
				}
			}
		})
	}
}
