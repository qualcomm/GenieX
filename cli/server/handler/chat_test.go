// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/spf13/viper"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

func TestWriteChatExecutionErrorStatus(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want int
	}{
		{name: "wrapped SDK invalid input", err: fmt.Errorf("media rejected: %w", geniex_sdk.ErrCommonInvalidInput), want: http.StatusBadRequest},
		{name: "image load failure", err: fmt.Errorf("media rejected: %w", geniex_sdk.ErrVlmImageLoad), want: http.StatusBadRequest},
		{name: "audio load failure", err: fmt.Errorf("media rejected: %w", geniex_sdk.ErrVlmAudioLoad), want: http.StatusBadRequest},
		{name: "SDK generation failure", err: geniex_sdk.ErrVlmGenerationFailed, want: http.StatusInternalServerError},
		{name: "non-SDK generation failure", err: fmt.Errorf("generation failed"), want: http.StatusInternalServerError},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gin.SetMode(gin.TestMode)
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest("POST", "/v1/chat/completions", nil)
			writeChatExecutionError(c, tc.err)
			if w.Code != tc.want {
				t.Errorf("HTTP status = %d, want %d", w.Code, tc.want)
			}
		})
	}
}

func TestReasoningFormatDefault(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{"omitted uses deepseek", `{}`, "deepseek"},
		{"none keeps reasoning inline", `{"reasoning_format":"none"}`, "none"},
		{"auto separates reasoning", `{"reasoning_format":"auto"}`, "auto"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := defaultChatCompletionRequest()
			if err := json.Unmarshal([]byte(tc.body), &p); err != nil {
				t.Fatalf("unmarshal %s: %v", tc.body, err)
			}
			if p.ReasoningFormat != tc.want {
				t.Errorf("reasoning format = %q, want %q", p.ReasoningFormat, tc.want)
			}
		})
	}
}

func TestParseTools(t *testing.T) {
	for _, tools := range []string{
		"", "[]",
		`[{"type":"function","function":{"name":"set","parameters":{"type":"object","properties":{"count":{"type":"integer"},"label":{"type":"string"}}}}}]`,
		`[{"type":"function","function":{"name":"set","parameters":{"type":"object","properties":{"count":{"type":"integer"},"disabled":false}}}},
		{"type":"function","function":{"name":"other","parameters":{"type":"object","properties":{"disabled":false}}}}]`,
	} {
		t.Run(tools, func(t *testing.T) {
			body := `{"model":"m","messages":[{"role":"user","content":"test"}]`
			if tools != "" {
				body += `,"tools":` + tools
			}
			_, request := bindRequest(t, body+"}")
			enabled, encoded, err := parseTools(request)
			if err != nil {
				t.Fatal(err)
			}
			if tools == "" || tools == "[]" {
				if enabled || encoded != "" {
					t.Errorf("empty tools=(%v, %q), want disabled", enabled, encoded)
				}
				return
			}
			var got, want any
			if err := json.Unmarshal([]byte(encoded), &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal([]byte(tools), &want); err != nil {
				t.Fatal(err)
			}
			if !enabled || !reflect.DeepEqual(got, want) {
				t.Errorf("tools=(%v, %s), want unchanged %s", enabled, encoded, tools)
			}
		})
	}
}

func TestChatBindingUsesServerModel(t *testing.T) {
	t.Setenv("GENIEX_MODEL", "")
	viper.Set("model", "local/default:Q4_0")
	t.Cleanup(func() { viper.Set("model", nil) })

	_, req := bindRequest(t, `{"messages":[{"role":"user","content":"Hello"}]}`)
	if req.Model != "local/default:Q4_0" {
		t.Errorf("chat model = %q, want server default", req.Model)
	}
}

func TestChatBindingUsesServerThinkDefault(t *testing.T) {
	t.Setenv("GENIEX_THINK", "")
	const omitted = `{"messages":[{"role":"user","content":"Hello"}]}`
	_, req := bindRequest(t, omitted)
	if !req.EnableThink {
		t.Error("enable_think should default to true")
	}

	viper.Set("think", false)
	t.Cleanup(func() { viper.Set("think", nil) })

	_, req = bindRequest(t, omitted)
	if req.EnableThink {
		t.Error("omitted enable_think should use the server default")
	}
	_, req = bindRequest(t, `{"messages":[{"role":"user","content":"Hello"}],"enable_think":true}`)
	if !req.EnableThink {
		t.Error("request enable_think should override the server default")
	}
}

func TestServeModelDefaultRequests(t *testing.T) {
	t.Setenv("GENIEX_MODEL", "")
	viper.Set("model", "local/default:Q4_0")
	t.Cleanup(func() { viper.Set("model", nil) })

	chat := defaultChatCompletionRequest()
	completion := defaultCompletionRequest()
	decisions := defaultDecisionsRequest()
	logits := defaultForwardLogitsRequest()
	for _, tc := range []struct {
		name  string
		req   any
		model func() string
	}{
		{"chat", &chat, func() string { return chat.Model }},
		{"completions", &completion, func() string { return string(completion.Model) }},
		{"decisions", &decisions, func() string { return decisions.Model }},
		{"logits", &logits, func() string { return logits.Model }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := json.Unmarshal([]byte(`{}`), tc.req); err != nil {
				t.Fatal(err)
			}
			if got := tc.model(); got != "local/default:Q4_0" {
				t.Errorf("omitted model = %q, want server default", got)
			}
			if err := json.Unmarshal([]byte(`{"model":"local/request:Q8_0"}`), tc.req); err != nil {
				t.Fatal(err)
			}
			if got := tc.model(); got != "local/request:Q8_0" {
				t.Errorf("request model = %q, want explicit model", got)
			}
			if err := json.Unmarshal([]byte(`{"model":""}`), tc.req); err != nil {
				t.Fatal(err)
			}
			if got := tc.model(); got != "" {
				t.Errorf("empty explicit model = %q, want empty", got)
			}
		})
	}
}

func TestSpeculativeDecodingDefaultsAndRequestOverrides(t *testing.T) {
	for _, name := range []string{
		"GENIEX_SPEC_TYPE",
		"GENIEX_DRAFT_MODEL",
		"GENIEX_DRAFT_TOKENS",
		"GENIEX_DRAFT_MIN",
		"GENIEX_DRAFT_P_MIN",
	} {
		t.Setenv(name, "")
	}

	t.Run("unset defaults leave speculation off", func(t *testing.T) {
		p := defaultChatCompletionRequest()
		if p.SpecType != "" || p.SpecDraftModel != "" {
			t.Errorf("speculation enabled without defaults: type=%q draft_model=%q", p.SpecType, p.SpecDraftModel)
		}
		if p.SpecNMax != 0 || p.SpecNMin != 0 || p.SpecPMin != 0 {
			t.Errorf("speculative decoding knobs without defaults = (%d, %d, %g), want zero values", p.SpecNMax, p.SpecNMin, p.SpecPMin)
		}
	})

	t.Run("request fields override defaults including empty and zero", func(t *testing.T) {
		for key, value := range map[string]any{
			"spectype":    "draft-mtp",
			"draftmodel":  "draft.gguf",
			"drafttokens": int32(6),
			"draftmin":    int32(2),
			"draftpmin":   float32(0.65),
		} {
			viper.Set(key, value)
			t.Cleanup(func() { viper.Set(key, nil) })
		}
		p := defaultChatCompletionRequest()
		if p.SpecType != "draft-mtp" || p.SpecDraftModel != "draft.gguf" || p.SpecNMax != 6 || p.SpecNMin != 2 || p.SpecPMin != 0.65 {
			t.Fatalf("chat MTP defaults = %+v, want configured defaults", p)
		}
		partial := defaultChatCompletionRequest()
		if err := json.Unmarshal([]byte(`{"spec_n_max":1}`), &partial); err != nil {
			t.Fatal(err)
		}
		if partial.SpecType != "draft-mtp" || partial.SpecDraftModel != "draft.gguf" || partial.SpecNMax != 1 || partial.SpecNMin != 2 || partial.SpecPMin != 0.65 {
			t.Errorf("partial request override = %+v, want only spec_n_max changed", partial)
		}
		body := `{"spec_type":"","spec_draft_model":"","spec_n_max":0,"spec_n_min":0,"spec_p_min":0}`
		if err := json.Unmarshal([]byte(body), &p); err != nil {
			t.Fatalf("unmarshal %s: %v", body, err)
		}
		if p.SpecType != "" || p.SpecDraftModel != "" {
			t.Errorf("empty request strings did not override defaults: type=%q draft_model=%q", p.SpecType, p.SpecDraftModel)
		}
		if p.SpecNMax != 0 || p.SpecNMin != 0 || p.SpecPMin != 0 {
			t.Errorf("zero request values did not override defaults: (%d, %d, %g)", p.SpecNMax, p.SpecNMin, p.SpecPMin)
		}
	})
}

// Mirrors the alias collapse ChatCompletions applies after binding.
func TestMaxCompletionTokensAlias(t *testing.T) {
	cases := []struct {
		name string
		body string
		want int64
	}{
		{"no cap in body falls back to default", `{}`, 2048},
		{"max_completion_tokens honoured", `{"max_completion_tokens":512}`, 512},
		{"deprecated max_tokens honoured", `{"max_tokens":512}`, 512},
		{"max_completion_tokens wins over max_tokens", `{"max_tokens":512,"max_completion_tokens":128}`, 128},
		{"explicit null falls back to default", `{"max_tokens":null}`, 2048},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := defaultChatCompletionRequest()
			if err := json.Unmarshal([]byte(tc.body), &p); err != nil {
				t.Fatalf("unmarshal %s: %v", tc.body, err)
			}
			if got := p.MaxCompletionTokens.Or(p.MaxTokens.Value); got != tc.want {
				t.Errorf("cap = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestChatGenerationConfig(t *testing.T) {
	cases := []struct {
		name string
		body string
		want []string
	}{
		{"omitted", `{}`, nil},
		{"null", `{"stop":null}`, nil},
		{"string", `{"stop":"<|endoftext|>"}`, []string{"<|endoftext|>"}},
		{"empty string", `{"stop":""}`, nil},
		{"array", `{"stop":["STOP","END"]}`, []string{"STOP", "END"}},
		{"empty array", `{"stop":[]}`, []string{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, req := bindRequest(t, tc.body)
			if got := chatGenerationConfig(req, nil, nil, nil).Stop; !reflect.DeepEqual(got, tc.want) {
				t.Errorf("Stop = %#v, want %#v", got, tc.want)
			}
		})
	}
}

func TestChatGenerationConfigForwardsFields(t *testing.T) {
	_, req := bindRequest(t, `{"max_completion_tokens":37,"stop":["END"]}`)
	sampler := &geniex_sdk.SamplerConfig{TopK: 9}
	images, audios := []string{"image.png"}, []string{"audio.wav"}
	got := chatGenerationConfig(req, sampler, images, audios)
	if got.MaxTokens != 37 || got.SamplerConfig != sampler ||
		!reflect.DeepEqual(got.ImagePaths, images) || !reflect.DeepEqual(got.AudioPaths, audios) {
		t.Errorf("config = %+v", got)
	}
}
