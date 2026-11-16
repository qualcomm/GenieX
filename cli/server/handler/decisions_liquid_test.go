// Copyright 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/internal/config"
	"github.com/qualcomm/GenieX/cli/internal/decision"
	"github.com/qualcomm/GenieX/cli/server/middleware"
	"github.com/qualcomm/GenieX/cli/server/service"
)

// Model downloads are deliberately opt-in. The map names both the protocol
// being exercised and the exact GGUF, so a CPU check cannot accidentally be
// reported as coverage of a different model or quantization.
func TestLiquidDecisionsCPU(t *testing.T) {
	files := os.Getenv("GENIEX_LIQUID_GGUFS")
	if files == "" {
		t.Skip("set GENIEX_LIQUID_GGUFS to a JSON map of protocol identifiers to GGUF paths")
	}
	var models map[string]string
	if err := json.Unmarshal([]byte(files), &models); err != nil {
		t.Fatal(err)
	}
	if len(models) == 0 {
		t.Fatal("GENIEX_LIQUID_GGUFS must name at least one model")
	}
	root := t.TempDir()
	t.Setenv("GENIEX_DATADIR", root)
	previousDataDir := config.Get().DataDir
	config.Get().DataDir = root
	t.Cleanup(func() { config.Get().DataDir = previousDataDir })
	if err := geniex_sdk.ModelInit(root); err != nil {
		t.Fatal(err)
	}
	if err := geniex_sdk.Init(); err != nil {
		t.Fatal(err)
	}
	defer geniex_sdk.DeInit()
	service.Init()
	defer service.DeInit()
	gin.SetMode(gin.TestMode)
	router := gin.New()
	// The real server shares this lock with its cache sweeper. Omitting it
	// lets a timed eviction destroy the model during an HTTP forward pass.
	router.Use(middleware.GIL)
	router.POST("/v1/decisions", Decisions)
	router.POST("/v1/completions", Completions)
	router.POST("/v1/chat/completions", ChatCompletions)

	for protocol, path := range models {
		t.Run(protocol, func(t *testing.T) {
			name := "local/" + protocol
			if err := geniex_sdk.ModelPull(geniex_sdk.ModelPullInput{
				ModelName: name, Hub: geniex_sdk.HubLocalFS, LocalPath: filepath.Dir(path),
			}); err != nil {
				t.Fatal(err)
			}
			paths, err := geniex_sdk.ModelGetPaths(name)
			if err != nil {
				t.Fatal(err)
			}
			if filepath.Base(paths.ModelPath) != filepath.Base(path) {
				t.Fatalf("resolved %s instead of requested %s; use a directory containing only the requested GGUF", paths.ModelPath, path)
			}
			if protocol != "lfm2-d1" {
				for endpoint, body := range map[string]string{
					"/v1/completions":      `,"prompt":"Hello","max_tokens":1}`,
					"/v1/chat/completions": `,"messages":[{"role":"user","content":"Hello"}],"max_tokens":1}`,
				} {
					recorder := httptest.NewRecorder()
					payload := `{"model":` + quoteJSON(name) + `,"compute":"cpu","nctx":4096` + body
					router.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, endpoint, strings.NewReader(payload)))
					var failure struct {
						Code int32 `json:"code"`
					}
					if err := json.Unmarshal(recorder.Body.Bytes(), &failure); err != nil || recorder.Code < 400 ||
						failure.Code != int32(geniex_sdk.ErrCommonParamNotSupported) {
						t.Fatalf("cache-less generation must fail safely: %s HTTP %d: %s", endpoint, recorder.Code, recorder.Body.String())
					}
				}
			}
			request := `{"model":` + quoteJSON(name) + `,"compute":"cpu","nctx":4096,
				"input":"I was charged twice this month, please refund one of them.","questions":[
				{"name":"refund","type":"predicate","instructions":"Is the customer asking for a refund?"},
				{"name":"team","type":"choice","instructions":"Which team should handle this?","choices":[
				{"value":"billing","description":"Charges, refunds, invoices"},
				{"value":"technical","description":"App or site faults"},
				{"value":"fraud","description":"Suspected unauthorised use"}]},
				{"name":"urgency","type":"score","instructions":"How urgent is this?","levels":[
				{"label":"low","description":"Can wait"},{"label":"medium","description":"Today"},
				{"label":"high","description":"Blocking the customer now"}]}]}`
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/decisions", strings.NewReader(request)))
			if recorder.Code != http.StatusOK {
				t.Fatalf("HTTP %d: %s", recorder.Code, recorder.Body.String())
			}
			var response decisionsResponse
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatal(err)
			}
			if len(response.Answers) != 3 || response.Usage.OutputTokens != 0 || response.Usage.InputTokens <= 0 {
				t.Fatalf("invalid decisions response: %s", recorder.Body.String())
			}
			for i, kind := range []string{"predicate", "choice", "score"} {
				answer := response.Answers[i]
				if answer.Type != kind || answer.Name == nil || *answer.Name != []string{"refund", "team", "urgency"}[i] {
					t.Fatalf("answer mapping/order: %+v", answer)
				}
			}
			predicate := response.Answers[0].Probability
			if predicate == nil || !(*predicate > 0.9 && *predicate <= 1) || response.Answers[1].Choice != "billing" {
				t.Fatalf("refund semantics did not match the native example: %s", recorder.Body.String())
			}
			if len(response.Answers[1].Probabilities) != 3 || len(response.Answers[2].Probabilities) != 3 {
				t.Fatalf("invalid option counts: %s", recorder.Body.String())
			}
			var expected float64
			for i, probability := range response.Answers[2].Probabilities {
				if probability.Value != float64(i) || probability.Label == nil || *probability.Label != []string{"low", "medium", "high"}[i] {
					t.Fatalf("score level mapping: %+v", probability)
				}
				expected += float64(i) * probability.Probability
			}
			if response.Answers[2].Score == nil || math.Abs(*response.Answers[2].Score-expected) > 1e-6 {
				t.Fatalf("expected-score contract: %s", recorder.Body.String())
			}
			// Exercise the VLM entry point without a projector too. Text-only
			// Liquid decisions must use the same native adapter in both handles.
			middleware.GILock.Lock()
			defer middleware.GILock.Unlock()
			vlmSequences, llmSequences := int32(1), int32(1)
			if protocol == "lfm3-d1" {
				vlmSequences, llmSequences = 4, 2
			}
			vlm, err := geniex_sdk.NewVLM(geniex_sdk.VlmCreateInput{
				ModelPath: paths.ModelPath, RuntimeID: geniex_sdk.RuntimeLlamaCpp, DeviceID: "CPU",
				Config: geniex_sdk.ModelConfig{NCtx: 4096, NSeqMax: vlmSequences},
			})
			if err != nil {
				t.Fatal(err)
			}
			defer vlm.Destroy()
			if protocol != "lfm2-d1" {
				generation := &geniex_sdk.GenerationConfig{MaxTokens: 1}
				if _, err := vlm.Generate(geniex_sdk.VlmGenerateInput{PromptUTF8: "Hello", Config: generation}); err != geniex_sdk.ErrCommonParamNotSupported {
					t.Fatalf("cache-less VLM generation must be rejected: %v", err)
				}
				llm, err := geniex_sdk.NewLLM(geniex_sdk.LlmCreateInput{
					ModelPath: paths.ModelPath, RuntimeID: geniex_sdk.RuntimeLlamaCpp, DeviceID: "CPU",
					Config: geniex_sdk.ModelConfig{NCtx: 4096, NSeqMax: llmSequences},
				})
				if err != nil {
					t.Fatal(err)
				}
				defer llm.Destroy()
				if _, err := llm.Generate(geniex_sdk.LlmGenerateInput{PromptUTF8: "Hello", Config: generation}); err != geniex_sdk.ErrCommonParamNotSupported {
					t.Fatalf("cache-less LLM generation must be rejected: %v", err)
				}
				if _, err := llm.ForwardLogits([]int32{0}, true, 0); err != geniex_sdk.ErrCommonParamNotSupported {
					t.Fatalf("cache-less all-position logits must be rejected: %v", err)
				}
				if protocol == "lfm3-d1" {
					if result, err := llm.ForwardLogits([]int32{0}, false, 0); err != nil || result.NRows != 1 {
						t.Fatalf("cache-less last-position logits: %+v, %v", result, err)
					}
				}
			}
			req := decodeDecisionRequest(t, request)
			questions, err := compileDecisionQuestions(req.Questions)
			if err != nil {
				t.Fatal(err)
			}
			modelQuestions := make([]decision.Question, len(questions))
			for i, question := range questions {
				modelQuestions[i] = question.modelQuestion()
			}
			options, err := decision.Questions(modelQuestions)
			if err != nil {
				t.Fatal(err)
			}
			for index, answer := range response.Answers {
				scored, err := vlm.Decide(geniex_sdk.DecideInput{
					State: "I was charged twice this month, please refund one of them.", Questions: options, QuestionIndex: index,
				})
				if err != nil || scored.DecisionType != protocol || len(scored.Probabilities) != len(options[index].Options) || len(scored.Logits) != 0 {
					t.Fatalf("native VLM decision: %+v, %v", scored, err)
				}
				for i, probability := range scored.Probabilities {
					want := *predicate
					if index == 0 && i == 1 {
						want = 1 - want
					} else if index != 0 {
						want = answer.Probabilities[i].Probability
					}
					if math.IsNaN(float64(probability)) || math.Abs(float64(probability)-want) > 1e-6 {
						t.Fatalf("VLM/HTTP probability mismatch: %v vs %v", probability, want)
					}
				}
			}
			for _, media := range []geniex_sdk.DecideInput{
				{ImagePaths: []string{path}},
				{AudioPaths: []string{path}},
				{ImagePaths: []string{path}, AudioPaths: []string{path}},
			} {
				media.State, media.Questions = "state", options
				if _, err := vlm.Decide(media); err != geniex_sdk.ErrCommonParamNotSupported {
					t.Fatalf("Liquid media input was not explicitly rejected: %v", err)
				}
			}
			t.Logf("CPU model %s: %s", filepath.Base(path), recorder.Body.String())
		})
	}
}

func quoteJSON(value string) string {
	text, _ := json.Marshal(value)
	return string(text)
}
