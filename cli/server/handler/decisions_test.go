// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v3"
)

func decodeDecisionRequest(t *testing.T, body string) decisionsRequest {
	t.Helper()
	var req decisionsRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	return req
}

func TestCompileDecisionInput(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "string",
			body: `{"input":"  Charge failed  "}`,
			want: "  Charge failed  ",
		},
		{
			name: "user text parts",
			body: `{"input":[{"role":"user","content":[{"type":"input_text","text":"Charge failed"},{"type":"input_text","text":"Retrying did not help"}]}]}`,
			want: "Charge failed\nRetrying did not help",
		},
		{
			name: "user string message",
			body: `{"input":[{"role":"user","content":"Charge failed"}]}`,
			want: "Charge failed",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := decodeDecisionRequest(t, tt.body)
			got, err := compileDecisionInput(req.Input)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("input = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCompileDecisionInputRejectsImagesClearly(t *testing.T) {
	req := decodeDecisionRequest(t, `{"input":[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,AAAA"}]}]}`)
	_, err := compileDecisionInput(req.Input)
	if err == nil || !strings.Contains(err.Error(), "input_image is unsupported") {
		t.Fatalf("image rejection should explain text-only support, got %v", err)
	}
}

func TestCompileDecisionQuestions(t *testing.T) {
	req := decodeDecisionRequest(t, `{"questions":[
		{"name":"priority","type":"choice","instructions":"Choose a route","choices":[{"value":"billing","description":"Payments"},{"value":false}]},
		{"type":"predicate","instructions":"Is payment blocked?"},
		{"name":"severity","type":"score","instructions":"Rate impact","levels":[{"label":"low","description":"Minor"},{"label":"high"}]}
	]}`)
	questions, err := compileDecisionQuestions(req.Questions)
	if err != nil {
		t.Fatal(err)
	}
	if len(questions) != 3 || questions[0].Name != "priority" || questions[1].Name != "" || questions[2].Name != "severity" {
		t.Fatalf("unexpected question order or names: %+v", questions)
	}
	if questions[0].Choices[0].Value != "billing" || questions[0].Choices[0].Description != "Payments" || questions[0].Choices[1].Value != false {
		t.Fatalf("choice values/descriptions were not preserved: %+v", questions[0].Choices)
	}
	if questions[1].Choices[0].Value != false || questions[1].Choices[1].Value != true {
		t.Fatalf("predicate choices are not false/true: %+v", questions[1].Choices)
	}
	if questions[2].Levels[0].Label != "low" || questions[2].Levels[0].Description != "Minor" || questions[2].Levels[1].Description != "high" {
		t.Fatalf("score levels were not preserved: %+v", questions[2].Levels)
	}
}

func TestCompileDecisionInvalidRequests(t *testing.T) {
	for _, tt := range []struct {
		name         string
		body         string
		invalidInput bool
	}{
		{name: "missing input", body: `{"questions":[{"type":"predicate","instructions":"valid"}]}`, invalidInput: true},
		{name: "empty input", body: `{"input":"  ","questions":[{"type":"predicate","instructions":"valid"}]}`, invalidInput: true},
		{name: "empty user messages", body: `{"input":[],"questions":[{"type":"predicate","instructions":"valid"}]}`, invalidInput: true},
		{name: "non-user role", body: `{"input":[{"role":"assistant","content":"text"}],"questions":[{"type":"predicate","instructions":"valid"}]}`, invalidInput: true},
		{name: "image modality", body: `{"input":[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,AAAA"}]}],"questions":[{"type":"predicate","instructions":"valid"}]}`, invalidInput: true},
		{name: "invalid input part", body: `{"input":[{"role":"user","content":[{"type":"input_audio","data":"AAAA"}]}],"questions":[{"type":"predicate","instructions":"valid"}]}`, invalidInput: true},
		{name: "no questions", body: `{"input":"text","questions":[]}`},
		{name: "unknown question type", body: `{"input":"text","questions":[{"type":"unsupported","instructions":"valid"}]}`},

		{name: "missing instructions", body: `{"input":"text","questions":[{"type":"predicate"}]}`},
		{name: "duplicate names", body: `{"input":"text","questions":[{"name":"q","type":"predicate","instructions":"one"},{"name":"q","type":"predicate","instructions":"two"}]}`},
		{name: "choice too small", body: `{"input":"text","questions":[{"type":"choice","instructions":"valid","choices":[{"value":"only"}]}]}`},
		{name: "choice number value", body: `{"input":"text","questions":[{"type":"choice","instructions":"valid","choices":[{"value":1},{"value":"two"}]}]}`},
		{name: "duplicate choice values", body: `{"input":"text","questions":[{"type":"choice","instructions":"valid","choices":[{"value":"a"},{"value":"a"}]}]}`},
		{name: "null description", body: `{"input":"text","questions":[{"type":"choice","instructions":"valid","choices":[{"value":"one","description":null},{"value":"two"}]}]}`},
		{name: "score too small", body: `{"input":"text","questions":[{"type":"score","instructions":"valid","levels":[{"label":"only"}]}]}`},
		{name: "empty score label", body: `{"input":"text","questions":[{"type":"score","instructions":"valid","levels":[{"label":" "},{"label":"high"}]}]}`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var req decisionsRequest
			if err := json.Unmarshal([]byte(tt.body), &req); err != nil {
				if tt.name == "unknown question type" || (tt.invalidInput && (tt.name == "invalid input part" || tt.name == "image modality")) {
					return // The SDK union decoder rejects unsupported variants at binding time.
				}
				t.Fatalf("request should reach validation: %v", err)
			}
			if tt.invalidInput {
				if _, err := compileDecisionInput(req.Input); err == nil {
					t.Fatal("accepted invalid input")
				}
				if len(req.Questions) > 0 {
					if _, err := compileDecisionQuestions(req.Questions); err != nil {
						t.Fatalf("valid question was rejected: %v", err)
					}
				}
				return
			}
			if _, err := compileDecisionInput(req.Input); err != nil {
				t.Fatalf("valid input was rejected: %v", err)
			}
			if _, err := compileDecisionQuestions(req.Questions); err == nil {
				t.Fatal("accepted invalid questions")
			}
		})
	}

	for _, tt := range []struct {
		name  string
		count int
		kind  string
	}{
		{name: "too many questions", count: 65, kind: "predicate"},
		{name: "too many choices", count: 27, kind: "choice"},
		{name: "too many levels", count: 11, kind: "score"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var questions string
			switch tt.kind {
			case "predicate":
				for i := 0; i < tt.count; i++ {
					questions += `{"type":"predicate","instructions":"test"},`
				}
			case "choice":
				for i := 0; i < tt.count; i++ {
					questions += fmt.Sprintf(`{"value":"%d"},`, i)
				}
				questions = `{"type":"choice","instructions":"test","choices":[` + strings.TrimSuffix(questions, ",") + `]}`
			case "score":
				for i := 0; i < tt.count; i++ {
					questions += fmt.Sprintf(`{"label":"%d"},`, i)
				}
				questions = `{"type":"score","instructions":"test","levels":[` + strings.TrimSuffix(questions, ",") + `]}`
			}
			if tt.kind == "predicate" {
				questions = strings.TrimSuffix(questions, ",")
			}
			body := `{"questions":[` + questions + `]}`
			var req decisionsRequest
			if err := json.Unmarshal([]byte(body), &req); err != nil {
				t.Fatal(err)
			}
			if _, err := compileDecisionQuestions(req.Questions); err == nil {
				t.Fatal("accepted limit violation")
			}
		})
	}
}

func TestDecisionAnswerJSONShape(t *testing.T) {
	questions := []decisionQuestion{
		{Type: "predicate", Name: "urgent", Choices: []decisionChoice{{Value: false}, {Value: true}}},
		{Type: "choice", Name: "label", Choices: []decisionChoice{{Value: "billing"}, {Value: "bug"}}},
		{Type: "score", Name: "severity", Levels: []decisionLevel{{Label: "low"}, {Label: "high"}}},
	}
	logits := [][]float32{{0, float32(math.Log(3))}, {0, 2}, {0, 0}}
	answers := make([]decisionAnswer, 0, len(questions))
	for i, question := range questions {
		answer, err := decisionAnswerFor(question, logits[i])
		if err != nil {
			t.Fatal(err)
		}
		answers = append(answers, answer)
	}
	if answers[0].Probability == nil || math.Abs(*answers[0].Probability-0.75) > 1e-6 {
		t.Fatalf("predicate probability = %v, want 0.75", answers[0].Probability)
	}
	if answers[1].Choice != "bug" || len(answers[1].Probabilities) != 2 || answers[1].Confidence == nil {
		t.Fatalf("incorrect choice answer: %+v", answers[1])
	}
	if answers[2].Score == nil || *answers[2].Score != 0.5 || answers[2].Probabilities[0].Value != 0 || answers[2].Probabilities[0].Label == nil || *answers[2].Probabilities[0].Label != "low" {
		t.Fatalf("incorrect score answer: %+v", answers[2])
	}

	body, err := json.Marshal(decisionsResponse{
		Answers: answers,
		Model:   "local/nimble:Q4_K_M",
		Usage: openai.DecisionUsage{
			InputTokens:         12,
			InputTokensDetails:  openai.DecisionUsageInputTokensDetails{},
			OutputTokensDetails: openai.DecisionUsageOutputTokensDetails{},
			TotalTokens:         12,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got["model"] != "local/nimble:Q4_K_M" {
		t.Fatalf("top-level JSON shape is wrong: %s", body)
	}
	gotAnswers := got["answers"].([]any)
	if len(gotAnswers) != 3 || gotAnswers[0].(map[string]any)["name"] != "urgent" || gotAnswers[1].(map[string]any)["name"] != "label" || gotAnswers[2].(map[string]any)["name"] != "severity" {
		t.Fatalf("answers are not an ordered array: %s", body)
	}
	predicate := gotAnswers[0].(map[string]any)
	choice := gotAnswers[1].(map[string]any)
	score := gotAnswers[2].(map[string]any)
	if len(predicate) != 3 || predicate["type"] != "predicate" || math.Abs(predicate["probability"].(float64)-0.75) > 1e-6 {
		t.Fatalf("predicate JSON shape is wrong: %#v", predicate)
	}
	if len(choice) != 5 || choice["type"] != "choice" || choice["choice"] != "bug" || len(choice["probabilities"].([]any)) != 2 {
		t.Fatalf("choice JSON shape is wrong: %#v", choice)
	}
	scoreProbabilities := score["probabilities"].([]any)
	if len(score) != 5 || score["type"] != "score" || score["score"] != 0.5 || scoreProbabilities[0].(map[string]any)["value"] != float64(0) || scoreProbabilities[0].(map[string]any)["label"] != "low" {
		t.Fatalf("score JSON shape is wrong: %#v", score)
	}
	usage := got["usage"].(map[string]any)
	inputDetails := usage["input_tokens_details"].(map[string]any)
	outputDetails := usage["output_tokens_details"].(map[string]any)
	if len(usage) != 5 || usage["input_tokens"] != float64(12) || usage["output_tokens"] != float64(0) || usage["total_tokens"] != float64(12) || len(inputDetails) != 2 || len(outputDetails) != 1 {
		t.Fatalf("usage JSON shape is wrong: %#v", usage)
	}
	var sdkDecision openai.Decision
	if err := json.Unmarshal(body, &sdkDecision); err != nil {
		t.Fatalf("OpenAI Go SDK cannot decode response: %v", err)
	}
	if math.Abs(sdkDecision.Answers[0].AsPredicate().Probability-0.75) > 1e-6 || sdkDecision.Answers[1].AsChoice().Choice.AsString() != "bug" || sdkDecision.Answers[2].AsScore().Score != 0.5 {
		t.Fatalf("OpenAI Go SDK decoded the answers incorrectly: %+v", sdkDecision.Answers)
	}

	unnamedAnswer, err := decisionAnswerFor(decisionQuestion{
		Type: "predicate", Choices: []decisionChoice{{Value: false}, {Value: true}},
	}, []float32{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	unnamedJSON, err := json.Marshal(unnamedAnswer)
	if err != nil {
		t.Fatal(err)
	}
	var unnamedShape map[string]any
	if err := json.Unmarshal(unnamedJSON, &unnamedShape); err != nil {
		t.Fatal(err)
	}
	if name, exists := unnamedShape["name"]; !exists || name != nil {
		t.Fatalf("unnamed question must return null name: %s", unnamedJSON)
	}

	booleanAnswer, err := decisionAnswerFor(decisionQuestion{
		Type: "choice", Name: "enabled", Choices: []decisionChoice{{Value: false}, {Value: true}},
	}, []float32{1, 0})
	if err != nil {
		t.Fatal(err)
	}
	booleanJSON, err := json.Marshal(booleanAnswer)
	if err != nil {
		t.Fatal(err)
	}
	var booleanShape map[string]any
	if err := json.Unmarshal(booleanJSON, &booleanShape); err != nil {
		t.Fatal(err)
	}
	if value, exists := booleanShape["choice"]; !exists || value != false {
		t.Fatalf("false-valued choice was omitted or changed: %s", booleanJSON)
	}
}

func TestDecisionAnswerRejectsInvalidLogits(t *testing.T) {
	question := decisionQuestion{Type: "choice", Name: "route", Choices: []decisionChoice{{Value: "a"}, {Value: "b"}}}
	for _, logits := range [][]float32{{0}, {float32(math.NaN()), 0}, {float32(math.Inf(1)), 0}} {
		if _, err := decisionAnswerFor(question, logits); err == nil {
			t.Errorf("accepted invalid logits %v", logits)
		}
	}
}

func TestDecisionAllRequestFields(t *testing.T) {
	req := decodeDecisionRequest(t, `{
		"model":"local/example:Q4_K_M", "nctx":2048, "ngl":-1,
		"compute":"hybrid", "power_mode":"burst",
		"input":[
			{"role":"user", "type":"message", "content":"First"},
			{"role":"user", "content":[{"type":"input_text","text":"Second"},{"type":"input_text","text":"Third"}]}
		],
		"questions":[
			{"type":"predicate", "instructions":"Is it true?"},
			{"type":"choice", "name":"route", "instructions":"Select one", "choices":[{"value":false,"description":"No"},{"value":"false","description":"Literal false"}]},
			{"type":"score", "name":"grade", "instructions":"Rate it", "levels":[{"label":"low","description":"Minor"},{"label":"high"}]}
		]
	}`)
	if req.Model != "local/example:Q4_K_M" || req.NCtx != 2048 || req.Ngl != -1 || req.Compute != "hybrid" || req.PowerMode != "burst" {
		t.Fatalf("request fields changed during SDK decoding: %+v", req)
	}
	input, err := compileDecisionInput(req.Input)
	if err != nil || input != "First\nSecond\nThird" {
		t.Fatalf("input = %q, err = %v", input, err)
	}
	questions, err := compileDecisionQuestions(req.Questions)
	if err != nil || len(questions) != 3 {
		t.Fatalf("questions = %+v, err = %v", questions, err)
	}
	if questions[0].Name != "" || questions[1].Choices[0].Value != false || questions[1].Choices[1].Value != "false" || questions[1].Choices[0].Description != "No" || questions[2].Levels[0].Description != "Minor" {
		t.Fatalf("question fields changed during compilation: %+v", questions)
	}
	for i, wantCode := range []string{"A", "B"} {
		payload, err := decisionPromptFor(input, questions[i+1])
		if err != nil || !strings.Contains(string(payload), `"code":"`+wantCode+`"`) {
			t.Fatalf("prompt did not preserve ordered candidate code %s: %s, %v", wantCode, payload, err)
		}
	}
}

func TestDecisionMaximumSupportedCounts(t *testing.T) {
	parts := make([]string, 64)
	for i := range parts {
		parts[i] = `{"type":"predicate","instructions":"test"}`
	}
	fields := decodeDecisionRequest(t, `{"questions":[`+strings.Join(parts, ",")+`]}`).Questions
	if got, err := compileDecisionQuestions(fields); err != nil || len(got) != 64 {
		t.Fatalf("64 questions rejected: %d, %v", len(got), err)
	}
	for _, tc := range []struct {
		kind, key, item, lastCode string
		count                     int
	}{
		{"choice", "choices", `{"value":"%d"}`, "Z", 26},
		{"score", "levels", `{"label":"%d"}`, "J", 10},
	} {
		items := make([]string, tc.count)
		for i := range items {
			items[i] = fmt.Sprintf(tc.item, i)
		}
		body := fmt.Sprintf(`{"questions":[{"type":%q,"instructions":"test",%q:[%s]}]}`, tc.kind, tc.key, strings.Join(items, ","))
		questions, err := compileDecisionQuestions(decodeDecisionRequest(t, body).Questions)
		if err != nil {
			t.Fatalf("%d %s entries rejected: %v", tc.count, tc.kind, err)
		}
		prompt, err := decisionPromptFor("text", questions[0])
		if err != nil || !strings.Contains(string(prompt), `"code":"`+tc.lastCode+`"`) {
			t.Fatalf("last candidate code for %s = %s, err = %v", tc.kind, prompt, err)
		}
	}
}

func TestDecisionTypedValuesAndExtremeProbabilities(t *testing.T) {
	choice := decisionQuestion{Type: "choice", Choices: []decisionChoice{{Value: false}, {Value: "false"}}}
	answer, err := decisionAnswerFor(choice, []float32{1000, -1000})
	if err != nil || answer.Choice != false || answer.Probabilities[0].Probability != 1 || answer.Probabilities[1].Probability != 0 || *answer.Confidence != 1 {
		t.Fatalf("typed false choice or softmax failed: %+v, %v", answer, err)
	}
	for _, tc := range []struct {
		logits []float32
		want   float64
	}{{[]float32{1000, -1000}, 0}, {[]float32{-1000, 1000}, 1}} {
		predicate := decisionQuestion{Type: "predicate", Choices: []decisionChoice{{Value: false}, {Value: true}}}
		answer, err := decisionAnswerFor(predicate, tc.logits)
		if err != nil || *answer.Probability != tc.want {
			t.Fatalf("predicate probability = %+v, err = %v; want %v", answer, err, tc.want)
		}
	}
}

func TestDecisionsRejectInvalidHTTPRequests(t *testing.T) {
	router := gin.New()
	router.POST("/decisions", Decisions)
	for _, tc := range []struct {
		name, body string
		want       int
	}{
		{"missing model", `{"input":"text","questions":[{"type":"predicate","instructions":"test"}]}`, http.StatusUnprocessableEntity},
		{"null input", `{"model":"model","input":null,"questions":[{"type":"predicate","instructions":"test"}]}`, http.StatusUnprocessableEntity},
		{"unsupported image", `{"model":"model","input":[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,AAAA"}]}],"questions":[{"type":"predicate","instructions":"test"}]}`, http.StatusUnprocessableEntity},
		{"unknown question", `{"model":"model","input":"text","questions":[{"type":"unknown","instructions":"test"}]}`, http.StatusUnprocessableEntity},
		{"null name", `{"model":"model","input":"text","questions":[{"type":"predicate","name":null,"instructions":"test"}]}`, http.StatusUnprocessableEntity},
		{"invalid JSON", `{`, http.StatusUnprocessableEntity},
		{"oversize", `{"padding":"` + strings.Repeat("x", 1<<20) + `"}`, http.StatusRequestEntityTooLarge},
	} {
		t.Run(tc.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, "/decisions", strings.NewReader(tc.body))
			request.Header.Set("Content-Type", "application/json")
			router.ServeHTTP(recorder, request)
			if recorder.Code != tc.want {
				t.Fatalf("HTTP %d, want %d: %s", recorder.Code, tc.want, recorder.Body.String())
			}
		})
	}
}
