// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"
)

func TestCompileSystemOne(t *testing.T) {
	var req systemOneRequest
	if err := json.Unmarshal([]byte(`{
		"model":"local/nimble:Q4_K_M", "state":{"text":"Charged twice"},
		"questions":{
			"priority":{"type":"choice","instructions":"Route this","criteria":{"billing":{"about":"Payments"},"bug":null}},
			"refund":{"type":"noul","instructions":{"policy":"Requested refund?"},"criteria":{"true":{"meaning":"Yes"}}},
			"grade":{"type":"score","instructions":"Urgency","criteria":["Low",{"what":"High"}]}
		}
	}`), &req); err != nil {
		t.Fatal(err)
	}
	state, fields, err := compileSystemOne(req)
	if err != nil {
		t.Fatal(err)
	}
	if state != `{"text":"Charged twice"}` || len(fields) != 3 || fields[0].Name != "priority" || fields[1].Name != "refund" || fields[2].Name != "grade" {
		t.Fatalf("state=%q fields=%+v", state, fields)
	}
	schema := systemOneSchema(fields)
	if schema[0].Name != "field_0" || schema[1].Name != "field_1" || fields[0].Name != "priority" {
		t.Fatalf("question ids must not reach the model: %+v", schema)
	}
	if fields[0].Choices[0].Value != "billing" || fields[0].Choices[0].Description != `{"about":"Payments"}` || fields[0].Choices[1].Description != "bug" || fields[1].Choices[0].Value != false || fields[1].Choices[1].Description != `{"meaning":"Yes"}` || fields[2].Choices[1].Code != "B" {
		t.Fatalf("unexpected schema: %+v", fields)
	}
	answer, err := systemOneAnswer(fields[0], []float32{0, 2})
	if err != nil {
		t.Fatal(err)
	}
	choice := answer.(map[string]any)
	if choice["choice"] != "bug" || choice["probabilities"].(map[string]float64)["bug"] < 0.88 {
		t.Fatalf("incorrect answer: %+v", choice)
	}
	answer, err = systemOneAnswer(fields[1], []float32{-1000, 1000})
	if err != nil || answer.(map[string]any)["noul"] != float64(1) {
		t.Fatalf("incorrect noul: %v %v", answer, err)
	}
	answer, err = systemOneAnswer(fields[2], []float32{0, 0})
	if err != nil || answer.(map[string]any)["score"] != 0.5 || answer.(map[string]any)["confidence"] != float64(0) || string(answer.(map[string]any)["legend"].(map[string]json.RawMessage)["1"]) != `{"what":"High"}` {
		t.Fatalf("incorrect score: %v %v", answer, err)
	}
}

func TestSystemOneInvalidRequests(t *testing.T) {
	for _, body := range []string{
		`{}`, `{"model":"nimble","state":"x","questions":{}}`,
		`{"model":"nimble","state":null,"questions":{"q":{"type":"noul","instructions":"test"}}}`,
		`{"model":"nimble","state":"x","questions":{"q":{"type":"choice","instructions":"pick","criteria":{"only":"one"}}}}`,
		`{"model":"nimble","state":"x","questions":{"q":{"type":"score","instructions":"rate","criteria":["ok",42]}}}`,
		`{"model":"nimble","state":"x","questions":{"q":{"type":"noul","instructions":"test","criteria":{"yes":"Yes"}}}}`,
		`{"model":"nimble","state":"x","questions":{"q":{"type":"noul","instructions":"test"},"q":{"type":"noul","instructions":"test"}}}`,
	} {
		var req systemOneRequest
		if err := json.Unmarshal([]byte(body), &req); err != nil {
			t.Fatal(err)
		}
		if _, _, err := compileSystemOne(req); err == nil {
			t.Errorf("accepted %s", body)
		}
	}
	var criteria []string
	for i := 0; i < 27; i++ {
		criteria = append(criteria, fmt.Sprintf(`"%d":"x"`, i))
	}
	var req systemOneRequest
	if err := json.Unmarshal([]byte(`{"model":"nimble","state":"x","questions":{"q":{"type":"choice","instructions":"pick","criteria":{`+strings.Join(criteria, ",")+`}}}}`), &req); err != nil {
		t.Fatal(err)
	}
	if _, _, err := compileSystemOne(req); err == nil {
		t.Fatal("accepted >26 candidates")
	}
	var levels []string
	for i := 0; i < 11; i++ {
		levels = append(levels, fmt.Sprintf(`"level%d"`, i))
	}
	if err := json.Unmarshal([]byte(`{"model":"nimble","state":"x","questions":{"q":{"type":"score","instructions":"rate","criteria":[`+strings.Join(levels, ",")+`]}}}`), &req); err != nil {
		t.Fatal(err)
	}
	if _, _, err := compileSystemOne(req); err == nil {
		t.Fatal("accepted >10 score levels")
	}
	if _, err := systemOneAnswer(systemOneField{Choices: []systemOneChoice{{Value: "a"}, {Value: "b"}}, kind: "choice"}, []float32{float32(math.NaN()), 0}); err == nil {
		t.Fatal("accepted NaN logit")
	}
}
