// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package eval

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParsePack(t *testing.T) {
	pack, err := parsePack([]byte(`{
		"name": "p",
		"tasks": [
			{"type": "multiple_choice", "prompt": "q?", "choices": ["x", "y"], "answer": "b"},
			{"id": "t2", "type": "exact_match", "prompt": "q2?", "expected": "yes"},
			{"id": "t3", "type": "contains", "prompt": "q3?", "expected": "part"}
		]
	}`))
	if err != nil {
		t.Fatalf("parsePack: %v", err)
	}
	if pack.Tasks[0].ID != "task-1" {
		t.Errorf("auto id = %q, want task-1", pack.Tasks[0].ID)
	}
	if pack.Tasks[0].Answer != "B" {
		t.Errorf("answer not upcased: %q", pack.Tasks[0].Answer)
	}

	bad := map[string]string{
		"not json":         `{`,
		"no name":          `{"tasks": [{"type": "contains", "prompt": "p", "expected": "e"}]}`,
		"no tasks":         `{"name": "p", "tasks": []}`,
		"unknown type":     `{"name": "p", "tasks": [{"type": "essay", "prompt": "p"}]}`,
		"missing prompt":   `{"name": "p", "tasks": [{"type": "contains", "expected": "e"}]}`,
		"missing expected": `{"name": "p", "tasks": [{"type": "exact_match", "prompt": "p"}]}`,
		"one choice":       `{"name": "p", "tasks": [{"type": "multiple_choice", "prompt": "p", "choices": ["x"], "answer": "A"}]}`,
		"answer range":     `{"name": "p", "tasks": [{"type": "multiple_choice", "prompt": "p", "choices": ["x", "y"], "answer": "C"}]}`,
		"duplicate ids":    `{"name": "p", "tasks": [{"id": "t", "type": "contains", "prompt": "p", "expected": "e"}, {"id": "t", "type": "contains", "prompt": "p", "expected": "e"}]}`,
	}
	for name, data := range bad {
		if _, err := parsePack([]byte(data)); err == nil {
			t.Errorf("%s: expected error, got none", name)
		}
	}
}

func TestResolve(t *testing.T) {
	path := filepath.Join(t.TempDir(), "custom.json")
	data := `{"name": "custom", "tasks": [{"type": "contains", "prompt": "p", "expected": "e"}]}`
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	pack, err := Resolve(path)
	if err != nil || pack.Name != "custom" {
		t.Fatalf("Resolve(file) = %v, %v", pack, err)
	}
	for _, name := range BuiltinNames() {
		if _, err := Resolve(name); err != nil {
			t.Errorf("built-in %q does not parse: %v", name, err)
		}
	}
	if _, err := Resolve("no-such-pack"); err == nil || !strings.Contains(err.Error(), "basic") {
		t.Fatalf("Resolve(unknown) should list built-ins, got: %v", err)
	}
}

func TestRenderPrompt(t *testing.T) {
	mc := Task{Type: TaskMultipleChoice, Prompt: "Pick one.", Choices: []string{"first", "second"}}
	got := RenderPrompt(mc)
	for _, want := range []string{"Pick one.", "A. first", "B. second", "only the letter"} {
		if !strings.Contains(got, want) {
			t.Errorf("RenderPrompt missing %q in:\n%s", want, got)
		}
	}
	plain := Task{Type: TaskExactMatch, Prompt: "Just this."}
	if RenderPrompt(plain) != "Just this." {
		t.Errorf("non-MC prompt should pass through, got %q", RenderPrompt(plain))
	}
}

func TestScore(t *testing.T) {
	mc := Task{Type: TaskMultipleChoice, Prompt: "q", Choices: []string{"alpha", "beta", "gamma", "delta"}, Answer: "B"}
	correct := []string{
		"B", "b", "(B)", "B.", "**B**", "B) beta",
		"Answer: B", "The answer is B.",
		"<think>hmm A or B... let me think</think>B",
		"beta", "It is beta.",
	}
	for _, out := range correct {
		if !Score(mc, out) {
			t.Errorf("Score(%q) = false, want true", out)
		}
	}
	wrong := []string{
		"A", "C", "alpha", "", "no idea", "E",
		"The answer is a fruit.", // article "a" is not choice A
		"alpha or beta",          // ambiguous
		"<think>the answer is B", // unterminated think block
	}
	for _, out := range wrong {
		if Score(mc, out) {
			t.Errorf("Score(%q) = true, want false", out)
		}
	}

	exact := Task{Type: TaskExactMatch, Prompt: "q", Expected: "Paris"}
	for _, out := range []string{"Paris", "paris", " Paris. ", "PARIS!"} {
		if !Score(exact, out) {
			t.Errorf("Score(%q) = false, want true", out)
		}
	}
	for _, out := range []string{"The capital is Paris", "Pariss", ""} {
		if Score(exact, out) {
			t.Errorf("Score(%q) = true, want false", out)
		}
	}

	contains := Task{Type: TaskContains, Prompt: "q", Expected: "children"}
	if !Score(contains, "The plural of child is CHILDREN.") {
		t.Error("case-insensitive contains failed")
	}
	if Score(contains, "The plural is childs.") {
		t.Error("contains matched wrong text")
	}
}

func TestReports(t *testing.T) {
	pack := &Pack{Name: "p", Tasks: []Task{{ID: "a"}, {ID: "b"}, {ID: "c"}, {ID: "d"}}}
	report := &ModelReport{
		Model: "model-x",
		Results: []Result{
			{TaskID: "a", Category: "math", Correct: true},
			{TaskID: "b", Category: "math", Correct: false},
			{TaskID: "c", Category: "lang", Correct: true},
			{TaskID: "d", Category: "lang", Correct: true},
		},
		DecodeTokens: 100,
		DecodeTimeUs: 2_000_000,
	}
	if report.Correct() != 3 || report.Accuracy() != 0.75 || report.TokPerSec() != 50 {
		t.Errorf("got correct=%d accuracy=%f tok/s=%f", report.Correct(), report.Accuracy(), report.TokPerSec())
	}
	empty := &ModelReport{Model: "empty"}
	if empty.Accuracy() != 0 || empty.TokPerSec() != 0 {
		t.Error("empty report should report zeros, not NaN/Inf")
	}

	table := RenderTable(pack, []*ModelReport{report})
	for _, want := range []string{"model-x", "75.0%", "3/4", "50.0", "MATH", "LANG", "2/2"} {
		if !strings.Contains(table, want) {
			t.Errorf("table missing %q:\n%s", want, table)
		}
	}

	path := filepath.Join(t.TempDir(), "out.json")
	if err := WriteJSON(path, pack, []*ModelReport{report}); err != nil {
		t.Fatalf("WriteJSON: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Eval      string         `json:"eval"`
		TaskCount int            `json:"task_count"`
		Models    []*ModelReport `json:"models"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatalf("output is not valid JSON: %v", err)
	}
	if doc.Eval != "p" || doc.TaskCount != 4 || len(doc.Models) != 1 {
		t.Errorf("round-trip mismatch: %+v", doc)
	}
}
