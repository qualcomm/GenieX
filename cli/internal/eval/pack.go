// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// Package eval implements the harness behind `geniex eval`: loading eval
// packs, scoring model output, and rendering comparison reports.
package eval

import (
	"embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	TaskMultipleChoice = "multiple_choice"
	TaskExactMatch     = "exact_match"
	TaskContains       = "contains"
)

type Task struct {
	ID       string   `json:"id"`
	Type     string   `json:"type"`
	Category string   `json:"category,omitempty"`
	Prompt   string   `json:"prompt"`
	Choices  []string `json:"choices,omitempty"`  // multiple_choice
	Answer   string   `json:"answer,omitempty"`   // multiple_choice: correct letter
	Expected string   `json:"expected,omitempty"` // exact_match / contains
}

type Pack struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Tasks       []Task `json:"tasks"`
}

//go:embed packs/*.json
var builtinFS embed.FS

// BuiltinNames lists the packs shipped with the CLI, sorted.
func BuiltinNames() []string {
	entries, err := builtinFS.ReadDir("packs")
	if err != nil {
		return nil
	}
	var names []string
	for _, e := range entries {
		names = append(names, strings.TrimSuffix(e.Name(), ".json"))
	}
	sort.Strings(names)
	return names
}

// Resolve loads a pack from `ref`: an existing file path wins, otherwise the
// name of a built-in pack.
func Resolve(ref string) (*Pack, error) {
	if info, err := os.Stat(ref); err == nil && !info.IsDir() {
		data, err := os.ReadFile(ref)
		if err != nil {
			return nil, fmt.Errorf("read eval pack %s: %w", ref, err)
		}
		pack, err := parsePack(data)
		if err != nil {
			return nil, fmt.Errorf("eval pack %s: %w", filepath.Base(ref), err)
		}
		return pack, nil
	}
	data, err := builtinFS.ReadFile("packs/" + ref + ".json")
	if err != nil {
		return nil, fmt.Errorf("no eval pack %q: not a file, and not a built-in (available: %s)",
			ref, strings.Join(BuiltinNames(), ", "))
	}
	return parsePack(data)
}

func parsePack(data []byte) (*Pack, error) {
	var pack Pack
	if err := json.Unmarshal(data, &pack); err != nil {
		return nil, fmt.Errorf("invalid JSON: %w", err)
	}
	if pack.Name == "" {
		return nil, fmt.Errorf("missing \"name\"")
	}
	if len(pack.Tasks) == 0 {
		return nil, fmt.Errorf("no tasks")
	}
	seen := make(map[string]bool, len(pack.Tasks))
	for i := range pack.Tasks {
		t := &pack.Tasks[i]
		if t.ID == "" {
			t.ID = fmt.Sprintf("task-%d", i+1)
		}
		if seen[t.ID] {
			return nil, fmt.Errorf("duplicate task id %q", t.ID)
		}
		seen[t.ID] = true
		if t.Prompt == "" {
			return nil, fmt.Errorf("task %q: missing \"prompt\"", t.ID)
		}
		switch t.Type {
		case TaskMultipleChoice:
			if len(t.Choices) < 2 || len(t.Choices) > 26 {
				return nil, fmt.Errorf("task %q: multiple_choice needs 2-26 choices, got %d", t.ID, len(t.Choices))
			}
			letter := strings.ToUpper(strings.TrimSpace(t.Answer))
			if len(letter) != 1 || letter[0] < 'A' || int(letter[0]-'A') >= len(t.Choices) {
				return nil, fmt.Errorf("task %q: \"answer\" must be a letter A-%c", t.ID, 'A'+len(t.Choices)-1)
			}
			t.Answer = letter
		case TaskExactMatch, TaskContains:
			if t.Expected == "" {
				return nil, fmt.Errorf("task %q: missing \"expected\"", t.ID)
			}
		default:
			return nil, fmt.Errorf("task %q: unknown type %q (supported: %s, %s, %s)",
				t.ID, t.Type, TaskMultipleChoice, TaskExactMatch, TaskContains)
		}
	}
	return &pack, nil
}

// RenderPrompt returns the prompt sent to the model; multiple-choice tasks
// get lettered options and an answer-format instruction.
func RenderPrompt(t Task) string {
	if t.Type != TaskMultipleChoice {
		return t.Prompt
	}
	var b strings.Builder
	b.WriteString(t.Prompt)
	b.WriteString("\n")
	for i, choice := range t.Choices {
		fmt.Fprintf(&b, "\n%c. %s", 'A'+i, choice)
	}
	b.WriteString("\n\nAnswer with only the letter of the correct choice.")
	return b.String()
}
