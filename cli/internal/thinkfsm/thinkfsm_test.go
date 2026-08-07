// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package thinkfsm

import "testing"

// split feeds tokens through a fresh splitter and returns the concatenated
// content and reasoning streams (markers dropped).
func split(tokens []string) (content, reasoning string) {
	s := New()
	for _, tok := range tokens {
		ev := s.Feed(tok)
		if ev.Consumed {
			continue
		}
		if ev.Reasoning {
			reasoning += ev.Text
		} else {
			content += ev.Text
		}
	}
	return content, reasoning
}

func TestSplit(t *testing.T) {
	cases := []struct {
		name          string
		tokens        []string
		wantContent   string
		wantReasoning string
	}{
		{
			name:        "plain content, no thinking",
			tokens:      []string{"2", "+", "2", " is ", "4"},
			wantContent: "2+2 is 4",
		},
		{
			name:          "inline think block",
			tokens:        []string{"<think>", "hmm", " 2+2", "</think>", "4"},
			wantContent:   "4",
			wantReasoning: "hmm 2+2",
		},
		{
			name:          "gpt-oss channels",
			tokens:        []string{"<|channel|>", "analysis", "<|message|>", "reason", "<|end|>", "<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "answer"},
			wantContent:   "answer",
			wantReasoning: "reason",
		},
		{
			name:          "gemma channel thought",
			tokens:        []string{"<|channel>", "thought", "\n", "think", "<channel|>", "reply"},
			wantContent:   "reply",
			wantReasoning: "think",
		},
		{
			name:        "unterminated think stays reasoning",
			tokens:      []string{"<think>", "still", " going"},
			wantContent: "",
			// everything after <think> is reasoning until a close marker
			wantReasoning: "still going",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			content, reasoning := split(tc.tokens)
			if content != tc.wantContent {
				t.Errorf("content = %q, want %q", content, tc.wantContent)
			}
			if reasoning != tc.wantReasoning {
				t.Errorf("reasoning = %q, want %q", reasoning, tc.wantReasoning)
			}
		})
	}
}

func TestFeedBoundaries(t *testing.T) {
	s := New()

	ev := s.Feed("<think>")
	if !ev.Consumed || ev.Boundary != EnterReasoning {
		t.Fatalf("<think>: got %+v, want consumed EnterReasoning", ev)
	}
	ev = s.Feed("x")
	if ev.Consumed || ev.Reasoning != true || ev.Text != "x" {
		t.Fatalf("x: got %+v, want reasoning text", ev)
	}
	ev = s.Feed("</think>")
	if !ev.Consumed || ev.Boundary != ExitReasoning {
		t.Fatalf("</think>: got %+v, want consumed ExitReasoning", ev)
	}
	ev = s.Feed("y")
	if ev.Consumed || ev.Reasoning || ev.Text != "y" {
		t.Fatalf("y: got %+v, want content text", ev)
	}
}
