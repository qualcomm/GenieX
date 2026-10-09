// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package jev

import (
	"context"
	"testing"
)

func TestShortlistPreservesCanonicalCandidates(t *testing.T) {
	candidates := []Candidate{{ID: "a", Text: "Alpha"}, {ID: "b", Text: "Beta"}, {ID: "c", Text: "Gamma"}}
	got, err := Shortlist(context.Background(), func(_ context.Context, query string, input []Candidate, limit int) ([]Candidate, error) {
		if query != "question" || len(input) != 3 || limit != 2 {
			t.Fatalf("ranker input = (%q, %#v, %d)", query, input, limit)
		}
		return []Candidate{{ID: "c", Text: "untrusted replacement"}, {ID: "a", Text: "also ignored"}}, nil
	}, "question", candidates, 2)
	if err != nil {
		t.Fatalf("Shortlist() error = %v", err)
	}
	if len(got) != 2 || got[0] != candidates[2] || got[1] != candidates[0] {
		t.Fatalf("Shortlist() = %#v", got)
	}
}

func TestShortlistRejectsMalformedRankerOutput(t *testing.T) {
	candidates := []Candidate{{ID: "a", Text: "Alpha"}, {ID: "b", Text: "Beta"}}
	for _, test := range []struct {
		name   string
		ranked []Candidate
	}{
		{"empty", nil},
		{"unknown", []Candidate{{ID: "x", Text: "Unknown"}}},
		{"duplicate", []Candidate{{ID: "a", Text: "Alpha"}, {ID: "a", Text: "Alpha"}}},
		{"too many", []Candidate{{ID: "a", Text: "Alpha"}, {ID: "b", Text: "Beta"}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := Shortlist(context.Background(), func(context.Context, string, []Candidate, int) ([]Candidate, error) {
				return test.ranked, nil
			}, "question", candidates, 1)
			if err == nil {
				t.Fatal("Shortlist() accepted malformed ranker output")
			}
		})
	}
}

func TestShortlistSkipsRankerForSmallSets(t *testing.T) {
	candidates := []Candidate{{ID: "a", Text: "Alpha"}}
	got, err := Shortlist(context.Background(), func(context.Context, string, []Candidate, int) ([]Candidate, error) {
		t.Fatal("ranker called for a small candidate set")
		return nil, nil
	}, "question", candidates, 1)
	if err != nil || len(got) != 1 || got[0] != candidates[0] {
		t.Fatalf("Shortlist() = (%#v, %v)", got, err)
	}
}
