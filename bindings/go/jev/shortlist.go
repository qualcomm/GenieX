// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package jev

import (
	"context"
	"fmt"
)

// Candidate is a stable, host-owned candidate for an optional coarse-to-fine
// decision. ID is the only selectable value; Text is untrusted descriptive
// content for a host-supplied ranker.
type Candidate struct {
	ID   string
	Text string
}

// ShortlistFunc ranks host-owned candidates for one query. It may only narrow
// the supplied set; Shortlist restores canonical candidate text and validates
// every returned ID before a model is invoked.
type ShortlistFunc func(context.Context, string, []Candidate, int) ([]Candidate, error)

// Shortlist returns at most limit candidates. Small candidate sets are returned
// unchanged without invoking ranker. A ranker can never add candidates or alter
// canonical host text, and malformed results fail closed.
func Shortlist(ctx context.Context, ranker ShortlistFunc, query string, candidates []Candidate, limit int) ([]Candidate, error) {
	if limit < 1 {
		return nil, fmt.Errorf("shortlist limit must be at least 1")
	}
	canonical, err := canonicalCandidates(candidates)
	if err != nil {
		return nil, err
	}
	if len(candidates) <= limit {
		return append([]Candidate(nil), candidates...), nil
	}
	if ranker == nil {
		return nil, fmt.Errorf("shortlist ranker is required when %d candidates exceed limit %d", len(candidates), limit)
	}
	ranked, err := ranker(ctx, query, append([]Candidate(nil), candidates...), limit)
	if err != nil {
		return nil, fmt.Errorf("rank shortlist candidates: %w", err)
	}
	if len(ranked) == 0 || len(ranked) > limit {
		return nil, fmt.Errorf("shortlist returned %d candidates, want between 1 and %d", len(ranked), limit)
	}
	selected := make([]Candidate, 0, len(ranked))
	seen := make(map[string]struct{}, len(ranked))
	for _, candidate := range ranked {
		original, ok := canonical[candidate.ID]
		if !ok {
			return nil, fmt.Errorf("shortlist returned unknown candidate %q", candidate.ID)
		}
		if _, duplicate := seen[candidate.ID]; duplicate {
			return nil, fmt.Errorf("shortlist returned duplicate candidate %q", candidate.ID)
		}
		seen[candidate.ID] = struct{}{}
		selected = append(selected, original)
	}
	return selected, nil
}

func canonicalCandidates(candidates []Candidate) (map[string]Candidate, error) {
	if len(candidates) == 0 {
		return nil, fmt.Errorf("shortlist candidates are required")
	}
	canonical := make(map[string]Candidate, len(candidates))
	for _, candidate := range candidates {
		if candidate.ID == "" {
			return nil, fmt.Errorf("shortlist candidate ID is required")
		}
		if candidate.Text == "" {
			return nil, fmt.Errorf("shortlist candidate %q text is required", candidate.ID)
		}
		if _, duplicate := canonical[candidate.ID]; duplicate {
			return nil, fmt.Errorf("shortlist candidates contain duplicate ID %q", candidate.ID)
		}
		canonical[candidate.ID] = candidate
	}
	return canonical, nil
}
