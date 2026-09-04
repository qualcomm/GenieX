// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package eval

import (
	"regexp"
	"strings"
)

var (
	// A closed <think> block, or an unterminated one (model never answered).
	thinkRe = regexp.MustCompile(`(?s)<think>(.*?</think>|.*$)`)
	// Uppercase only: a standalone lowercase letter is usually the article "a".
	letterRe     = regexp.MustCompile(`\b([A-Z])\b`)
	loneLetterRe = regexp.MustCompile(`^[^A-Za-z0-9]*([A-Za-z])[^A-Za-z0-9]*$`)
)

// Score reports whether output answers t correctly. Unparseable output is
// simply wrong, never an error.
func Score(t Task, output string) bool {
	answer := strings.TrimSpace(thinkRe.ReplaceAllString(output, ""))
	switch t.Type {
	case TaskMultipleChoice:
		idx, ok := extractChoice(answer, t.Choices)
		return ok && idx == int(t.Answer[0]-'A')
	case TaskExactMatch:
		return normalize(answer) == normalize(t.Expected)
	case TaskContains:
		return strings.Contains(normalize(answer), normalize(t.Expected))
	default:
		return false
	}
}

func normalize(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	return strings.TrimSpace(strings.TrimRight(s, ".!?"))
}

// extractChoice maps output to a choice index: a bare letter ("b", "(B)."),
// then the first standalone uppercase letter in range ("Answer: B"), then an
// output matching exactly one choice's text.
func extractChoice(answer string, choices []string) (int, bool) {
	if m := loneLetterRe.FindStringSubmatch(answer); m != nil {
		idx := int(strings.ToUpper(m[1])[0] - 'A')
		return idx, idx < len(choices)
	}
	for _, m := range letterRe.FindAllStringSubmatch(answer, -1) {
		if idx := int(m[1][0] - 'A'); idx < len(choices) {
			return idx, true
		}
	}
	normalized := normalize(answer)
	if normalized == "" {
		return 0, false
	}
	matched, count := -1, 0
	for i, choice := range choices {
		if strings.Contains(normalized, normalize(choice)) {
			matched = i
			count++
		}
	}
	return matched, count == 1
}
