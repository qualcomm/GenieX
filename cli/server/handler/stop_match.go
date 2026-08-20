// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import "strings"

// stopMatcher enforces OpenAI-style stop sequences host side, for plugins
// (qairt) that reject the parameter outright. It buffers just enough of the
// generated tail to catch a stop sequence split across two token
// boundaries, without holding back more text than necessary.
type stopMatcher struct {
	stops []string
	buf   string
}

func newStopMatcher(stops []string) *stopMatcher {
	return &stopMatcher{stops: stops}
}

// feed appends token to the internal buffer and returns the portion that is
// now safe to emit to the client, plus whether a stop sequence has fully
// matched. Once stopped is true, the caller must cancel generation and stop
// calling feed — the match itself (and anything after it) is dropped.
func (m *stopMatcher) feed(token string) (safe string, stopped bool) {
	m.buf += token

	if idx := firstStopIndex(m.buf, m.stops); idx >= 0 {
		safe = m.buf[:idx]
		m.buf = ""
		return safe, true
	}

	// Hold back a suffix that could still grow into a stop sequence on the
	// next token; only the rest is guaranteed clean of any stop sequence.
	hold := longestStopPrefixSuffixLen(m.buf, m.stops)
	safe, m.buf = m.buf[:len(m.buf)-hold], m.buf[len(m.buf)-hold:]
	return safe, false
}

// firstStopIndex returns the earliest byte offset in buf where a stop
// sequence starts, or -1 if none has fully matched yet.
func firstStopIndex(buf string, stops []string) int {
	idx := -1
	for _, s := range stops {
		if s == "" {
			continue
		}
		if i := strings.Index(buf, s); i >= 0 && (idx == -1 || i < idx) {
			idx = i
		}
	}
	return idx
}

// longestStopPrefixSuffixLen returns the length of the longest suffix of buf
// that is also a strict prefix of some stop sequence — the part that must be
// held back because the next token could still complete a match.
func longestStopPrefixSuffixLen(buf string, stops []string) int {
	longest := 0
	for _, s := range stops {
		if s == "" {
			continue
		}
		limit := len(s) - 1
		if limit > len(buf) {
			limit = len(buf)
		}
		for l := limit; l > 0; l-- {
			if strings.HasSuffix(buf, s[:l]) {
				if l > longest {
					longest = l
				}
				break
			}
		}
	}
	return longest
}
