// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import (
	"log/slog"
	"strings"

	"github.com/openai/openai-go/v3"
)

type toolCallFn = openai.ChatCompletionMessageFunctionToolCallFunction

// toolCallFormat is one tool-call syntax. feed is stateful: one value per stream.
type toolCallFormat interface {
	// parse searches s for every call of this syntax, so one region works too.
	parse(s string) []toolCallFn
	// feed consumes what it has not seen of all and reports where this syntax
	// sits, ignoring anything before from:
	//
	//	(-1, -1)  nothing here; every byte of all is safe to emit
	//	(n, -1)   a call may begin at n but has not finished
	//	(n, m)    all[n:m] is a finished candidate; parse decides
	//
	// all only grows and from never decreases, so offsets stay absolute. A region
	// is reported again until from moves past it, so an earlier one can win first.
	feed(all string, from int) (int, int)
}

// markerScan finds one literal in a growing string, one pass, no backtracking.
type markerScan struct {
	marker string
	pos    int // bytes of all consumed
	start  int // where the candidate began; == pos when none is forming
	done   int // where the marker ended; 0 until it matches, never 0 after
}

func (m *markerScan) reset(from int) { m.pos, m.start, m.done = from, from, 0 }

func (m *markerScan) feed(all string) {
	for m.done == 0 && m.pos < len(all) {
		if m.start == m.pos {
			i := strings.IndexByte(all[m.pos:], m.marker[0])
			if i < 0 {
				m.start, m.pos = len(all), len(all)
				return
			}
			m.start, m.pos = m.pos+i, m.pos+i
		}
		m.pos++
		// Sliding start up to pos gives up: all[pos-1:pos] was the last try.
		for m.start < m.pos && !strings.HasPrefix(m.marker, all[m.start:m.pos]) {
			m.start++
		}
		if m.pos-m.start == len(m.marker) {
			m.done = m.pos
		}
	}
}

// markerFormat covers a syntax wrapped in two literals. Without a closing one, a
// format implements feed itself, as jsonToolCall does.
//
// BUG: the scan is not string-aware, so an end literal quoted inside the call's
// own arguments cuts the region short and the call goes out as text instead. What
// quotes a string differs per format, so a fix needs a hook the formats fill in.
type markerFormat struct {
	begin markerScan
	end   markerScan
}

func newMarkerFormat(begin, end string) markerFormat {
	return markerFormat{begin: markerScan{marker: begin}, end: markerScan{marker: end}}
}

func (m *markerFormat) feed(all string, from int) (int, int) {
	if from > m.begin.start { // the region was consumed or bypassed: start over
		m.begin.reset(from)
		m.end.reset(from)
	}
	m.begin.feed(all)
	if m.begin.done == 0 {
		if m.begin.start < len(all) {
			return m.begin.start, -1 // only a prefix of begin so far
		}
		return -1, -1
	}
	if m.end.pos < m.begin.done {
		m.end.reset(m.begin.done)
	}
	m.end.feed(all)
	at := m.begin.done - len(m.begin.marker)
	if m.end.done == 0 {
		return at, -1
	}
	return at, m.end.done
}

// ToolCallScanner splits a token stream into text and tool calls, holding back
// only what could still be part of a call.
type ToolCallScanner struct {
	buf     strings.Builder // String() shares its bytes: keeping it all is free
	emitted int
	formats []toolCallFormat
}

// NewToolCallScanner registers the formats most specific first, since bare JSON
// matches inside anything.
func NewToolCallScanner() *ToolCallScanner {
	return &ToolCallScanner{formats: []toolCallFormat{
		newGemma4ToolCall(),
		newQwen3ToolCall(),
		newQwen35ToolCall(),
		newGptOssToolCall(),
		newLFM2ToolCall(),
		&jsonToolCall{},
	}}
}

// Push appends a chunk and returns the text safe to emit plus the calls that finished.
func (s *ToolCallScanner) Push(token string) (string, []toolCallFn) {
	s.buf.WriteString(token)

	// Each pass consumes at most one region, and a chunk can hold several.
	text, calls := "", []toolCallFn(nil)
	for {
		// The earliest format wins: what a later one found may sit inside its region.
		all, from := s.buf.String(), s.emitted
		hold, end := len(all), -1
		for _, f := range s.formats {
			n, m := f.feed(all, from)
			if n < 0 || n >= hold {
				continue
			}
			hold, end = n, m
		}

		// Nothing finished: emit up to the hold point, buffer the rest.
		if end < 0 {
			s.emitted = hold
			if text == "" { // the hot path: nothing to prepend, so skip the concat
				return all[from:hold], calls
			}
			return text + all[from:hold], calls
		}

		s.emitted = end
		got := s.parse(all[hold:end])
		if len(got) == 0 {
			text += all[from:end] // the region was prose all along
			continue
		}
		slog.Debug("Parsed tool calls", "tool_calls", got)
		text += all[from:hold]
		calls = append(calls, got...)
	}
}

// parse reads a region with each format in turn: formats can share an opener —
// Qwen3 and Qwen3.5 both wrap their calls in `<tool_call>` — so the one that holds
// the region is not always the one whose syntax it is.
func (s *ToolCallScanner) parse(region string) []toolCallFn {
	for _, f := range s.formats {
		if calls := f.parse(region); len(calls) > 0 {
			return calls
		}
	}
	return nil
}

// Parse splits a whole response on a fresh scanner, keeping and dropping what the
// streaming path does.
func (s *ToolCallScanner) Parse(all string) (string, []toolCallFn) {
	text, calls := s.Push(all)
	tail, got := s.Tail()
	return text + tail, append(calls, got...)
}

// Tail ends the stream. What is buffered is text, unless the model stopped short of
// closing a call: last chance to find one, so try every format, not just the holder.
func (s *ToolCallScanner) Tail() (string, []toolCallFn) {
	tail := s.buf.String()[s.emitted:]
	s.emitted += len(tail) // terminal: a second call has nothing left to report
	calls := s.parse(tail)
	if len(calls) == 0 {
		return tail, nil
	}
	// parse reports no offsets, so text before or between the calls goes too.
	slog.Warn("Tool call recovered from unclosed output, dropping the text held with it",
		"held_bytes", len(tail), "tool_calls", len(calls))
	return "", calls
}
