// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import (
	"strings"

	"github.com/bytedance/sonic"
)

// GPT-OSS addresses a call to the tool namespace in a message header, the recipient
// sitting in either the role or the channel:
//
//	<|start|>assistant to=functions.NAME<|channel|>commentary<|message|>{"x":1}<|call|>
//	<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{"x":1}<|call|>
//
// The recipient is what marks a call — it also stands alone, which is what a
// reasoning splitter upstream leaves once it has eaten `<|channel|>` — so it is the
// one literal this scans for, and the header before it goes with the call whenever
// it has not been emitted yet. Arguments are JSON, so the object closing ends the
// call. Grammar: llama.cpp's common_chat_params_init_gpt_oss.
const (
	gptOssRecipient = " to=functions."
	gptOssMessage   = "<|message|>"
	gptOssCall      = "<|call|>"
)

// gptOssHeaders are the openers that stand in front of the recipient. A call can
// carry both, as `<|start|>assistant<|channel|>commentary to=functions.NAME` does.
var gptOssHeaders = []string{"<|channel|>commentary", "<|channel|>analysis", "<|start|>assistant"}

// gptOssChannels is what a reasoning splitter upstream leaves of a channel whose
// `<|channel|>` it has eaten. Bare like this it only ever sits against the recipient.
var gptOssChannels = []string{"commentary", "analysis"}

// gptOssToolCall keeps no scan state beyond how far the recipient has been looked
// for: a call is short, so once one starts the rest is re-read on each chunk.
type gptOssToolCall struct {
	pos int // bytes searched, less the tail a split recipient could straddle
}

func newGptOssToolCall() *gptOssToolCall { return &gptOssToolCall{} }

func (t *gptOssToolCall) parse(s string) []toolCallFn { return parseGptOssToolCalls(s) }

func (t *gptOssToolCall) feed(all string, from int) (int, int) {
	at, done := findMarker(all, gptOssRecipient, max(from, t.pos))
	if done == 0 {
		// Nothing whole yet: resume where a recipient split across chunks would
		// still be found, so no byte is searched twice.
		t.pos = max(from, len(all)-len(gptOssRecipient)+1)
		if at < 0 {
			return -1, -1
		}
		return at, -1 // only a prefix of the recipient so far
	}
	t.pos = at // idempotent: the same region is reported until it is consumed

	// The role and the channel are the call's syntax, not content. In a stream they
	// have gone out before the recipient arrives, and then the region starts at it.
	if role := headerStart(all, at); role >= from {
		at = role
	}

	_, msg := findMarker(all, gptOssMessage, done)
	if msg == 0 {
		return at, -1 // the header has not closed
	}
	brace := strings.IndexByte(all[msg:], '{')
	if brace < 0 {
		return at, -1 // the arguments have not started
	}
	var walk braceWalk
	n := walk.feed(all[msg+brace:])
	if n < 0 {
		return at, -1 // the arguments are still open
	}
	end, settled := optionalTail(all, msg+brace+n, gptOssCall)
	if !settled {
		return at, -1
	}
	return at, end
}

// headerStart is where the header holding the recipient at i begins, i itself when
// nothing of one stands in front of it.
func headerStart(all string, i int) int {
	for first := true; ; first = false {
		j := trimOpener(all, i, gptOssHeaders)
		if j == i && first { // only against the recipient, and only if nothing else fits
			j = trimOpener(all, i, gptOssChannels)
		}
		if j == i {
			return i
		}
		i = j
	}
}

// trimOpener steps i back over one of openers when all[:i] ends with it.
func trimOpener(all string, i int, openers []string) int {
	for _, opener := range openers {
		if strings.HasSuffix(all[:i], opener) {
			return i - len(opener)
		}
	}
	return i
}

// findMarker locates lit in all[from:]: (at, end) once it is whole, (at, 0) while
// only a prefix of it sits at the tail, and (-1, 0) when neither. strings.Index is
// vectorised, which beats walking a marker this long byte by byte.
func findMarker(all, lit string, from int) (at, end int) {
	if i := strings.Index(all[from:], lit); i >= 0 {
		return from + i, from + i + len(lit)
	}
	for i := max(from, len(all)-len(lit)+1); i < len(all); i++ {
		if all[i] == lit[0] && strings.HasPrefix(lit, all[i:]) {
			return i, 0
		}
	}
	return -1, 0
}

// optionalTail extends a region past lit when lit follows it, so a closing marker
// the syntax does not need does not go out as text. settled is false while lit may
// still be arriving: the caller has to wait for more input.
func optionalTail(all string, at int, lit string) (end int, settled bool) {
	i := skipSpace(all, at)
	switch rest := all[i:]; {
	case strings.HasPrefix(rest, lit):
		return i + len(lit), true
	case strings.HasPrefix(lit, rest): // a prefix so far; it may yet complete
		return at, false
	default:
		return at, true
	}
}

// parseGptOssToolCalls returns every call in s. Whatever sits between the
// recipient and `<|message|>` is header syntax — a channel, a constraint — and is
// ignored, since the recipient has already settled that this is a call.
func parseGptOssToolCalls(s string) []toolCallFn {
	var calls []toolCallFn
	for {
		i := strings.Index(s, gptOssRecipient)
		if i < 0 {
			return calls
		}
		s = s[i+len(gptOssRecipient):]
		name := s[:gptOssNameEnd(s)]

		msg := strings.Index(s, gptOssMessage)
		if msg < 0 {
			return calls
		}
		s = s[msg+len(gptOssMessage):]
		brace := strings.IndexByte(s, '{')
		if brace < 0 {
			return calls
		}
		var w braceWalk
		end := w.feed(s[brace:])
		if end < 0 {
			return calls // the arguments never closed
		}
		args := s[brace : brace+end]
		s = s[brace+end:]
		if name == "" || !sonic.ValidString(args) {
			continue
		}
		calls = append(calls, toolCallFn{Name: name, Arguments: args})
	}
}

// gptOssNameEnd is the length of the tool name at the head of s.
func gptOssNameEnd(s string) int {
	i := 0
	for i < len(s) {
		c := s[i]
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' ||
			c == '_' || c == '-' || c == '.') {
			break
		}
		i++
	}
	return i
}
