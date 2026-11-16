// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package thinkfsm

import (
	"slices"
	"strconv"
	"testing"
)

type parseCase struct {
	name   string
	tokens []string
	kinds  []EventKind
}

func checkTokens(t *testing.T, s *Splitter, tokens []string, kinds []EventKind) {
	t.Helper()
	if len(tokens) != len(kinds) {
		t.Fatalf("tokens = %d, expected events = %d", len(tokens), len(kinds))
	}
	for i, token := range tokens {
		ev := s.Feed(token)
		if ev.Kind != kinds[i] {
			t.Errorf("token %d %q: got %+v, want kind %v", i, token, ev, kinds[i])
		}
		switch kinds[i] {
		case Consume:
			if ev.Text != "" && ev.Text != token {
				t.Errorf("token %d %q: Consume text = %q, want empty or input token", i, token, ev.Text)
			}
		case EnterThinking, LeaveThinking, ToolCallStart, ToolCallEnd:
			if ev.Text != "" {
				t.Errorf("token %d %q: boundary event = %+v, want empty text", i, token, ev)
			}
		case Content, Thinking, ToolCall:
			if ev.Text != token {
				t.Errorf("token %d %q: event = %+v, want input token text", i, token, ev)
			}
		default:
			t.Errorf("token %d %q: unknown event kind %v", i, token, ev.Kind)
		}
	}
}

func checkCases(t *testing.T, cases []parseCase) {
	t.Helper()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			checkTokens(t, New(), tc.tokens, tc.kinds)
		})
	}
}

// Basic functionality: construction and one-token transition dispatch.
func TestBasicFunctions(t *testing.T) {
	t.Run("initial state", func(t *testing.T) {
		if s := New(); s.state != stateAssistant {
			t.Fatalf("initial state = %v", s.state)
		}
	})
	t.Run("empty input token", func(t *testing.T) {
		for _, tc := range []struct {
			name  string
			state state
			kind  EventKind
		}{
			{"content", stateAssistant, Content},
			{"thinking", stateThink, Thinking},
			{"ordinary header", stateGPTOSSAssistantHeader, Consume},
			{"tool header", stateGPTOSSToolHeader, ToolCall},
			{"ignored body", stateGPTOSSIgnored, Consume},
		} {
			t.Run(tc.name, func(t *testing.T) {
				s := &Splitter{state: tc.state}
				checkTokens(t, s, []string{""}, []EventKind{tc.kind})
				if s.state != tc.state {
					t.Fatalf("empty token: next state = %v, want %v", s.state, tc.state)
				}
			})
		}
	})
	t.Run("independent splitters", func(t *testing.T) {
		a, b := New(), New()
		checkTokens(t, a, []string{"<think>", "reason"}, []EventKind{EnterThinking, Thinking})
		checkTokens(t, b, []string{"answer"}, []EventKind{Content})
		checkTokens(t, a, []string{"</think>", "answer"}, []EventKind{LeaveThinking, Content})
	})
	t.Run("fallback for every state", func(t *testing.T) {
		for st := stateAssistant; st <= stateGemmaNormal; st++ {
			tr, ok := transitions[key{st, ""}]
			if !ok {
				t.Fatalf("state %v has no fallback", st)
			}
			s := &Splitter{state: st}
			ev := s.Feed("unmatched")
			text := "unmatched"
			switch tr.Kind {
			case EnterThinking, LeaveThinking, ToolCallStart, ToolCallEnd:
				text = ""
			}
			if ev != (Event{Kind: tr.Kind, Text: text}) || s.state != tr.next {
				t.Fatalf("state %v fallback: event = %+v, next = %v", st, ev, s.state)
			}
		}
	})
	t.Run("exact transitions", func(t *testing.T) {
		for k, tr := range transitions {
			if k.token == "" {
				continue
			}
			s := &Splitter{state: k.s}
			if got := s.Feed(k.token); got != tr.Event || s.state != tr.next {
				t.Fatalf("transition %+v: event = %+v, next = %v", k, got, s.state)
			}
		}
	})
}

// Ordinary models: literal content and <think> boundaries.
func TestOrdinaryModelParsing(t *testing.T) {
	checkCases(t, []parseCase{
		{
			name:   "plain text",
			tokens: []string{"2", "+", "2", " is ", "4"},
			kinds:  []EventKind{Content, Content, Content, Content, Content},
		},
		{
			name:   "thinking and answer",
			tokens: []string{"<think>", "hmm", " 2+2", "</think>", "4"},
			kinds:  []EventKind{EnterThinking, Thinking, Thinking, LeaveThinking, Content},
		},
		{
			name:   "empty thinking",
			tokens: []string{"<think>", "</think>", "answer"},
			kinds:  []EventKind{EnterThinking, LeaveThinking, Content},
		},
		{
			name:   "unterminated thinking",
			tokens: []string{"<think>", "still", " going"},
			kinds:  []EventKind{EnterThinking, Thinking, Thinking},
		},
		{
			name:   "unmatched closing tag",
			tokens: []string{"</think>", "answer"},
			kinds:  []EventKind{Content, Content},
		},
		{
			name:   "foreign markers remain thinking and content",
			tokens: []string{"<think>", "<|channel|>", "<|end|>", "<|call|>", "<|channel>", "<channel|>", "</think>", "<|channel|>", "<|end|>", "<|call|>", "<|channel>", "<channel|>"},
			kinds:  []EventKind{EnterThinking, Thinking, Thinking, Thinking, Thinking, Thinking, LeaveThinking, Content, Content, Content, Content, Content},
		},
		{
			name:   "no splitting or buffering",
			tokens: []string{"<thi", "nk>", "<think>reason</think>"},
			kinds:  []EventKind{Content, Content, Content},
		},
	})
}

func TestMinistral3Parsing(t *testing.T) {
	checkCases(t, []parseCase{
		{
			name:   "thinking and answer",
			tokens: []string{"[THINK]", "reason", "[/THINK]", "answer"},
			kinds:  []EventKind{EnterThinking, Thinking, LeaveThinking, Content},
		},
		{
			name:   "empty thinking",
			tokens: []string{"[THINK]", "[/THINK]", "answer"},
			kinds:  []EventKind{EnterThinking, LeaveThinking, Content},
		},
		{
			name:   "unterminated thinking",
			tokens: []string{"[THINK]", "reason"},
			kinds:  []EventKind{EnterThinking, Thinking},
		},
		{
			name:   "tool syntax inside thinking stays literal",
			tokens: []string{"[THINK]", "[TOOL_CALLS]", "f", "[ARGS]", "{}", "[/THINK]", "[TOOL_CALLS]", "g", "[ARGS]", "{}"},
			kinds:  []EventKind{EnterThinking, Thinking, Thinking, Thinking, Thinking, LeaveThinking, Content, Content, Content, Content},
		},
		{
			name:   "foreign markers stay literal",
			tokens: []string{"[THINK]", "<think>", "</think>", "<|channel|>", "[/THINK]", "[THINK]", "[/THINK]"},
			kinds:  []EventKind{EnterThinking, Thinking, Thinking, Thinking, LeaveThinking, Content, Content},
		},
		{
			name:   "Ministral closing marker cannot end ordinary thinking",
			tokens: []string{"<think>", "[/THINK]", "</think>"},
			kinds:  []EventKind{EnterThinking, Thinking, LeaveThinking},
		},
		{
			name:   "unmatched closing marker",
			tokens: []string{"[/THINK]", "answer"},
			kinds:  []EventKind{Content, Content},
		},
		{
			name:   "no splitting or buffering",
			tokens: []string{"[TH", "INK]", "[THINK]reason[/THINK]"},
			kinds:  []EventKind{Content, Content, Content},
		},
	})
}

// GPT-OSS: token-wise channels, headers, message boundaries and recovery.
func TestGPTOSSParsing(t *testing.T) {
	t.Run("messages", func(t *testing.T) {
		checkCases(t, []parseCase{
			{
				name:   "analysis to commentary to final",
				tokens: []string{"<|channel|>", "analysis", "<|message|>", "reason", "<|end|>", "<|start|>", "assistant", "<|channel|>", "comment", "ary", "<|message|>", "checking", "<|end|>", "<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "answer", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, EnterThinking, Thinking, LeaveThinking, Consume, Consume, Consume, Consume, Consume, Consume, Content, Consume, Consume, Consume, Consume, Consume, Consume, Content, Consume},
			},
			{
				name:   "multiple analysis messages",
				tokens: []string{"<|start|>", "assistant", "<|channel|>", "analysis", "<|message|>", "first", "<|end|>", "<|start|>", "assistant", "<|channel|>", "analysis", "<|message|>", "second", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, EnterThinking, Thinking, LeaveThinking, Consume, Consume, Consume, Consume, EnterThinking, Thinking, LeaveThinking},
			},
			{
				name:   "foreign tags remain body text",
				tokens: []string{"<|channel|>", "analysis", "<|message|>", "<think>", "</think>", "<|channel>", "<channel|>", "<|end|>", "<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "<think>", "</think>", "<|channel>", "<channel|>"},
				kinds:  []EventKind{Consume, Consume, EnterThinking, Thinking, Thinking, Thinking, Thinking, LeaveThinking, Consume, Consume, Consume, Consume, Consume, Content, Content, Content, Content},
			},
		})
	})
	t.Run("headers", func(t *testing.T) {
		checkCases(t, []parseCase{
			{
				name:   "recipient in commentary header",
				tokens: []string{"<|channel|>", "comment", "ary", " to", "=", "functions", ".", "f", "<|constrain|>", "json", "<|message|>", "{}", "<|call|>"},
				kinds:  []EventKind{Consume, Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, ToolCall, Consume, Consume, ToolCallStart, ToolCall, ToolCallEnd},
			},
			{
				name:   "recipient in assistant header",
				tokens: []string{"<|start|>", "assistant", " to", "=", "functions", ".", "f", "<|channel|>", "comment", "ary", "<|message|>", "{}", "<|call|>"},
				kinds:  []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, ToolCall, Consume, Consume, Consume, ToolCallStart, ToolCall, ToolCallEnd},
			},
			{
				name:   "recipient in assistant header before analysis",
				tokens: []string{"<|start|>", "assistant", " to", "=", "functions", ".", "f", "<|channel|>", "analysis", "<|message|>", "reason", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, ToolCall, Consume, Consume, ToolCallStart, ToolCall, ToolCallEnd},
			},
			{
				name:   "recipient in assistant header before final",
				tokens: []string{"<|start|>", "assistant", " to", "=", "functions", ".", "f", "<|channel|>", "final", "<|message|>", "{}", "<|call|>"},
				kinds:  []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, ToolCall, Consume, Consume, ToolCallStart, ToolCall, ToolCallEnd},
			},
			{
				name:   "recipient and structured analysis header",
				tokens: []string{"<|channel|>", "analysis", " to", "=", "functions", ".", "f", "<|constrain|>", "json", "<|message|>", "{}", "<|call|>"},
				kinds:  []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, ToolCall, Consume, Consume, ToolCallStart, ToolCall, ToolCallEnd},
			},
			{
				name:   "assistant header without channel",
				tokens: []string{"<|start|>", "assistant", "<|message|>", "answer", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, Consume, Content, Consume},
			},
			{
				name:   "recipient and structured final header",
				tokens: []string{"<|channel|>", "final", " to", "=", "functions", ".", "f", " ", "<|constrain|>", "json", "<|message|>", "{}"},
				kinds:  []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, ToolCall, ToolCall, Consume, Consume, ToolCallStart, ToolCall},
			},
			{
				name:   "commentary followed by analysis header",
				tokens: []string{"<|channel|>", "comment", "ary", " to", "=", "assistant", "<|channel|>", "analysis", "<|message|>", "reason"},
				kinds:  []EventKind{Consume, Consume, Consume, ToolCall, ToolCall, ToolCallEnd, Consume, Consume, EnterThinking, Thinking},
			},
			{
				name:   "replace analysis with final header",
				tokens: []string{"<|channel|>", "analysis", "<|channel|>", "final", "<|message|>", "answer"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Consume, Content},
			},
		})
	})
	t.Run("unknown channels", func(t *testing.T) {
		for _, channel := range []string{"justify", "confidence", "summary", "complete", "notification", "delegate_summary", "juicepred", "custom", "Analysis", "analysis_extra", "commentary", "commentary_extra", "final_extra", ""} {
			for _, ending := range []string{"<|end|>", "<|call|>"} {
				t.Run(channel+ending, func(t *testing.T) {
					s := New()
					checkTokens(t, s,
						[]string{"<|channel|>", channel, " to", "=", "functions", ".", "f", "<|constrain|>", "json", "<|message|>", "{}", "<think>", "</think>", ending},
						[]EventKind{Consume, Consume, Consume, Consume, Consume, Consume, Consume, Consume, Consume, Consume, Consume, Consume, Consume, Consume})
					checkTokens(t, s,
						[]string{"<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "answer"},
						[]EventKind{Consume, Consume, Consume, Consume, Consume, Content})
				})
			}
		}
	})
	t.Run("malformed tags", func(t *testing.T) {
		checkCases(t, []parseCase{
			{
				name:   "wrong start marker is literal content",
				tokens: []string{"<|strt|>", "assistant"},
				kinds:  []EventKind{Content, Content},
			},
			{
				name:   "partial start marker is not accumulated",
				tokens: []string{"<|sta", "rt|>", "assistant"},
				kinds:  []EventKind{Content, Content, Content},
			},
			{
				name:   "missing start marker leaves role as content",
				tokens: []string{"assistant", "<|channel|>", "final", "<|message|>", "answer"},
				kinds:  []EventKind{Content, Consume, Consume, Consume, Content},
			},
			{
				name:   "wrong channel marker stays in assistant header",
				tokens: []string{"<|start|>", "assistant", "<|chanel|>", "analysis", "<|message|>", "answer"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Consume, Content},
			},
			{
				name:   "partial channel marker is not accumulated",
				tokens: []string{"<|start|>", "assistant", "<|chan", "nel|>", "analysis", "<|message|>", "answer"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Consume, Consume, Content},
			},
			{
				name:   "missing channel marker does not select analysis",
				tokens: []string{"<|start|>", "assistant", "analysis", "<|message|>", "answer"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Content},
			},
			{
				name:   "missing message marker leaves body in header",
				tokens: []string{"<|channel|>", "analysis", "reason"},
				kinds:  []EventKind{Consume, Consume, Consume},
			},
			{
				name:   "empty channel",
				tokens: []string{"<|channel|>", "<|message|>", "hidden", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume},
			},
			{
				name:   "missing commentary suffix",
				tokens: []string{"<|channel|>", "comment", "<|message|>", "hidden", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Consume},
			},
			{
				name:   "incomplete commentary suffix is not accumulated",
				tokens: []string{"<|channel|>", "comment", "ar", "y", "<|message|>", "hidden", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Consume, Consume, Consume},
			},
			{
				name:   "partial analysis token is not accumulated",
				tokens: []string{"<|channel|>", "anal", "ysis", "<|message|>", "hidden", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Consume, Consume},
			},
			{
				name:   "partial final token is not accumulated",
				tokens: []string{"<|channel|>", "fin", "al", "<|message|>", "hidden", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Consume, Consume},
			},
			{
				name:   "wrong commentary suffix",
				tokens: []string{"<|channel|>", "comment", "ory", "<|message|>", "hidden", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Consume, Consume},
			},
			{
				name:   "unsupported role",
				tokens: []string{"<|start|>", "user", "<|message|>", "hidden", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Consume},
			},
			{
				name:   "role token requires exact match",
				tokens: []string{"<|start|>", "assistant_extra", "<|message|>", "hidden", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Consume},
			},
			{
				name:   "partial role token is not accumulated",
				tokens: []string{"<|start|>", "assis", "tant", "<|message|>", "hidden", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Consume, Consume},
			},
			{
				name:   "missing assistant role",
				tokens: []string{"<|start|>", "<|message|>", "hidden", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume},
			},
			{
				name:   "wrong message tag stays in header",
				tokens: []string{"<|channel|>", "analysis", "<|mesage|>", "reason"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume},
			},
			{
				name:   "partial message tag is not accumulated",
				tokens: []string{"<|channel|>", "final", "<|mess", "age|>", "hidden"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Consume},
			},
			{
				name:   "wrong end tag remains thinking",
				tokens: []string{"<|channel|>", "analysis", "<|message|>", "reason", "<|en|>", "more", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, EnterThinking, Thinking, Thinking, Thinking, LeaveThinking},
			},
			{
				name:   "partial end tag remains content",
				tokens: []string{"<|channel|>", "final", "<|message|>", "answer", "<|en", "d|>", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, Consume, Content, Content, Content, Consume},
			},
			{
				name:   "wrong call marker remains content",
				tokens: []string{"<|channel|>", "comment", "ary", "<|message|>", "{}", "<|cal|>", "<|call|>"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Content, Content, Consume},
			},
			{
				name:   "partial call marker is not accumulated",
				tokens: []string{"<|channel|>", "comment", "ary", "<|message|>", "{}", "<|ca", "ll|>", "<|call|>"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Content, Content, Content, Consume},
			},
			{
				name:   "duplicate channel marker",
				tokens: []string{"<|channel|>", "<|channel|>", "analysis", "<|message|>", "reason"},
				kinds:  []EventKind{Consume, Consume, Consume, EnterThinking, Thinking},
			},
			{
				name:   "duplicate analysis channel header",
				tokens: []string{"<|channel|>", "analysis", "<|channel|>", "analysis", "<|message|>", "reason"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, EnterThinking, Thinking},
			},
			{
				name:   "restart incomplete commentary header",
				tokens: []string{"<|channel|>", "comment", "<|channel|>", "analysis", "<|message|>", "reason"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, EnterThinking, Thinking},
			},
			{
				name:   "restart ignored header",
				tokens: []string{"<|channel|>", "custom", "<|channel|>", "final", "<|message|>", "answer"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Consume, Content},
			},
			{
				name:   "missing end before next content message",
				tokens: []string{"<|channel|>", "final", "<|message|>", "first", "<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "second"},
				kinds:  []EventKind{Consume, Consume, Consume, Content, Consume, Consume, Consume, Consume, Consume, Content},
			},
			{
				name:   "missing end before next ignored message",
				tokens: []string{"<|channel|>", "custom", "<|message|>", "hidden", "<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "answer"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Consume, Consume, Consume, Consume, Consume, Content},
			},
			{
				name:   "stray tags after message end",
				tokens: []string{"<|channel|>", "final", "<|message|>", "answer", "<|end|>", "<|end|>", "<|call|>", "<|message|>", "hidden"},
				kinds:  []EventKind{Consume, Consume, Consume, Content, Consume, Consume, Consume, Consume, Consume},
			},
		})
	})
	t.Run("next message recovery", func(t *testing.T) {
		checkCases(t, []parseCase{
			{
				name:   "wrong role without channel ends before valid analysis",
				tokens: []string{"<|start|>", "user", "<|message|>", "hidden", "<|end|>", "<|start|>", "assistant", "<|channel|>", "analysis", "<|message|>", "reason", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Consume, Consume, Consume, Consume, Consume, EnterThinking, Thinking, LeaveThinking},
			},
			{
				name:   "incomplete commentary ends before valid final",
				tokens: []string{"<|channel|>", "comment", "<|message|>", "hidden", "<|end|>", "<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "answer"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Consume, Consume, Consume, Consume, Consume, Consume, Content},
			},
		})
	})
	t.Run("truncated output", func(t *testing.T) {
		for _, tc := range []parseCase{
			{"analysis", []string{"<|start|>", "assistant", "<|channel|>", "analysis", "<|message|>", "reason", "<|end|>"}, []EventKind{Consume, Consume, Consume, Consume, EnterThinking, Thinking, LeaveThinking}},
			{"commentary call", []string{"<|channel|>", "comment", "ary", " to", "=", "functions", ".", "f", "<|constrain|>", "json", "<|message|>", "{", "}", "<|call|>"}, []EventKind{Consume, Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, ToolCall, Consume, Consume, ToolCallStart, ToolCall, ToolCall, ToolCallEnd}},
			{"final", []string{"<|channel|>", "final", "<|message|>", "answer", "<|end|>"}, []EventKind{Consume, Consume, Consume, Content, Consume}},
			{"ignored", []string{"<|channel|>", "custom", "<|message|>", "hidden", "<|end|>"}, []EventKind{Consume, Consume, Consume, Consume, Consume}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				for n := range len(tc.tokens) {
					t.Run("prefix "+strconv.Itoa(n), func(t *testing.T) {
						checkTokens(t, New(), tc.tokens[:n], tc.kinds[:n])
					})
				}
			})
		}
	})
	t.Run("end versus call", func(t *testing.T) {
		for _, tc := range []struct {
			name        string
			header      []string
			headerKinds []EventKind
			message     EventKind
			body        EventKind
			end         EventKind
			call        EventKind
		}{
			{"analysis", []string{"analysis"}, []EventKind{Consume}, EnterThinking, Thinking, LeaveThinking, LeaveThinking},
			{"commentary", []string{"comment", "ary"}, []EventKind{Consume, Consume}, Consume, Content, Consume, Consume},
			{"final", []string{"final"}, []EventKind{Consume}, Consume, Content, Consume, Consume},
			{"ignored", []string{"custom"}, []EventKind{Consume}, Consume, Consume, Consume, Consume},
		} {
			for _, ending := range []string{"<|end|>", "<|call|>"} {
				for _, body := range []string{"empty body", "nonempty body"} {
					t.Run(tc.name+"/"+ending+"/"+body, func(t *testing.T) {
						s := New()
						checkTokens(t, s, []string{"<|channel|>"}, []EventKind{Consume})
						checkTokens(t, s, tc.header, tc.headerKinds)
						checkTokens(t, s, []string{"<|message|>"}, []EventKind{tc.message})
						if body == "nonempty body" {
							checkTokens(t, s, []string{"body"}, []EventKind{tc.body})
						}
						want := tc.end
						if ending == "<|call|>" {
							want = tc.call
						}
						checkTokens(t, s, []string{ending}, []EventKind{want})
						checkTokens(t, s,
							[]string{"<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "answer"},
							[]EventKind{Consume, Consume, Consume, Consume, Consume, Content})
					})
				}
			}
		}
	})
}

// Recovery expectations are protocol rules, independent of the transition map.
func TestGPTOSSRecovery(t *testing.T) {
	t.Run("header control boundaries", func(t *testing.T) {
		prefixes := []struct {
			name     string
			tokens   []string
			kinds    []EventKind
			boundary EventKind
		}{
			{"await role", []string{"<|start|>"}, []EventKind{Consume}, Consume},
			{"assistant", []string{"<|start|>", "assistant"}, []EventKind{Consume, Consume}, Consume},
			{"assistant recipient", []string{"<|start|>", "assistant", " to", "=", "functions", ".", "f"}, []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, ToolCall}, ToolCallEnd},
			{"assistant awaiting equals", []string{"<|start|>", "assistant", " to"}, []EventKind{Consume, Consume, ToolCall}, ToolCallEnd},
			{"assistant awaiting recipient", []string{"<|start|>", "assistant", " to", "="}, []EventKind{Consume, Consume, ToolCall, ToolCall}, ToolCallEnd},
			{"analysis awaiting equals", []string{"<|channel|>", "analysis", " to"}, []EventKind{Consume, Consume, ToolCall}, ToolCallEnd},
			{"analysis awaiting recipient", []string{"<|channel|>", "analysis", " to", "="}, []EventKind{Consume, Consume, ToolCall, ToolCall}, ToolCallEnd},
			{"final awaiting equals", []string{"<|channel|>", "final", " to"}, []EventKind{Consume, Consume, ToolCall}, ToolCallEnd},
			{"final awaiting recipient", []string{"<|channel|>", "final", " to", "="}, []EventKind{Consume, Consume, ToolCall, ToolCall}, ToolCallEnd},
			{"tool channel", []string{"<|start|>", "assistant", " to", "=", "python", "<|channel|>"}, []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, Consume}, ToolCallEnd},
			{"tool comment suffix", []string{"<|start|>", "assistant", " to", "=", "browser", "<|channel|>", "comment"}, []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, Consume, Consume}, ToolCallEnd},
			{"tool ignored channel", []string{"<|start|>", "assistant", " to", "=", "functions", ".f", "<|channel|>", "custom"}, []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, Consume, Consume}, ToolCallEnd},
			{"channel-position tool header", []string{"<|channel|>", "analysis", " to", "=", "functions", ".f"}, []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall}, ToolCallEnd},
			{"channel", []string{"<|channel|>"}, []EventKind{Consume}, Consume},
			{"repeated channel", []string{"<|channel|>", "<|channel|>"}, []EventKind{Consume, Consume}, Consume},
			{"comment suffix", []string{"<|channel|>", "comment"}, []EventKind{Consume, Consume}, Consume},
			{"analysis", []string{"<|channel|>", "analysis"}, []EventKind{Consume, Consume}, Consume},
			{"analysis constrain", []string{"<|channel|>", "analysis", "<|constrain|>", "json"}, []EventKind{Consume, Consume, Consume, Consume}, Consume},
			{"final", []string{"<|channel|>", "final"}, []EventKind{Consume, Consume}, Consume},
			{"commentary", []string{"<|channel|>", "comment", "ary"}, []EventKind{Consume, Consume, Consume}, Consume},
			{"unknown channel", []string{"<|channel|>", "custom"}, []EventKind{Consume, Consume}, Consume},
			{"partial commentary", []string{"<|channel|>", "comment", "ar"}, []EventKind{Consume, Consume, Consume}, Consume},
			{"empty channel token", []string{"<|channel|>", ""}, []EventKind{Consume, Consume}, Consume},
			{"ignored role", []string{"<|start|>", "user"}, []EventKind{Consume, Consume}, Consume},
			{"ignored role channel", []string{"<|start|>", "tool", "<|channel|>", "final", " to"}, []EventKind{Consume, Consume, Consume, Consume, Consume}, Consume},
			{"empty role token", []string{"<|start|>", ""}, []EventKind{Consume, Consume}, Consume},
			{"tool role constraint marker", []string{"<|start|>", "assistant", " to", "=", "functions", ".f", "<|constrain|>"}, []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, Consume}, ToolCallEnd},
			{"tool role constraint type", []string{"<|start|>", "assistant", " to", "=", "functions", ".f", "<|constrain|>", "json", "header garbage"}, []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, Consume, Consume, Consume}, ToolCallEnd},
			{"tool constraint marker", []string{"<|channel|>", "analysis", " to", "=", "functions", ".f", "<|constrain|>"}, []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, Consume}, ToolCallEnd},
			{"tool constraint type", []string{"<|channel|>", "final", " to", "=", "functions", ".f", "<|constrain|>", "json", "header garbage"}, []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, Consume, Consume, Consume}, ToolCallEnd},
		}
		suffixes := []parseCase{
			{"analysis", []string{"assistant", "<|channel|>", "analysis", "<|message|>", "reason", "<|end|>"}, []EventKind{Consume, Consume, Consume, EnterThinking, Thinking, LeaveThinking}},
			{"commentary", []string{"assistant", "<|channel|>", "comment", "ary", "<|message|>", "update", "<|call|>"}, []EventKind{Consume, Consume, Consume, Consume, Consume, Content, Consume}},
			{"final", []string{"assistant", "<|channel|>", "final", "<|message|>", "answer", "<|end|>"}, []EventKind{Consume, Consume, Consume, Consume, Content, Consume}},
		}
		for _, prefix := range prefixes {
			for _, boundary := range []string{"<|start|>", "<|end|>", "<|call|>"} {
				for _, suffix := range suffixes {
					t.Run(prefix.name+"/"+boundary+"/"+suffix.name, func(t *testing.T) {
						s := New()
						checkTokens(t, s, prefix.tokens, prefix.kinds)
						checkTokens(t, s, []string{boundary}, []EventKind{prefix.boundary})
						wantState := stateGPTOSSStart
						if boundary != "<|start|>" {
							wantState = stateGPTOSSEnd
						}
						if s.state != wantState {
							t.Errorf("header %q + %q: next state = %v, want %v", prefix.tokens, boundary, s.state, wantState)
						}
						if boundary != "<|start|>" {
							// Termination does not accept another header without a new start.
							checkTokens(t, s,
								[]string{"<|end|>", "<|call|>", "assistant", "<|channel|>", "<|message|>", ""},
								[]EventKind{Consume, Consume, Consume, Consume, Consume, Consume})
							checkTokens(t, s, []string{"<|start|>"}, []EventKind{Consume})
						}
						checkTokens(t, s, suffix.tokens, suffix.kinds)
					})
				}
			}
		}
	})
	t.Run("non assistant role cannot select channels", func(t *testing.T) {
		roles := []struct {
			name   string
			tokens []string
		}{
			{"user", []string{"user"}},
			{"system", []string{"system"}},
			{"developer", []string{"developer"}},
			{"tool", []string{"tool"}},
			{"browser", []string{"browser"}},
			{"python", []string{"python"}},
			{"function name", []string{"functions.f"}},
			{"unknown", []string{"custom_role"}},
			{"case mismatch", []string{"Assistant"}},
			{"role suffix", []string{"assistant_extra"}},
			{"role fragments", []string{"assis", "tant"}},
			{"empty role", []string{""}},
		}
		headers := []struct {
			name   string
			tokens []string
		}{
			{"analysis", []string{"<|channel|>", "analysis"}},
			{"commentary", []string{"<|channel|>", "comment", "ary"}},
			{"final", []string{"<|channel|>", "final"}},
			{"reordered repeated channels", []string{"assistant", "<|channel|>", "custom", "<|channel|>", "analysis", "<|channel|>", "final"}},
		}
		for _, role := range roles {
			for _, header := range headers {
				for _, boundary := range []string{"<|end|>", "<|call|>", "<|start|>"} {
					t.Run(role.name+"/"+header.name+"/"+boundary, func(t *testing.T) {
						tokens := append([]string{"<|start|>"}, role.tokens...)
						tokens = append(tokens, header.tokens...)
						tokens = append(tokens, " to", "=", "functions", ".", "f", "<|constrain|>", "json", "<|message|>", "hidden", "!", "<think>", "<channel|>", boundary)
						// Consume is zero; no ignored-role token may leak to any output stream.
						s := New()
						checkTokens(t, s, tokens, make([]EventKind, len(tokens)))
						wantState := stateGPTOSSStart
						if boundary != "<|start|>" {
							wantState = stateGPTOSSEnd
						}
						if s.state != wantState {
							t.Errorf("ignored role + %q: next state = %v, want %v", boundary, s.state, wantState)
						}
						if boundary != "<|start|>" {
							checkTokens(t, s, []string{"<|start|>"}, []EventKind{Consume})
						}
						checkTokens(t, s,
							[]string{"assistant", "<|channel|>", "final", "<|message|>", "visible", "<|end|>"},
							[]EventKind{Consume, Consume, Consume, Consume, Content, Consume})
					})
				}
			}
		}
	})
	t.Run("thinking start closes and reparses", func(t *testing.T) {
		for _, body := range []parseCase{
			{"empty", nil, nil},
			{"punctuation and foreign markers", []string{"", "reason", ".", "<think>", "</think>", "<|channel>", "<channel|>"}, []EventKind{Thinking, Thinking, Thinking, Thinking, Thinking, Thinking, Thinking}},
		} {
			for _, suffix := range []parseCase{
				{"analysis", []string{"assistant", "<|channel|>", "analysis", "<|message|>", "second", "<|call|>"}, []EventKind{Consume, Consume, Consume, EnterThinking, Thinking, LeaveThinking}},
				{"commentary", []string{"assistant", "<|channel|>", "comment", "ary", "<|message|>", "update", "<|end|>"}, []EventKind{Consume, Consume, Consume, Consume, Consume, Content, Consume}},
				{"final", []string{"assistant", "<|channel|>", "final", "<|message|>", "answer", "<|end|>"}, []EventKind{Consume, Consume, Consume, Consume, Content, Consume}},
				{"unknown", []string{"assistant", "<|channel|>", "custom", "<|message|>", "hidden", "<|end|>"}, []EventKind{Consume, Consume, Consume, Consume, Consume, Consume}},
			} {
				t.Run(body.name+"/"+suffix.name, func(t *testing.T) {
					s := New()
					checkTokens(t, s, []string{"<|channel|>", "analysis", "<|message|>"}, []EventKind{Consume, Consume, EnterThinking})
					checkTokens(t, s, body.tokens, body.kinds)
					checkTokens(t, s, []string{"<|start|>"}, []EventKind{LeaveThinking})
					if s.state != stateGPTOSSStart {
						t.Errorf("thinking + start: next state = %v, want %v", s.state, stateGPTOSSStart)
					}
					checkTokens(t, s, suffix.tokens, suffix.kinds)
					checkTokens(t, s,
						[]string{"<|start|>", "assistant", "<|message|>", "visible", "<|end|>"},
						[]EventKind{Consume, Consume, Consume, Content, Consume})
				})
			}
		}
	})
	t.Run("multiple irregular recoveries", func(t *testing.T) {
		checkCases(t, []parseCase{
			{
				name: "mixed ignored and supported messages with body punctuation",
				tokens: []string{
					"<|start|>", "browser", "<|message|>", "hidden", "<|end|>",
					"<|start|>", "assistant", "<|channel|>", "custom", "<|message|>", "hidden", "<|call|>",
					"<|start|>", "assistant", "<|channel|>", "analysis", "<|message|>", "", "why", "?", "<|end|>",
					"<|start|>", "assistant", "<|channel|>", "comment", "ary", "<|message|>", "checking", "<|call|>",
					"<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "yes", ".", "</think>", "<channel|>", "<|end|>",
				},
				kinds: []EventKind{
					Consume, Consume, Consume, Consume, Consume,
					Consume, Consume, Consume, Consume, Consume, Consume, Consume,
					Consume, Consume, Consume, Consume, EnterThinking, Thinking, Thinking, Thinking, LeaveThinking,
					Consume, Consume, Consume, Consume, Consume, Consume, Content, Consume,
					Consume, Consume, Consume, Consume, Consume, Content, Content, Content, Content, Consume,
				},
			},
			{
				name: "repeated starts and abandoned headers before thinking restart",
				tokens: []string{
					"<|start|>", "<|start|>", "assistant", "<|channel|>", "comment", "<|call|>",
					"<|end|>", "<|call|>", "<|start|>", "assistant", "<|channel|>", "analysis", "<|message|>", "first",
					"<|start|>", "assistant", "<|channel|>", "custom", "<|message|>", "hidden",
					"<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "answer", "!", "<|end|>",
				},
				kinds: []EventKind{
					Consume, Consume, Consume, Consume, Consume, Consume,
					Consume, Consume, Consume, Consume, Consume, Consume, EnterThinking, Thinking,
					LeaveThinking, Consume, Consume, Consume, Consume, Consume,
					Consume, Consume, Consume, Consume, Consume, Content, Content, Consume,
				},
			},
			{
				name: "ignored role then unknown header call then content restart",
				tokens: []string{
					"<|start|>", "developer", "<|channel|>", "analysis", "<|message|>", "hidden", "<|end|>",
					"<|start|>", "assistant", "<|channel|>", "custom", "<|call|>",
					"<|start|>", "assistant", "<|channel|>", "comment", "ary", "<|message|>", "checking",
					"<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "", "done", ".", "</think>", "<channel|>", "<|end|>",
				},
				kinds: []EventKind{
					Consume, Consume, Consume, Consume, Consume, Consume, Consume,
					Consume, Consume, Consume, Consume, Consume,
					Consume, Consume, Consume, Consume, Consume, Consume, Content,
					Consume, Consume, Consume, Consume, Consume, Content, Content, Content, Content, Content, Consume,
				},
			},
		})
	})
	t.Run("truncated restarted thinking stream", func(t *testing.T) {
		tokens := []string{"<|channel|>", "analysis", "<|message|>", "reason", "<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "answer", "<|end|>"}
		kinds := []EventKind{Consume, Consume, EnterThinking, Thinking, LeaveThinking, Consume, Consume, Consume, Consume, Content, Consume}
		for n := range tokens {
			t.Run("prefix "+strconv.Itoa(n), func(t *testing.T) {
				checkTokens(t, New(), tokens[:n], kinds[:n])
			})
		}
	})
}

// Recipient routing is independent of whether raw syntax can be parsed as a function call.
func TestGPTOSSRecipientRouting(t *testing.T) {
	channels := []struct {
		name   string
		tokens []string
	}{
		{"analysis", []string{"analysis"}},
		{"commentary", []string{"comment", "ary"}},
		{"final", []string{"final"}},
	}
	recipients := []struct {
		name   string
		tokens []string
		tool   bool
	}{
		{"functions f", []string{"functions", ".f"}, true},
		{"functions get_weather", []string{"functions", ".get", "_weather"}, true},
		{"browser", []string{"browser"}, true},
		{"python", []string{"python"}, true},
		{"other", []string{"other"}, true},
		{"assistant", []string{"assistant"}, false},
	}
	for _, position := range []string{"role", "channel"} {
		for _, channel := range channels {
			for _, recipient := range recipients {
				for _, ending := range []string{"<|end|>", "<|call|>"} {
					t.Run(position+"/"+channel.name+"/"+recipient.name+"/"+ending, func(t *testing.T) {
						tokens := []string{"<|start|>", "assistant"}
						kinds := []EventKind{Consume, Consume}
						addChannel := func() {
							tokens = append(tokens, "<|channel|>")
							tokens = append(tokens, channel.tokens...)
							channelKinds := make([]EventKind, 1+len(channel.tokens))
							kinds = append(kinds, channelKinds...)
						}
						if position == "channel" {
							addChannel()
						}
						tokens = append(tokens, " to", "=")
						kinds = append(kinds, ToolCall, ToolCall)
						for _, token := range recipient.tokens {
							tokens = append(tokens, token)
							kind := ToolCallEnd
							if recipient.tool {
								kind = ToolCall
							}
							kinds = append(kinds, kind)
						}
						if position == "role" {
							addChannel()
						}
						// Constraint markers and types are consumed, even on the tool route.
						tokens = append(tokens, " ", "<|constrain|>", "json", "<|message|>")
						message, body, end := Consume, Content, Consume
						header := Consume
						if recipient.tool {
							header, message, body, end = ToolCall, ToolCallStart, ToolCall, ToolCallEnd
						} else if channel.name == "analysis" {
							message, body, end = EnterThinking, Thinking, LeaveThinking
						}
						kinds = append(kinds, header, Consume, Consume, message)
						for _, token := range []string{"", "{", `"city"`, ":", `"Paris"`, ",", `"units":"C"`, "}", "!"} {
							tokens = append(tokens, token)
							kinds = append(kinds, body)
						}
						tokens = append(tokens, ending)
						kinds = append(kinds, end)
						for n := 0; n <= len(tokens); n++ {
							t.Run("prefix "+strconv.Itoa(n), func(t *testing.T) {
								checkTokens(t, New(), tokens[:n], kinds[:n])
							})
						}
					})
				}
			}
		}
	}
	t.Run("ordinary preamble and structured final", func(t *testing.T) {
		checkCases(t, []parseCase{
			{
				name:   "commentary preamble without recipient",
				tokens: []string{"<|start|>", "assistant", "<|channel|>", "comment", "ary", "ordinary header text", "<|message|>", "Let me check.", "<|call|>"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Consume, Consume, Consume, Content, Consume},
			},
			{
				name:   "structured final without recipient",
				tokens: []string{"<|start|>", "assistant", "<|channel|>", "final", " ", "<|constrain|>", "json", "<|message|>", `{"answer":42}`, "<|end|>"},
				kinds:  []EventKind{Consume, Consume, Consume, Consume, Consume, Consume, Consume, Consume, Content, Consume},
			},
			{
				name:   "recipient requires exact assistant token",
				tokens: []string{"<|start|>", "assistant", " to", "=", "assistant_extra", "<|channel|>", "analysis", "<|message|>", "raw", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, Consume, Consume, ToolCallStart, ToolCall, ToolCallEnd},
			},
			{
				name:   "recipient fragments are not accumulated",
				tokens: []string{"<|start|>", "assistant", " to", "=", "assis", "tant", "<|message|>", "raw", "<|call|>"},
				kinds:  []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, ToolCallStart, ToolCall, ToolCallEnd},
			},
		})
	})
	t.Run("channel reselection ends incomplete recipient", func(t *testing.T) {
		for _, position := range []string{"role", "channel"} {
			for _, recipient := range recipients {
				for _, first := range []struct {
					name   string
					tokens []string
				}{
					{"supported", []string{"comment", "ary"}},
					{"unknown", []string{"custom"}},
					{"partial", []string{"comment"}},
				} {
					// A channel-position recipient requires a supported initial channel.
					if position == "channel" && first.name != "supported" {
						continue
					}
					for _, last := range channels {
						t.Run(position+"/"+recipient.name+"/"+first.name+"/"+last.name, func(t *testing.T) {
							s := New()
							checkTokens(t, s, []string{"<|start|>", "assistant"}, []EventKind{Consume, Consume})
							selectFirst := func(kind EventKind) {
								checkTokens(t, s, []string{"<|channel|>"}, []EventKind{kind})
								checkTokens(t, s, first.tokens, make([]EventKind, len(first.tokens)))
							}
							if position == "channel" {
								selectFirst(Consume)
							}
							checkTokens(t, s, []string{" to", "="}, []EventKind{ToolCall, ToolCall})
							kind := ToolCallEnd
							if recipient.tool {
								kind = ToolCall
							}
							for _, token := range recipient.tokens {
								checkTokens(t, s, []string{token}, []EventKind{kind})
							}
							if position == "role" {
								selectFirst(Consume)
							}
							// Any later channel separator ends the old candidate, even
							// when channels repeat or an unsupported selection intervenes.
							tokens := []string{"<|channel|>", "analysis", "<|channel|>", "analysis", "<|channel|>", "custom", "<|channel|>", "comment", "ary", "<|channel|>"}
							tokens = append(tokens, last.tokens...)
							kinds := make([]EventKind, len(tokens))
							if recipient.tool {
								kinds[0] = ToolCallEnd
							}
							checkTokens(t, s, tokens, kinds)
							message, body, end := Consume, Content, Consume
							if last.name == "analysis" {
								message, body, end = EnterThinking, Thinking, LeaveThinking
							}
							checkTokens(t, s, []string{"<|message|>", "ordinary", "<|end|>"}, []EventKind{message, body, end})
						})
					}
				}
			}
		}
	})
	t.Run("reselected channel accepts fresh recipient", func(t *testing.T) {
		checkCases(t, []parseCase{
			{
				name:   "old channel recipient replaced by fresh function",
				tokens: []string{"<|channel|>", "analysis", " to", "=", "functions", ".f", "<|channel|>", "comment", "ary", " to", "=", "functions", ".get", "_weather", "<|message|>", "{}", "<|call|>"},
				kinds:  []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, ToolCallEnd, Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, ToolCall, ToolCallStart, ToolCall, ToolCallEnd},
			},
		})
	})
	t.Run("unsupported channel and wrong role do not leak", func(t *testing.T) {
		for _, recipient := range recipients {
			for _, prefix := range []parseCase{
				{"unknown channel first", []string{"<|start|>", "assistant", "<|channel|>", "custom"}, []EventKind{Consume, Consume, Consume, Consume}},
				{"wrong role", []string{"<|start|>", "user", "<|channel|>", "analysis"}, []EventKind{Consume, Consume, Consume, Consume}},
			} {
				t.Run(recipient.name+"/"+prefix.name, func(t *testing.T) {
					s := New()
					checkTokens(t, s, prefix.tokens, prefix.kinds)
					tokens := append([]string{" to", "="}, recipient.tokens...)
					tokens = append(tokens, "<|message|>", `{"raw":true}`, "<think>", "</think>", "<|call|>")
					checkTokens(t, s, tokens, make([]EventKind, len(tokens)))
					checkTokens(t, s, []string{"<|start|>", "assistant", "<|message|>", "visible", "<|end|>"}, []EventKind{Consume, Consume, Consume, Content, Consume})
				})
			}
			if !recipient.tool {
				continue
			}
			t.Run(recipient.name+"/unknown channel after recipient", func(t *testing.T) {
				s := New()
				checkTokens(t, s, []string{"<|start|>", "assistant", " to", "="}, []EventKind{Consume, Consume, ToolCall, ToolCall})
				for _, token := range recipient.tokens {
					checkTokens(t, s, []string{token}, []EventKind{ToolCall})
				}
				checkTokens(t, s, []string{"<|channel|>"}, []EventKind{Consume})
				tokens := []string{"custom", " to", "=", "functions", ".f", "<|message|>", "hidden", "<|end|>"}
				checkTokens(t, s, tokens, []EventKind{Consume, Consume, Consume, Consume, Consume, ToolCallEnd, Consume, Consume})
			})
		}
	})
	t.Run("tool body restart and multiple messages", func(t *testing.T) {
		checkCases(t, []parseCase{
			{
				name:   "complete JSON body without end before fresh final",
				tokens: []string{"<|channel|>", "comment", "ary", " to", "=", "functions", ".f", "<|message|>", `{"x":1}`, "<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "answer", "<|end|>"},
				kinds:  []EventKind{Consume, Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, ToolCallStart, ToolCall, ToolCallEnd, Consume, Consume, Consume, Consume, Content, Consume},
			},
		})
		for _, boundary := range []string{"<|start|>", "<|end|>", "<|call|>"} {
			t.Run(boundary, func(t *testing.T) {
				s := New()
				checkTokens(t, s,
					[]string{"<|start|>", "assistant", " to", "=", "python", "<|channel|>", "analysis", "<|message|>", "print", "(", "1", ")"},
					[]EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, Consume, Consume, ToolCallStart, ToolCall, ToolCall, ToolCall, ToolCall})
				checkTokens(t, s, []string{boundary}, []EventKind{ToolCallEnd})
				if boundary != "<|start|>" {
					checkTokens(t, s, []string{"<|start|>"}, []EventKind{Consume})
				}
				checkTokens(t, s,
					[]string{"assistant", "<|channel|>", "final", "<|message|>", "answer", "<|end|>", "<|start|>", "assistant", "<|channel|>", "comment", "ary", " to", "=", "functions", ".get", "_weather", "<|message|>", "{}", "<|call|>"},
					[]EventKind{Consume, Consume, Consume, Consume, Content, Consume, Consume, Consume, Consume, Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, ToolCall, ToolCallStart, ToolCall, ToolCallEnd})
			})
		}
	})
}

func TestGPTOSSToolConstraints(t *testing.T) {
	for _, tc := range []struct {
		name   string
		prefix []string
		kinds  []EventKind
		role   bool
	}{
		{"role", []string{"<|start|>", "assistant", " to", "=", "functions", ".f"}, []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall}, true},
		{"channel", []string{"<|channel|>", "analysis", " to", "=", "functions", ".f"}, []EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, ending := range []string{"<|end|>", "<|call|>"} {
				t.Run("body/"+ending, func(t *testing.T) {
					s := New()
					checkTokens(t, s, tc.prefix, tc.kinds)
					checkTokens(t, s, []string{"<|constrain|>", "json", "header garbage", "<|constrain|>", "<|message|>", "{}", ending},
						[]EventKind{Consume, Consume, Consume, Consume, ToolCallStart, ToolCall, ToolCallEnd})
				})
			}
			t.Run("channel selection", func(t *testing.T) {
				s := New()
				checkTokens(t, s, tc.prefix, tc.kinds)
				checkTokens(t, s, []string{"<|constrain|>", "json", "header garbage"}, []EventKind{Consume, Consume, Consume})
				if tc.role {
					checkTokens(t, s, []string{"<|channel|>", "analysis", "<|message|>", "{}", "<|end|>"},
						[]EventKind{Consume, Consume, ToolCallStart, ToolCall, ToolCallEnd})
				} else {
					checkTokens(t, s, []string{"<|channel|>", "analysis", "<|message|>", "reason", "<|end|>"},
						[]EventKind{ToolCallEnd, Consume, EnterThinking, Thinking, LeaveThinking})
				}
			})
		})
	}
	for _, st := range []state{stateGPTOSSToolConstraint, stateGPTOSSToolRoleConstraint} {
		s := &Splitter{state: st}
		if got := s.Feed("header garbage"); got != (Event{Kind: Consume, Text: "header garbage"}) || s.state != st {
			t.Errorf("constraint state %v fallback = %+v, next = %v, want Consume with input and unchanged state", st, got, s.state)
		}
	}
}

// ff_log.txt records full_text, not token callbacks. These sanitized vectors
// reproduce its message structure using synthetic one-token protocol inputs.
func TestGPTOSSLogStructures(t *testing.T) {
	cases := []parseCase{
		{
			name:   "headerless analysis then final",
			tokens: []string{"<|channel|>", "analysis", "<|message|>", "Consider options.", "<|end|>", "<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "Updated answer."},
			kinds:  []EventKind{Consume, Consume, EnterThinking, Thinking, LeaveThinking, Consume, Consume, Consume, Consume, Consume, Content},
		},
		{
			name:   "analysis then constrained commentary call at EOF",
			tokens: []string{"<|channel|>", "analysis", "<|message|>", "Consider options.", "<|end|>", "<|start|>", "assistant", "<|channel|>", "comment", "ary", " to", "=", "functions", ".f", " ", "<|constrain|>", "json", "<|message|>", `{"topic":""}`},
			kinds:  []EventKind{Consume, Consume, EnterThinking, Thinking, LeaveThinking, Consume, Consume, Consume, Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, ToolCall, Consume, Consume, ToolCallStart, ToolCall},
		},
		{
			name:   "incomplete tool header ended before second thinking body",
			tokens: []string{"<|channel|>", "analysis", "<|message|>", "Consider options.", "<|end|>", "<|start|>", "assistant", "<|channel|>", "comment", "ary", " to", "=", "functions", ".f", "<|channel|>", "comment", "ary", "<|channel|>", "analysis", "<|message|>", "Need an indoor option.", "<|end|>", "<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "Updated answer."},
			kinds:  []EventKind{Consume, Consume, EnterThinking, Thinking, LeaveThinking, Consume, Consume, Consume, Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, ToolCallEnd, Consume, Consume, Consume, Consume, EnterThinking, Thinking, LeaveThinking, Consume, Consume, Consume, Consume, Consume, Content},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for n := 0; n <= len(tc.tokens); n++ {
				t.Run("prefix "+strconv.Itoa(n), func(t *testing.T) {
					checkTokens(t, New(), tc.tokens[:n], tc.kinds[:n])
				})
			}
		})
	}
}

// Gemma4: thought headers and model-specific body delimiters.
func TestGemma4Parsing(t *testing.T) {
	checkCases(t, []parseCase{
		{
			name:   "thinking and answer",
			tokens: []string{"<|channel>", "thought", "\n", "reason", "<channel|>", "answer"},
			kinds:  []EventKind{Consume, Consume, EnterThinking, Thinking, LeaveThinking, Content},
		},
		{
			name:   "empty thought header",
			tokens: []string{"<|channel>", "thought", "<channel|>", "answer"},
			kinds:  []EventKind{Consume, Consume, Consume, Content},
		},
		{
			name:   "empty thinking body",
			tokens: []string{"<|channel>", "thought", "\n", "<channel|>", "answer"},
			kinds:  []EventKind{Consume, Consume, EnterThinking, LeaveThinking, Content},
		},
		{
			name:   "unterminated thinking",
			tokens: []string{"<|channel>", "thought", "\n", "reason"},
			kinds:  []EventKind{Consume, Consume, EnterThinking, Thinking},
		},
		{
			name:   "missing body delimiter",
			tokens: []string{"<|channel>", "thought", "hidden"},
			kinds:  []EventKind{Consume, Consume, Consume},
		},
		{
			name:   "foreign markers stay in their body",
			tokens: []string{"<|channel>", "thought", "\n", "<think>", "</think>", "<|start|>", "<|channel|>", "<|end|>", "<|call|>", "<channel|>", "<think>", "</think>", "<|start|>", "<|channel|>", "<|end|>", "<|call|>"},
			kinds:  []EventKind{Consume, Consume, EnterThinking, Thinking, Thinking, Thinking, Thinking, Thinking, Thinking, LeaveThinking, Content, Content, Content, Content, Content, Content},
		},
		{
			name:   "tool syntax remains content",
			tokens: []string{"<|channel>", "thought", "<channel|>", "<|tool_call>", "call:go{}", "<tool_call|>"},
			kinds:  []EventKind{Consume, Consume, Consume, Content, Content, Content},
		},
	})
}

func TestGPTOSSAwaitingEqualsEndsAbandonedRecipient(t *testing.T) {
	prefixes := []struct {
		name   string
		tokens []string
		kinds  []EventKind
		body   EventKind
	}{
		{"role", []string{"<|start|>", "assistant", " to"}, []EventKind{Consume, Consume, ToolCall}, ToolCallEnd},
		{"analysis", []string{"<|channel|>", "analysis", " to"}, []EventKind{Consume, Consume, ToolCall}, EnterThinking},
		{"final", []string{"<|channel|>", "final", " to"}, []EventKind{Consume, Consume, ToolCall}, ToolCallEnd},
		{"commentary", []string{"<|channel|>", "comment", "ary", " to"}, []EventKind{Consume, Consume, Consume, ToolCall}, ToolCallEnd},
	}
	for _, prefix := range prefixes {
		for _, exit := range []struct {
			name   string
			tokens []string
			kinds  []EventKind
		}{
			{"unmatched then end", []string{"x", "<|end|>"}, []EventKind{ToolCallEnd, Consume}},
			{"empty ordinary body", []string{"<|message|>", "<|end|>"}, []EventKind{prefix.body, Consume}},
		} {
			t.Run(prefix.name+"/"+exit.name, func(t *testing.T) {
				s := New()
				checkTokens(t, s, prefix.tokens, prefix.kinds)
				if prefix.body == EnterThinking && exit.name == "empty ordinary body" {
					exit.kinds = []EventKind{EnterThinking, LeaveThinking}
				}
				checkTokens(t, s, exit.tokens, exit.kinds)
				checkTokens(t, s,
					[]string{"<|start|>", "assistant", " to", "=", "functions", ".f", "<|channel|>", "analysis", "<|message|>", "{}", "<|call|>"},
					[]EventKind{Consume, Consume, ToolCall, ToolCall, ToolCall, ToolCall, Consume, Consume, ToolCallStart, ToolCall, ToolCallEnd})
			})
		}
	}
}

func TestRestoreOpener(t *testing.T) {
	tokens := []string{"reason", "</think>", "answer"}
	for _, tc := range []struct {
		name, prompt string
		pieces       []string
		kinds        []EventKind
	}{
		{"open", "assistant\n<think>", []string{"<think>", "reason", "</think>", "answer"},
			[]EventKind{EnterThinking, Thinking, LeaveThinking, Content}},
		{"open with newline", "assistant\n<think>\n", []string{"<think>", "\n", "reason", "</think>", "answer"},
			[]EventKind{EnterThinking, Thinking, Thinking, LeaveThinking, Content}},
		{"closed", "assistant\n<think>\n\n</think>\n\n", tokens, []EventKind{Content, Content, Content}},
		{"no thinking", "assistant\n", tokens, []EventKind{Content, Content, Content}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := New()
			var pieces []string
			var kinds []EventKind
			onToken := RestoreOpener(tc.prompt, func(piece string) bool {
				pieces = append(pieces, piece)
				kinds = append(kinds, s.Feed(piece).Kind)
				return true
			})
			for _, token := range tokens {
				onToken(token)
			}
			if !slices.Equal(pieces, tc.pieces) {
				t.Errorf("pieces = %q, want %q", pieces, tc.pieces)
			}
			if !slices.Equal(kinds, tc.kinds) {
				t.Errorf("kinds = %v, want %v", kinds, tc.kinds)
			}
		})
	}
}
