// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// Package thinkfsm splits a model's token stream into reasoning ("thinking")
// and normal content. Markers (`<think>`, gpt-oss `<|channel|>`, gemma
// `<|channel>`) arrive as single tokens, so it must be fed one token at a time
// — the same tags concatenated into a text blob would not match. Both terminal
// rendering (geniex infer/run) and reasoning_content separation (geniex serve)
// drive this one table.
package thinkfsm

type state int

const (
	stateAssistant state = iota // init state
	stateThink                  // reasoning
	stateNormal

	// gpt-oss channels
	stateStart
	stateChannel
	stateAnalysis
	stateFinal
	stateEnd

	// gemma channels
	stateGemmaChannel
	stateGemmaChannelThought
)

var reasoningStates = map[state]bool{
	stateThink: true,
}

type Boundary int

const (
	NoBoundary Boundary = iota
	EnterReasoning
	ExitReasoning
)

type transition struct {
	next     state
	boundary Boundary
	// block spans its own lines (channel-style), vs the inline <think>...</think>
	// pair. Terminal rendering uses it to pick decoration; the server ignores it.
	block bool
}

type key struct {
	s     state
	token string
}

// transitions is the single source of truth for think-token handling. Any
// (state, token) absent here is an ordinary token, classified by reasoningStates.
var transitions = map[key]transition{
	// normal <think>...</think>
	{stateAssistant, "<think>"}: {next: stateThink, boundary: EnterReasoning},
	{stateThink, "</think>"}:    {next: stateNormal, boundary: ExitReasoning},

	// gpt-oss
	{stateAssistant, "<|channel|>"}: {next: stateChannel},
	{stateChannel, "analysis"}:      {next: stateAnalysis},
	{stateChannel, "final"}:         {next: stateFinal},
	{stateAnalysis, "<|message|>"}:  {next: stateThink, boundary: EnterReasoning, block: true},
	{stateFinal, "<|message|>"}:     {next: stateNormal},
	{stateThink, "<|end|>"}:         {next: stateEnd, boundary: ExitReasoning, block: true},
	{stateNormal, "<|end|>"}:        {next: stateEnd},
	{stateEnd, "<|start|>"}:         {next: stateStart},
	{stateStart, "assistant"}:       {next: stateAssistant},

	// gemma4
	{stateAssistant, "<|channel>"}:   {next: stateGemmaChannel},
	{stateGemmaChannel, "thought"}:   {next: stateGemmaChannelThought},
	{stateGemmaChannelThought, "\n"}: {next: stateThink, boundary: EnterReasoning, block: true},
	{stateThink, "<channel|>"}:       {next: stateNormal, boundary: ExitReasoning, block: true},
}

// Event routes one token. A consumed marker (e.g. the <think> tag) belongs to
// no output stream; otherwise Text goes to the reasoning or content stream.
type Event struct {
	Consumed  bool
	Boundary  Boundary // only when Consumed
	Block     bool     // only when Consumed
	Text      string   // only when !Consumed
	Reasoning bool     // only when !Consumed
}

// Splitter tracks one generation's think state. Create one per generation with
// New; not safe for concurrent use.
type Splitter struct {
	state state
}

func New() *Splitter {
	return &Splitter{state: stateAssistant}
}

func (s *Splitter) Feed(token string) Event {
	if t, ok := transitions[key{s.state, token}]; ok {
		s.state = t.next
		return Event{Consumed: true, Boundary: t.boundary, Block: t.block}
	}
	return Event{Text: token, Reasoning: reasoningStates[s.state]}
}
