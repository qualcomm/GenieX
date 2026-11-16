// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// Package thinkfsm routes one model token at a time. It does not split or
// accumulate text; callers must preserve token boundaries.
package thinkfsm

import "strings"

type state int

const (
	stateAssistant state = iota // init state
	stateThink
	stateNormal
	stateMinistralThink

	// gpt-oss
	stateGPTOSSStart
	stateGPTOSSAssistantHeader
	stateGPTOSSAssistantTo
	stateGPTOSSAssistantRecipient
	stateGPTOSSToolRoleHeader
	stateGPTOSSToolRoleConstraint
	stateGPTOSSChannel
	stateGPTOSSChannelComment
	stateGPTOSSAnalysis
	stateGPTOSSAnalysisTo
	stateGPTOSSAnalysisRecipient
	stateGPTOSSFinal
	stateGPTOSSFinalTo
	stateGPTOSSFinalRecipient
	stateGPTOSSToolHeader
	stateGPTOSSToolConstraint
	stateGPTOSSToolChannel
	stateGPTOSSToolChannelComment
	stateGPTOSSToolIgnoredHeader
	stateGPTOSSIgnoredHeader
	stateGPTOSSIgnoredRoleHeader
	stateGPTOSSThink
	stateGPTOSSNormal
	stateGPTOSSTool
	stateGPTOSSIgnored
	stateGPTOSSEnd

	// gemma4
	stateGemmaChannel
	stateGemmaChannelThought
	stateGemmaThink
	stateGemmaNormal
)

type EventKind int

const (
	Consume EventKind = iota
	EnterThinking
	Thinking
	LeaveThinking
	Content
	ToolCall
	ToolCallStart
	ToolCallEnd
)

// ToolCall text is scanner input, not an already parsed call.
type Event struct {
	Kind EventKind
	Text string
}

type transition struct {
	next state
	Event
}

type key struct {
	s     state
	token string
}

// An empty token key is the fallback for that state.
var transitions = func() map[key]transition {
	ts := map[key]transition{
		// common
		{stateAssistant, "<think>"}: {stateThink, Event{Kind: EnterThinking}},
		{stateAssistant, ""}:        {stateAssistant, Event{Kind: Content}},
		{stateThink, "</think>"}:    {stateNormal, Event{Kind: LeaveThinking}},
		{stateThink, ""}:            {stateThink, Event{Kind: Thinking}},
		{stateNormal, ""}:           {stateNormal, Event{Kind: Content}},

		{stateAssistant, "[THINK]"}:       {stateMinistralThink, Event{Kind: EnterThinking}},
		{stateMinistralThink, "[/THINK]"}: {stateNormal, Event{Kind: LeaveThinking}},
		{stateMinistralThink, ""}:         {stateMinistralThink, Event{Kind: Thinking}},

		// gpt-oss message start
		{stateAssistant, "<|start|>"}:   {stateGPTOSSStart, Event{}},
		{stateAssistant, "<|channel|>"}: {stateGPTOSSChannel, Event{}},
		{stateGPTOSSStart, "assistant"}: {stateGPTOSSAssistantHeader, Event{}},
		{stateGPTOSSStart, ""}:          {stateGPTOSSIgnoredRoleHeader, Event{}},
		// gpt-oss assistant header
		{stateGPTOSSAssistantHeader, "<|channel|>"}: {stateGPTOSSChannel, Event{}},
		{stateGPTOSSAssistantHeader, "<|message|>"}: {stateGPTOSSNormal, Event{}},
		{stateGPTOSSAssistantHeader, " to"}:         {stateGPTOSSAssistantTo, Event{ToolCall, " to"}},
		{stateGPTOSSAssistantHeader, ""}:            {stateGPTOSSAssistantHeader, Event{}},
		// gpt-oss assistant recipient
		{stateGPTOSSAssistantTo, "="}:                  {stateGPTOSSAssistantRecipient, Event{ToolCall, "="}},
		{stateGPTOSSAssistantTo, "<|channel|>"}:        {stateGPTOSSChannel, Event{Kind: ToolCallEnd}},
		{stateGPTOSSAssistantTo, "<|message|>"}:        {stateGPTOSSNormal, Event{Kind: ToolCallEnd}},
		{stateGPTOSSAssistantTo, ""}:                   {stateGPTOSSAssistantHeader, Event{Kind: ToolCallEnd}},
		{stateGPTOSSAssistantRecipient, "assistant"}:   {stateGPTOSSAssistantHeader, Event{Kind: ToolCallEnd}},
		{stateGPTOSSAssistantRecipient, "<|channel|>"}: {stateGPTOSSIgnoredHeader, Event{Kind: ToolCallEnd}},
		{stateGPTOSSAssistantRecipient, "<|message|>"}: {stateGPTOSSIgnored, Event{Kind: ToolCallEnd}},
		{stateGPTOSSAssistantRecipient, ""}:            {stateGPTOSSToolRoleHeader, Event{Kind: ToolCall}},
		// gpt-oss role recipient before first channel
		{stateGPTOSSToolRoleHeader, "<|channel|>"}:   {stateGPTOSSToolChannel, Event{}},
		{stateGPTOSSToolRoleHeader, "<|constrain|>"}: {stateGPTOSSToolRoleConstraint, Event{}},
		{stateGPTOSSToolRoleHeader, "<|message|>"}:   {stateGPTOSSTool, Event{Kind: ToolCallStart}},
		{stateGPTOSSToolRoleHeader, ""}:              {stateGPTOSSToolRoleHeader, Event{Kind: ToolCall}},
		// gpt-oss role tool constraint
		{stateGPTOSSToolRoleConstraint, "<|channel|>"}: {stateGPTOSSToolChannel, Event{}},
		{stateGPTOSSToolRoleConstraint, "<|message|>"}: {stateGPTOSSTool, Event{Kind: ToolCallStart}},
		{stateGPTOSSToolRoleConstraint, ""}:            {stateGPTOSSToolRoleConstraint, Event{}},
		// gpt-oss channel
		{stateGPTOSSChannel, "analysis"}:    {stateGPTOSSAnalysis, Event{}},
		{stateGPTOSSChannel, "comment"}:     {stateGPTOSSChannelComment, Event{}},
		{stateGPTOSSChannel, "final"}:       {stateGPTOSSFinal, Event{}},
		{stateGPTOSSChannel, "<|channel|>"}: {stateGPTOSSChannel, Event{}},
		{stateGPTOSSChannel, "<|message|>"}: {stateGPTOSSIgnored, Event{}},
		{stateGPTOSSChannel, ""}:            {stateGPTOSSIgnoredHeader, Event{}},
		// gpt-oss commentary suffix
		{stateGPTOSSChannelComment, "ary"}:         {stateGPTOSSFinal, Event{}},
		{stateGPTOSSChannelComment, "<|channel|>"}: {stateGPTOSSChannel, Event{}},
		{stateGPTOSSChannelComment, "<|message|>"}: {stateGPTOSSIgnored, Event{}},
		{stateGPTOSSChannelComment, ""}:            {stateGPTOSSIgnoredHeader, Event{}},
		// gpt-oss analysis header
		{stateGPTOSSAnalysis, "<|channel|>"}: {stateGPTOSSChannel, Event{}},
		{stateGPTOSSAnalysis, "<|message|>"}: {stateGPTOSSThink, Event{Kind: EnterThinking}},
		{stateGPTOSSAnalysis, " to"}:         {stateGPTOSSAnalysisTo, Event{ToolCall, " to"}},
		{stateGPTOSSAnalysis, ""}:            {stateGPTOSSAnalysis, Event{}},
		// gpt-oss analysis recipient
		{stateGPTOSSAnalysisTo, "="}:                  {stateGPTOSSAnalysisRecipient, Event{ToolCall, "="}},
		{stateGPTOSSAnalysisTo, "<|channel|>"}:        {stateGPTOSSChannel, Event{Kind: ToolCallEnd}},
		{stateGPTOSSAnalysisTo, "<|message|>"}:        {stateGPTOSSThink, Event{Kind: EnterThinking}},
		{stateGPTOSSAnalysisTo, ""}:                   {stateGPTOSSAnalysis, Event{Kind: ToolCallEnd}},
		{stateGPTOSSAnalysisRecipient, "assistant"}:   {stateGPTOSSAnalysis, Event{Kind: ToolCallEnd}},
		{stateGPTOSSAnalysisRecipient, "<|channel|>"}: {stateGPTOSSIgnoredHeader, Event{Kind: ToolCallEnd}},
		{stateGPTOSSAnalysisRecipient, "<|message|>"}: {stateGPTOSSIgnored, Event{Kind: ToolCallEnd}},
		{stateGPTOSSAnalysisRecipient, ""}:            {stateGPTOSSToolHeader, Event{Kind: ToolCall}},
		// gpt-oss final/commentary header
		{stateGPTOSSFinal, "<|channel|>"}: {stateGPTOSSChannel, Event{}},
		{stateGPTOSSFinal, "<|message|>"}: {stateGPTOSSNormal, Event{}},
		{stateGPTOSSFinal, " to"}:         {stateGPTOSSFinalTo, Event{ToolCall, " to"}},
		{stateGPTOSSFinal, ""}:            {stateGPTOSSFinal, Event{}},
		// gpt-oss final/commentary recipient
		{stateGPTOSSFinalTo, "="}:                  {stateGPTOSSFinalRecipient, Event{ToolCall, "="}},
		{stateGPTOSSFinalTo, "<|channel|>"}:        {stateGPTOSSChannel, Event{Kind: ToolCallEnd}},
		{stateGPTOSSFinalTo, "<|message|>"}:        {stateGPTOSSNormal, Event{Kind: ToolCallEnd}},
		{stateGPTOSSFinalTo, ""}:                   {stateGPTOSSFinal, Event{Kind: ToolCallEnd}},
		{stateGPTOSSFinalRecipient, "assistant"}:   {stateGPTOSSFinal, Event{Kind: ToolCallEnd}},
		{stateGPTOSSFinalRecipient, "<|channel|>"}: {stateGPTOSSIgnoredHeader, Event{Kind: ToolCallEnd}},
		{stateGPTOSSFinalRecipient, "<|message|>"}: {stateGPTOSSIgnored, Event{Kind: ToolCallEnd}},
		{stateGPTOSSFinalRecipient, ""}:            {stateGPTOSSToolHeader, Event{Kind: ToolCall}},
		// gpt-oss tool header
		{stateGPTOSSToolHeader, "<|channel|>"}:   {stateGPTOSSChannel, Event{Kind: ToolCallEnd}},
		{stateGPTOSSToolHeader, "<|constrain|>"}: {stateGPTOSSToolConstraint, Event{}},
		{stateGPTOSSToolHeader, "<|message|>"}:   {stateGPTOSSTool, Event{Kind: ToolCallStart}},
		{stateGPTOSSToolHeader, ""}:              {stateGPTOSSToolHeader, Event{Kind: ToolCall}},
		// gpt-oss tool constraint
		{stateGPTOSSToolConstraint, "<|channel|>"}: {stateGPTOSSChannel, Event{Kind: ToolCallEnd}},
		{stateGPTOSSToolConstraint, "<|message|>"}: {stateGPTOSSTool, Event{Kind: ToolCallStart}},
		{stateGPTOSSToolConstraint, ""}:            {stateGPTOSSToolConstraint, Event{}},
		// gpt-oss tool channel
		{stateGPTOSSToolChannel, "analysis"}:    {stateGPTOSSToolHeader, Event{}},
		{stateGPTOSSToolChannel, "comment"}:     {stateGPTOSSToolChannelComment, Event{}},
		{stateGPTOSSToolChannel, "final"}:       {stateGPTOSSToolHeader, Event{}},
		{stateGPTOSSToolChannel, "<|channel|>"}: {stateGPTOSSChannel, Event{Kind: ToolCallEnd}},
		{stateGPTOSSToolChannel, "<|message|>"}: {stateGPTOSSIgnored, Event{Kind: ToolCallEnd}},
		{stateGPTOSSToolChannel, ""}:            {stateGPTOSSToolIgnoredHeader, Event{}},
		// gpt-oss tool commentary suffix
		{stateGPTOSSToolChannelComment, "ary"}:         {stateGPTOSSToolHeader, Event{}},
		{stateGPTOSSToolChannelComment, "<|channel|>"}: {stateGPTOSSChannel, Event{Kind: ToolCallEnd}},
		{stateGPTOSSToolChannelComment, "<|message|>"}: {stateGPTOSSIgnored, Event{Kind: ToolCallEnd}},
		{stateGPTOSSToolChannelComment, ""}:            {stateGPTOSSToolIgnoredHeader, Event{}},
		// gpt-oss ignored tool channel header
		{stateGPTOSSToolIgnoredHeader, "<|channel|>"}: {stateGPTOSSChannel, Event{Kind: ToolCallEnd}},
		{stateGPTOSSToolIgnoredHeader, "<|message|>"}: {stateGPTOSSIgnored, Event{Kind: ToolCallEnd}},
		{stateGPTOSSToolIgnoredHeader, ""}:            {stateGPTOSSToolIgnoredHeader, Event{}},
		// gpt-oss ignored header
		{stateGPTOSSIgnoredHeader, "<|channel|>"}: {stateGPTOSSChannel, Event{}},
		{stateGPTOSSIgnoredHeader, "<|message|>"}: {stateGPTOSSIgnored, Event{}},
		{stateGPTOSSIgnoredHeader, ""}:            {stateGPTOSSIgnoredHeader, Event{}},
		// gpt-oss ignored role header
		{stateGPTOSSIgnoredRoleHeader, "<|message|>"}: {stateGPTOSSIgnored, Event{}},
		{stateGPTOSSIgnoredRoleHeader, ""}:            {stateGPTOSSIgnoredRoleHeader, Event{}},
		// gpt-oss thinking body
		{stateGPTOSSThink, "<|end|>"}:   {stateGPTOSSEnd, Event{Kind: LeaveThinking}},
		{stateGPTOSSThink, "<|call|>"}:  {stateGPTOSSEnd, Event{Kind: LeaveThinking}},
		{stateGPTOSSThink, "<|start|>"}: {stateGPTOSSStart, Event{Kind: LeaveThinking}},
		{stateGPTOSSThink, ""}:          {stateGPTOSSThink, Event{Kind: Thinking}},
		// gpt-oss content body
		{stateGPTOSSNormal, "<|end|>"}:   {stateGPTOSSEnd, Event{}},
		{stateGPTOSSNormal, "<|call|>"}:  {stateGPTOSSEnd, Event{}},
		{stateGPTOSSNormal, "<|start|>"}: {stateGPTOSSStart, Event{}},
		{stateGPTOSSNormal, ""}:          {stateGPTOSSNormal, Event{Kind: Content}},
		// gpt-oss tool body
		{stateGPTOSSTool, "<|end|>"}:   {stateGPTOSSEnd, Event{Kind: ToolCallEnd}},
		{stateGPTOSSTool, "<|call|>"}:  {stateGPTOSSEnd, Event{Kind: ToolCallEnd}},
		{stateGPTOSSTool, "<|start|>"}: {stateGPTOSSStart, Event{Kind: ToolCallEnd}},
		{stateGPTOSSTool, ""}:          {stateGPTOSSTool, Event{Kind: ToolCall}},
		// gpt-oss ignored body
		{stateGPTOSSIgnored, "<|end|>"}:   {stateGPTOSSEnd, Event{}},
		{stateGPTOSSIgnored, "<|call|>"}:  {stateGPTOSSEnd, Event{}},
		{stateGPTOSSIgnored, "<|start|>"}: {stateGPTOSSStart, Event{}},
		{stateGPTOSSIgnored, ""}:          {stateGPTOSSIgnored, Event{}},
		// gpt-oss next message
		{stateGPTOSSEnd, "<|start|>"}: {stateGPTOSSStart, Event{}},
		{stateGPTOSSEnd, ""}:          {stateGPTOSSEnd, Event{}},

		// gemma4 channel
		{stateAssistant, "<|channel>"}: {stateGemmaChannel, Event{}},
		{stateGemmaChannel, "thought"}: {stateGemmaChannelThought, Event{}},
		{stateGemmaChannel, ""}:        {stateGemmaChannel, Event{}},
		// gemma4 thought header
		{stateGemmaChannelThought, "\n"}:         {stateGemmaThink, Event{Kind: EnterThinking}},
		{stateGemmaChannelThought, "<channel|>"}: {stateGemmaNormal, Event{}},
		{stateGemmaChannelThought, ""}:           {stateGemmaChannelThought, Event{}},
		// gemma4 thinking body
		{stateGemmaThink, "<channel|>"}: {stateGemmaNormal, Event{Kind: LeaveThinking}},
		{stateGemmaThink, ""}:           {stateGemmaThink, Event{Kind: Thinking}},
		// gemma4 content body
		{stateGemmaNormal, ""}: {stateGemmaNormal, Event{Kind: Content}},
	}
	for st, kind := range map[state]EventKind{
		stateGPTOSSStart:              Consume,
		stateGPTOSSAssistantHeader:    Consume,
		stateGPTOSSAssistantTo:        ToolCallEnd,
		stateGPTOSSAssistantRecipient: ToolCallEnd,
		stateGPTOSSToolRoleHeader:     ToolCallEnd,
		stateGPTOSSToolRoleConstraint: ToolCallEnd,
		stateGPTOSSChannel:            Consume,
		stateGPTOSSChannelComment:     Consume,
		stateGPTOSSAnalysis:           Consume,
		stateGPTOSSAnalysisTo:         ToolCallEnd,
		stateGPTOSSAnalysisRecipient:  ToolCallEnd,
		stateGPTOSSFinal:              Consume,
		stateGPTOSSFinalTo:            ToolCallEnd,
		stateGPTOSSFinalRecipient:     ToolCallEnd,
		stateGPTOSSToolHeader:         ToolCallEnd,
		stateGPTOSSToolConstraint:     ToolCallEnd,
		stateGPTOSSToolChannel:        ToolCallEnd,
		stateGPTOSSToolChannelComment: ToolCallEnd,
		stateGPTOSSToolIgnoredHeader:  ToolCallEnd,
		stateGPTOSSIgnoredHeader:      Consume,
		stateGPTOSSIgnoredRoleHeader:  Consume,
	} {
		ts[key{st, "<|start|>"}] = transition{stateGPTOSSStart, Event{Kind: kind}}
		for _, token := range []string{"<|end|>", "<|call|>"} {
			ts[key{st, token}] = transition{stateGPTOSSEnd, Event{Kind: kind}}
		}
	}
	return ts
}()

// Splitter tracks one generation; not safe for concurrent use.
type Splitter struct {
	state state
}

func New() *Splitter {
	return &Splitter{state: stateAssistant}
}

func (s *Splitter) Feed(token string) Event {
	t, ok := transitions[key{s.state, token}]
	if !ok {
		t = transitions[key{s.state, ""}]
		switch t.Kind {
		case Consume, Thinking, Content, ToolCall:
			t.Text = token
		}
	}
	s.state = t.next
	return t.Event
}

// RestoreOpener wraps onToken so that a <think> already opened by the prompt
// (LFM3, MiniCPM5, ...) is replayed, with the whitespace the template left
// after it, before the first generated token. The model cannot emit it, and
// the Splitter needs it to enter thinking.
func RestoreOpener(prompt string, onToken func(string) bool) func(string) bool {
	trimmed := strings.TrimRight(prompt, " \t\r\n")
	if !strings.HasSuffix(trimmed, "<think>") {
		return onToken
	}
	replay := []string{"<think>"}
	if tail := prompt[len(trimmed):]; tail != "" {
		replay = append(replay, tail)
	}
	return func(token string) bool {
		for _, piece := range replay {
			if !onToken(piece) {
				return false
			}
		}
		replay = nil
		return onToken(token)
	}
}
