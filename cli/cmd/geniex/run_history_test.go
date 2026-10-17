// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import (
	"errors"
	"testing"

	"github.com/openai/openai-go/v3"
)

func TestRetryChatHistory(t *testing.T) {
	tooLong := errors.New("prompt too long")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage("rules"),
		openai.UserMessage("old"), openai.AssistantMessage("answer"),
		openai.UserMessage("new"),
	}
	calls := 0
	result, kept, dropped, err := retryChatHistory(messages, 1,
		func(got []openai.ChatCompletionMessageParamUnion) (string, error) {
			calls++
			if calls == 1 {
				return "", tooLong
			}
			if len(got) != 2 || got[0].OfSystem == nil || got[1].OfUser == nil {
				t.Fatalf("retry lost system or current user: %+v", got)
			}
			return "ok", nil
		}, func(err error) bool { return errors.Is(err, tooLong) })
	if err != nil || result != "ok" || dropped != 1 || len(kept) != 2 || len(messages) != 4 {
		t.Fatalf("result=%q kept=%d dropped=%d calls=%d err=%v", result, len(kept), dropped, calls, err)
	}
}

func TestRetryChatHistoryStopsAtCurrentTurn(t *testing.T) {
	tooLong := errors.New("prompt too long")
	for _, protected := range []int{0, 1} {
		messages := []openai.ChatCompletionMessageParamUnion{openai.UserMessage("current")}
		if protected == 1 {
			messages = append([]openai.ChatCompletionMessageParamUnion{openai.SystemMessage("rules")}, messages...)
		}
		calls := 0
		_, kept, dropped, err := retryChatHistory(messages, protected,
			func([]openai.ChatCompletionMessageParamUnion) (struct{}, error) {
				calls++
				return struct{}{}, tooLong
			}, func(err error) bool { return errors.Is(err, tooLong) })
		if !errors.Is(err, tooLong) || calls != 1 || dropped != 0 || len(kept) != len(messages) {
			t.Fatalf("protected=%d calls=%d dropped=%d err=%v", protected, calls, dropped, err)
		}
	}
}

func TestRetryChatHistoryDropsMultipleTurns(t *testing.T) {
	tooLong := errors.New("prompt too long")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("first"), openai.AssistantMessage("one"),
		openai.UserMessage("second"), openai.AssistantMessage("two"),
		openai.UserMessage("current"),
	}
	calls := 0
	_, kept, dropped, err := retryChatHistory(messages, 0,
		func(got []openai.ChatCompletionMessageParamUnion) (struct{}, error) {
			calls++
			if calls < 3 {
				return struct{}{}, tooLong
			}
			if len(got) != 1 || got[0].OfUser == nil {
				t.Fatalf("current user turn was lost: %+v", got)
			}
			return struct{}{}, nil
		}, func(err error) bool { return errors.Is(err, tooLong) })
	if err != nil || calls != 3 || dropped != 2 || len(kept) != 1 || len(messages) != 5 {
		t.Fatalf("calls=%d dropped=%d kept=%d err=%v", calls, dropped, len(kept), err)
	}
}

func TestRetryChatHistoryKeepsHistoryOnOtherErrors(t *testing.T) {
	failure := errors.New("network failure")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("old"), openai.AssistantMessage("answer"), openai.UserMessage("new"),
	}
	calls := 0
	_, kept, dropped, err := retryChatHistory(messages, 0,
		func([]openai.ChatCompletionMessageParamUnion) (struct{}, error) {
			calls++
			return struct{}{}, failure
		}, func(error) bool { return false })
	if !errors.Is(err, failure) || calls != 1 || dropped != 0 || len(kept) != len(messages) {
		t.Fatalf("calls=%d dropped=%d err=%v", calls, dropped, err)
	}
}
