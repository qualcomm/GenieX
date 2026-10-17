// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import "github.com/openai/openai-go/v3"

func retryChatHistory[T any](messages []openai.ChatCompletionMessageParamUnion, protected int,
	request func([]openai.ChatCompletionMessageParamUnion) (T, error), retryable func(error) bool,
) (T, []openai.ChatCompletionMessageParamUnion, int, error) {
	dropped := 0
	for {
		result, err := request(messages)
		if err == nil || !retryable(err) || len(messages)-protected < 3 ||
			messages[protected].OfUser == nil || messages[protected+1].OfAssistant == nil {
			return result, messages, dropped, err
		}
		messages = append(messages[:protected:protected], messages[protected+2:]...)
		dropped++
	}
}
