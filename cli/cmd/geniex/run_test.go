// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/ssestream"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

func TestTagStreamErrorRetainsNegativeSDKCode(t *testing.T) {
	streamErr := &ssestream.StreamError{
		Message: "Input prompt too long",
		Event:   ssestream.Event{Data: []byte(`{"code":-200103,"error":"Input prompt too long"}`)},
	}
	if err := tagStreamError(streamErr); !errors.Is(err, geniex_sdk.ErrLlmGenerationPromptTooLong) {
		t.Fatalf("got %v, want prompt-too-long SDK error", err)
	}
}

func TestTagStreamErrorKeepsNonSDKError(t *testing.T) {
	streamErr := &ssestream.StreamError{
		Message: "other error",
		Event:   ssestream.Event{Data: []byte(`{"code":-1,"error":"other error"}`)},
	}
	if err := tagStreamError(streamErr); err != streamErr {
		t.Fatalf("got %v, want original stream error", err)
	}
}

func TestRunHistoryRetriesStreamPromptError(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		var body struct {
			Messages []json.RawMessage `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		want := 3
		if requests == 2 {
			want = 1
		}
		if len(body.Messages) != want {
			t.Errorf("request %d: got %d messages, want %d", requests, len(body.Messages), want)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		if requests == 1 {
			fmt.Fprint(w, "data: {\"error\":\"Input prompt too long\",\"code\":-200103}\n\n")
			return
		}
		fmt.Fprint(w, "data: {\"id\":\"test\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	api := openai.NewClient(option.WithBaseURL(server.URL + "/v1"))
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("old"), openai.AssistantMessage("answer"), openai.UserMessage("current"),
	}
	text, kept, dropped, err := retryChatHistory(messages, 0,
		func(got []openai.ChatCompletionMessageParamUnion) (string, error) {
			stream := api.Chat.Completions.NewStreaming(context.Background(), openai.ChatCompletionNewParams{
				Model: "test", Messages: got,
			})
			defer stream.Close()
			var content string
			for stream.Next() {
				if chunk := stream.Current(); len(chunk.Choices) > 0 {
					content += chunk.Choices[0].Delta.Content
				}
			}
			if stream.Err() != nil {
				return "", tagStreamError(stream.Err())
			}
			return content, nil
		}, func(err error) bool { return errors.Is(err, geniex_sdk.ErrLlmGenerationPromptTooLong) })
	if err != nil || text != "ok" || dropped != 1 || len(kept) != 1 || requests != 2 {
		t.Fatalf("text=%q dropped=%d kept=%d requests=%d err=%v", text, dropped, len(kept), requests, err)
	}
}

func TestRunCompletionsContinuesAfterContextOverflow(t *testing.T) {
	oldClient, oldPrompt, oldSystemPrompt := client, prompt, systemPrompt
	defer func() {
		client, prompt, systemPrompt = oldClient, oldPrompt, oldSystemPrompt
	}()
	prompt = []string{"old", "current", "next"}
	systemPrompt = "rules"

	var requests [][]string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		roles := make([]string, len(body.Messages))
		for i, message := range body.Messages {
			roles[i] = message.Role + ":" + message.Content
		}
		requests = append(requests, roles)
		if len(requests) == 1 {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, "null")
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		if len(requests) == 3 {
			fmt.Fprint(w, "data: {\"error\":\"Input prompt too long\",\"code\":-200103}\n\n")
			return
		}
		fmt.Fprint(w, "data: {\"id\":\"test\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	client = openai.NewClient(option.WithBaseURL(server.URL + "/v1"))

	if err := runCompletions(context.Background(), "test", geniex_sdk.ModelTypeLLM); err != nil {
		t.Fatalf("runCompletions: %v", err)
	}
	want := [][]string{
		{"system:rules"},
		{"system:rules", "user:old"},
		{"system:rules", "user:old", "assistant:ok", "user:current"},
		{"system:rules", "user:current"},
		{"system:rules", "user:current", "assistant:ok", "user:next"},
	}
	if !reflect.DeepEqual(requests, want) {
		t.Fatalf("requests = %v, want %v", requests, want)
	}
}
