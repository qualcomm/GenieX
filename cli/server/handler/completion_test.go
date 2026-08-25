// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"reflect"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
)

func TestCompletionPrompt(t *testing.T) {
	tests := []struct {
		name    string
		prompt  openai.CompletionNewParamsPromptUnion
		want    string
		wantErr bool
	}{
		{"string", openai.CompletionNewParamsPromptUnion{OfString: param.NewOpt("<|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>")}, "<|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>", false},
		{"empty string", openai.CompletionNewParamsPromptUnion{OfString: param.NewOpt("")}, "", true},
		{"single-element array", openai.CompletionNewParamsPromptUnion{OfArrayOfStrings: []string{"abc"}}, "abc", false},
		{"multi-element array", openai.CompletionNewParamsPromptUnion{OfArrayOfStrings: []string{"a", "b"}}, "", true},
		{"empty array", openai.CompletionNewParamsPromptUnion{OfArrayOfStrings: []string{}}, "", true},
		{"token array", openai.CompletionNewParamsPromptUnion{OfArrayOfTokens: []int64{1, 2}}, "", true},
		{"omitted", openai.CompletionNewParamsPromptUnion{}, "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := completionPrompt(tt.prompt)
			if (err != nil) != tt.wantErr {
				t.Fatalf("completionPrompt() error = %v, wantErr %v", err, tt.wantErr)
			}
			if got != tt.want {
				t.Errorf("completionPrompt() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCompletionStop(t *testing.T) {
	tests := []struct {
		name string
		stop openai.CompletionNewParamsStopUnion
		want []string
	}{
		{"omitted", openai.CompletionNewParamsStopUnion{}, nil},
		{"string", openai.CompletionNewParamsStopUnion{OfString: param.NewOpt("\n\n")}, []string{"\n\n"}},
		{"empty string", openai.CompletionNewParamsStopUnion{OfString: param.NewOpt("")}, nil},
		{"array", openai.CompletionNewParamsStopUnion{OfStringArray: []string{"<|endoftext|>", "\n"}}, []string{"<|endoftext|>", "\n"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := completionStop(tt.stop); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("completionStop() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestCompletionUnsupported(t *testing.T) {
	tests := []struct {
		name    string
		params  CompletionNewParams
		wantErr bool
	}{
		{"defaults", CompletionNewParams{}, false},
		{"n 1", CompletionNewParams{N: param.NewOpt[int64](1)}, false},
		{"best_of 1", CompletionNewParams{BestOf: param.NewOpt[int64](1)}, false},
		{"empty suffix", CompletionNewParams{Suffix: param.NewOpt("")}, false},
		{"suffix", CompletionNewParams{Suffix: param.NewOpt("tail")}, true},
		{"n 2", CompletionNewParams{N: param.NewOpt[int64](2)}, true},
		{"best_of 2", CompletionNewParams{BestOf: param.NewOpt[int64](2)}, true},
		{"logprobs", CompletionNewParams{Logprobs: param.NewOpt[int64](0)}, true},
		{"logit_bias", CompletionNewParams{LogitBias: map[string]int64{"50256": -100}}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := completionUnsupported(tt.params); (err != nil) != tt.wantErr {
				t.Errorf("completionUnsupported() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
