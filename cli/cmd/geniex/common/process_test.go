// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package common

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/internal/thinkfsm"
)

func TestProcessContextLengthExceeded(t *testing.T) {
	cases := []struct {
		name        string
		runErr      error
		resetErr    error
		wantReset   int  // expected number of Reset calls
		wantRuns    int  // expected number of Run calls
		wantProcErr bool // Process should return a non-nil error
	}{
		{
			name:      "context length exceeded resets and continues",
			runErr:    geniex_sdk.ErrLlmTokenizationContextLength,
			wantReset: 1,
			wantRuns:  2, // first errors+resets, second returns nil then EOF
		},
		{
			name:      "prompt too long resets and continues",
			runErr:    geniex_sdk.ErrLlmGenerationPromptTooLong,
			wantReset: 1,
			wantRuns:  2,
		},
		{
			name:        "reset failure aborts the loop",
			runErr:      geniex_sdk.ErrLlmTokenizationContextLength,
			resetErr:    errors.New("reset boom"),
			wantReset:   1,
			wantRuns:    1,
			wantProcErr: true,
		},
		{
			name:        "unrelated error aborts without reset",
			runErr:      errors.New("some other failure"),
			wantReset:   0,
			wantRuns:    1,
			wantProcErr: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var resetCalls, runCalls int
			firstRun := true

			p := &Processor{
				GetPrompt: func() (string, error) {
					return "hi", nil
				},
				Run: func(_ string, _, _ []string, _ func(string) bool) (string, geniex_sdk.ProfileData, error) {
					runCalls++
					if firstRun {
						firstRun = false
						return "", geniex_sdk.ProfileData{}, tc.runErr
					}
					return "", geniex_sdk.ProfileData{}, nil
				},
				Reset: func() error {
					resetCalls++
					return tc.resetErr
				},
			}
			// After a successful second round, stop the loop with EOF.
			if tc.wantRuns > 1 {
				prompts := 0
				p.GetPrompt = func() (string, error) {
					prompts++
					if prompts > tc.wantRuns {
						return "", io.EOF
					}
					return "hi", nil
				}
			}

			err := p.Process()
			if tc.wantProcErr && err == nil {
				t.Fatalf("expected Process to return an error, got nil")
			}
			if !tc.wantProcErr && err != nil {
				t.Fatalf("unexpected Process error: %v", err)
			}
			if resetCalls != tc.wantReset {
				t.Errorf("reset calls = %d, want %d", resetCalls, tc.wantReset)
			}
			if runCalls != tc.wantRuns {
				t.Errorf("run calls = %d, want %d", runCalls, tc.wantRuns)
			}
		})
	}
}

func TestParseFiles(t *testing.T) {
	tmp := t.TempDir()

	plain := filepath.Join(tmp, "plain.png")
	mustWriteFile(t, plain)

	spaceDir := filepath.Join(tmp, "OneDrive - Qualcomm", "Documents")
	if err := os.MkdirAll(spaceDir, 0o755); err != nil {
		t.Fatal(err)
	}
	spacePath := filepath.Join(spaceDir, "icehockey.png")
	mustWriteFile(t, spacePath)

	audio := filepath.Join(tmp, "clip.wav")
	mustWriteFile(t, audio)

	cases := []struct {
		name       string
		input      string
		wantImages []string
		wantAudios []string
		wantPrompt string
		wantErr    bool
	}{
		{
			name:       "path with spaces, double-quoted",
			input:      `Can you describe this image: "` + spacePath + `"`,
			wantImages: []string{spacePath},
			wantPrompt: "Can you describe this image:",
		},
		{
			name:       "path with spaces, unquoted",
			input:      `describe ` + spacePath + ` please`,
			wantImages: []string{spacePath},
			wantPrompt: "describe  please",
		},
		{
			name:       "plain path",
			input:      `look at "` + plain + `"`,
			wantImages: []string{plain},
			wantPrompt: "look at",
		},
		{
			name:       "audio file routed separately",
			input:      `transcribe "` + audio + `"`,
			wantAudios: []string{audio},
			wantPrompt: "transcribe",
		},
		{
			name:    "non-existent path is a hard error",
			input:   `describe "` + filepath.Join(tmp, "missing", "ghost.png") + `"`,
			wantErr: true,
		},
		{
			name:       "no media path leaves prompt unchanged",
			input:      `hello world`,
			wantPrompt: "hello world",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := &Processor{}
			gotPrompt, gotImages, gotAudios, err := p.parseFiles(tc.input)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got prompt=%q images=%v", gotPrompt, gotImages)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if strings.TrimSpace(gotPrompt) != strings.TrimSpace(tc.wantPrompt) {
				t.Errorf("prompt mismatch:\n  got:  %q\n  want: %q", gotPrompt, tc.wantPrompt)
			}
			if !reflect.DeepEqual(gotImages, normalizeNil(tc.wantImages)) {
				t.Errorf("images mismatch:\n  got:  %v\n  want: %v", gotImages, tc.wantImages)
			}
			if !reflect.DeepEqual(gotAudios, normalizeNil(tc.wantAudios)) {
				t.Errorf("audios mismatch:\n  got:  %v\n  want: %v", gotAudios, tc.wantAudios)
			}
		})
	}
}

func TestFSMEventThinkFormatting(t *testing.T) {
	for _, tc := range []struct {
		name   string
		tokens []string
		want   string
	}{
		{"missing newlines", []string{"<think>", "reason", "</think>", "answer"}, "<think>\nreason\n</think>\n\nanswer"},
		{"existing newlines", []string{"<think>", "\n", "reason", "\n", "</think>", "\n\n", "answer"}, "<think>\nreason\n</think>\n\nanswer"},
		{"attached newlines", []string{"<think>", "\nreason\n", "</think>", "\n\nanswer"}, "<think>\nreason\n</think>\n\nanswer"},

		{"standalone CR after opener", []string{"<think>", "\r", "reason", "</think>", "answer"}, "<think>\n\rreason\n</think>\n\nanswer"},
		{"standalone CR after closer", []string{"<think>", "reason\n", "</think>", "\r", "answer"}, "<think>\nreason\n</think>\n\n\ranswer"},
		{"standalone CR before boundary", []string{"<think>", "\r", "</think>", "answer"}, "<think>\n\r\n</think>\n\nanswer"},
		{"split answer newlines", []string{"<think>", "reason", "</think>", "\n", "", "\nanswer"}, "<think>\nreason\n</think>\n\nanswer"},
		{"one answer newline", []string{"<think>", "reason", "</think>", "\nanswer"}, "<think>\nreason\n</think>\n\nanswer"},
		{"extra boundary newlines", []string{"<think>", "\n", "\nreason\n\n", "</think>", "\n", "\n", "\nanswer"}, "<think>\n\nreason\n\n</think>\n\n\nanswer"},
		{"internal newlines and indentation", []string{"<think>", "  first\n\n", "  second", "</think>", "  answer\n\nmore"}, "<think>\n  first\n\n  second\n</think>\n\n  answer\n\nmore"},
		{"empty thinking", []string{"<think>", "", "</think>", "answer"}, "<think>\n</think>\n\nanswer"},
		{"unterminated thinking", []string{"<think>", "reason"}, "<think>\nreason"},
		{"plain text", []string{"\nanswer", "\n\nmore"}, "\nanswer\n\nmore"},
		{"unmatched closing tag", []string{"</think>", "answer"}, "</think>answer"},
		{"Ministral", []string{"[THINK]", "\nreason\n", "[/THINK]", "\n\nanswer"}, "<think>\nreason\n</think>\n\nanswer"},
		{"Gemma", []string{"<|channel>", "thought", "\n", "reason\n", "<channel|>", "\n\nanswer"}, "<think>\nreason\n</think>\n\nanswer"},
		{"Gemma leading body newline", []string{"<|channel>", "thought", "\n", "\nreason\n", "<channel|>", "answer"}, "<think>\n\nreason\n</think>\n\nanswer"},
		{"Gemma split leading body newlines", []string{"<|channel>", "thought", "\n", "\n", "", "\nreason\n", "<channel|>", "answer"}, "<think>\n\n\nreason\n</think>\n\nanswer"},
		{"GPT-OSS", []string{"<|channel|>", "analysis", "<|message|>", "reason\n", "<|end|>", "<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "\n\nanswer"}, "<think>\nreason\n</think>\n\nanswer"},
		{"multiple thinking messages", []string{"<|channel|>", "analysis", "<|message|>", "first", "<|end|>", "<|start|>", "assistant", "<|channel|>", "analysis", "<|message|>", "\nsecond", "<|end|>"}, "<think>\nfirst\n</think>\n\n<think>\nsecond\n</think>\n\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := &Processor{splitter: thinkfsm.New()}
			got := captureProcessorOutput(t, func() {
				for _, token := range tc.tokens {
					p.fsmEvent(token)
				}
			})
			if got != tc.want {
				t.Errorf("output = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestFSMEventRestoredThinkOpener(t *testing.T) {
	for _, prompt := range []string{"assistant\n<think>", "assistant\n<think>\n", "assistant\n<think>\n\n"} {
		t.Run(prompt, func(t *testing.T) {
			p := &Processor{splitter: thinkfsm.New()}
			got := captureProcessorOutput(t, func() {
				onToken := thinkfsm.RestoreOpener(prompt, func(token string) bool {
					p.fsmEvent(token)
					return true
				})
				for _, token := range []string{"reason", "</think>", "answer"} {
					onToken(token)
				}
			})
			want := "<think>\nreason\n</think>\n\nanswer"
			if strings.HasSuffix(prompt, "\n\n") {
				want = "<think>\n\nreason\n</think>\n\nanswer"
			}
			if got != want {
				t.Errorf("output = %q, want %q", got, want)
			}
		})
	}
}

func TestProcessResetsThinkFormatting(t *testing.T) {
	p := &Processor{endsWithNewline: true, newlinesToSkip: 2}
	prompts := 0
	p.GetPrompt = func() (string, error) {
		prompts++
		if prompts > 2 {
			return "", io.EOF
		}
		return "hi", nil
	}
	p.Run = func(_ string, _, _ []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
		if p.endsWithNewline || p.newlinesToSkip != 0 {
			t.Fatalf("formatting state not reset: endsWithNewline = %v, newlinesToSkip = %d", p.endsWithNewline, p.newlinesToSkip)
		}
		for _, token := range []string{"<think>", "reason", "</think>"} {
			onToken(token)
		}
		return "", geniex_sdk.ProfileData{}, nil
	}
	captureProcessorOutput(t, func() {
		if err := p.Process(); err != nil {
			t.Fatal(err)
		}
	})
}

func captureProcessorOutput(t *testing.T, run func()) string {
	t.Helper()
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	defer writer.Close()
	stdout := os.Stdout
	defer func() { os.Stdout = stdout }()
	os.Stdout = writer
	run()
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	os.Stdout = stdout
	output, err := io.ReadAll(reader)
	if err != nil {
		t.Fatal(err)
	}
	return string(output)
}

func mustWriteFile(t *testing.T, path string) {
	t.Helper()
	if err := os.WriteFile(path, []byte("fake"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func normalizeNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}
