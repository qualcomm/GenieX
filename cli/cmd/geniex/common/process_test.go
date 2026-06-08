// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package common

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	geniex_sdk "github.com/qcom-it-nexa-ai/geniex/bindings/go"
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
