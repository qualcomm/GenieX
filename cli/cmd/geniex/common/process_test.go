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
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

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
