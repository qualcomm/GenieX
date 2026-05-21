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

package model_hub

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"testing"

	"github.com/lmittmann/tint"
)

// MODEL_NAME is the well-known AI Hub model used by the network-dependent
// tests below. They only run when -short is not set.
const MODEL_NAME = "qualcomm/Qwen3-4B-Instruct-2507"

func TestMain(m *testing.M) {
	slog.SetDefault(slog.New(tint.NewHandler(os.Stderr, &tint.Options{
		AddSource: true,
		Level:     slog.LevelDebug,
	})))

	// Tests in this file need a real AIHub hub; chipset is hard-coded since
	// there is no store singleton in the test binary.
	hubs = []ModelHub{NewAIHub(
		func() string { return "qualcomm-snapdragon-x-elite" },
	)}

	os.Exit(m.Run())
}

func TestModelInfo(t *testing.T) {
	if testing.Short() {
		t.Skip("network test")
	}
	files, mf, err := ModelInfo(context.Background(), MODEL_NAME)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("files: %+v", files)
	if mf != nil {
		t.Errorf("expected no manifest from AI Hub ModelInfo, got %+v", mf)
	}
	if len(files) != 1 {
		t.Fatalf("expected 1 file (the zip), got %d: %+v", len(files), files)
	}
	if !strings.HasSuffix(files[0].Name, ".zip") || files[0].Size <= 0 {
		t.Errorf("unexpected file: %+v", files[0])
	}
}

func TestDownload(t *testing.T) {
	if testing.Short() {
		t.Skip("network test")
	}
	files, _, err := ModelInfo(context.Background(), MODEL_NAME)
	if err != nil {
		t.Fatal(err)
	}

	outDir := t.TempDir()
	resCh, errCh := StartDownload(context.Background(), MODEL_NAME, outDir, files)
	for p := range resCh {
		t.Logf("Downloaded: %d / %d", p.TotalDownloaded, p.TotalSize)
	}
	for e := range errCh {
		t.Error(e)
	}
}

func TestChoosePluginId(t *testing.T) {
	cases := []struct {
		name    string
		files   []ModelFileInfo
		want    string
		wantErr bool
	}{
		{"gguf only", []ModelFileInfo{{Name: "qwen3-q4_k_m.gguf"}}, PluginLlamaCpp, false},
		{"mixed picks llama_cpp", []ModelFileInfo{{Name: "model.bin"}, {Name: "extra.gguf"}}, PluginLlamaCpp, false},
		{"single genie zip", []ModelFileInfo{{Name: "qwen3-genie-x-elite.zip"}}, PluginQairt, false},
		{"directory with genie_config", []ModelFileInfo{{Name: "model.bin"}, {Name: "genie_config.json"}}, PluginQairt, false},
		{"single non-genie zip rejected", []ModelFileInfo{{Name: "model.zip"}}, "", true},
		{"bin shards without genie_config rejected", []ModelFileInfo{{Name: "model.bin"}, {Name: "metadata.json"}}, "", true},
		{"empty rejected", nil, "", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ChoosePluginId(tc.files)
			if (err != nil) != tc.wantErr {
				t.Fatalf("ChoosePluginId err = %v, wantErr = %v", err, tc.wantErr)
			}
			if got != tc.want {
				t.Errorf("ChoosePluginId = %q, want %q", got, tc.want)
			}
		})
	}
}

func BenchmarkDownload(b *testing.B) {
	files, _, err := ModelInfo(context.Background(), MODEL_NAME)
	if err != nil {
		b.Fatal(err)
	}

	resCh, errCh := StartDownload(context.Background(), MODEL_NAME, b.TempDir(), files)
	for p := range resCh {
		b.Logf("Downloaded: %d / %d", p.TotalDownloaded, p.TotalSize)
	}
	for e := range errCh {
		b.Error(e)
	}
}
