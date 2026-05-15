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
	"testing"

	"github.com/lmittmann/tint"
)

const MODEL_NAME = "qualcomm/Qwen3-4B-Instruct-2507"

func TestMain(m *testing.M) {
	slog.SetDefault(slog.New(tint.NewHandler(os.Stderr, &tint.Options{
		AddSource: true,
		Level:     slog.LevelDebug,
	})))

	// only test aihub; chipset getter returns a fixed Snapdragon X Elite
	// since there is no store singleton in the test binary.
	hubs = []ModelHub{NewAIHub(
		func() string { return "qualcomm-snapdragon-x-elite" },
	)}

	os.Exit(m.Run())
}

func TestModelInfo(t *testing.T) {
	files, mf, err := ModelInfo(context.Background(), MODEL_NAME)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("files: %+v", files)
	t.Logf("pseudo manifest: %+v", mf)

	if mf == nil {
		t.Fatal("expected pseudo geniex.json to be parsed into *types.ModelManifest")
	}
	if mf.PluginId != "qairt" {
		t.Errorf("PluginId: got %q, want qairt", mf.PluginId)
	}
}

func TestGetFileContent(t *testing.T) {
	// The pseudo geniex.json is always present; round-trip it via
	// GetFileContent so we don't depend on a real zip download.
	if _, _, err := ModelInfo(context.Background(), MODEL_NAME); err != nil {
		t.Fatal(err)
	}
	data, err := GetFileContent(context.Background(), MODEL_NAME, "geniex.json")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("GetFileContent(geniex.json):\n%s", data)
}

func TestDownload(t *testing.T) {
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
