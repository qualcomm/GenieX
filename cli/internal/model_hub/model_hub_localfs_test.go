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
	"archive/zip"
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/types"
)

func TestLocalFSPostDownload_DetectsVLM(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "metadata.json"),
		[]byte(`{"genie":{"supports_vision":true}}`), 0o644); err != nil {
		t.Fatalf("write metadata.json: %v", err)
	}

	mf := &types.ModelManifest{}
	d := NewLocalFS(dir)
	if err := d.PostDownload(context.Background(), "x/y", dir, mf); err != nil {
		t.Fatalf("PostDownload: %v", err)
	}
	if mf.ModelType != types.ModelTypeVLM {
		t.Fatalf("ModelType = %q, want %q", mf.ModelType, types.ModelTypeVLM)
	}
}

func TestLocalFSPostDownload_DefaultsToLLMWithoutMetadata(t *testing.T) {
	dir := t.TempDir()
	mf := &types.ModelManifest{}
	d := NewLocalFS(dir)
	if err := d.PostDownload(context.Background(), "x/y", dir, mf); err != nil {
		t.Fatalf("PostDownload: %v", err)
	}
	if mf.ModelType != types.ModelTypeLLM {
		t.Fatalf("ModelType = %q, want %q", mf.ModelType, types.ModelTypeLLM)
	}
}

func TestLocalFSPostDownload_PreservesPresetType(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "metadata.json"),
		[]byte(`{"genie":{"supports_vision":true}}`), 0o644); err != nil {
		t.Fatalf("write metadata.json: %v", err)
	}

	mf := &types.ModelManifest{ModelType: types.ModelTypeLLM}
	d := NewLocalFS(dir)
	if err := d.PostDownload(context.Background(), "x/y", dir, mf); err != nil {
		t.Fatalf("PostDownload: %v", err)
	}
	if mf.ModelType != types.ModelTypeLLM {
		t.Fatalf("ModelType = %q, want it preserved as %q", mf.ModelType, types.ModelTypeLLM)
	}
}

func TestLocalFSPostDownload_FillsQairtPlaceholder(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "metadata.json"),
		[]byte(`{"genie":{"supports_vision":true}}`), 0o644); err != nil {
		t.Fatalf("write metadata.json: %v", err)
	}

	mf := &types.ModelManifest{
		PluginId: "qairt",
		ExtraFiles: []types.ModelFileInfo{
			{Name: "metadata.json", Size: 33, Downloaded: true},
			{Name: "model.bin", Size: 22, Downloaded: true},
			{Name: "tokenizer.json", Size: 11, Downloaded: true},
		},
	}
	d := NewLocalFS(dir)
	if err := d.PostDownload(context.Background(), "x/y", dir, mf); err != nil {
		t.Fatalf("PostDownload: %v", err)
	}

	// qairt loads the directory whole, so ModelFile is just a placeholder
	// entry — ExtraFiles still owns every real file.
	main, ok := mf.ModelFile["N/A"]
	if !ok {
		t.Fatalf("expected N/A entry, got %#v", mf.ModelFile)
	}
	if !main.Downloaded {
		t.Errorf("placeholder must be Downloaded, got %+v", main)
	}
	if len(mf.ExtraFiles) != 3 {
		t.Fatalf("ExtraFiles = %d, want 3 (untouched)", len(mf.ExtraFiles))
	}
	if mf.ModelType != types.ModelTypeVLM {
		t.Errorf("ModelType = %q, want %q", mf.ModelType, types.ModelTypeVLM)
	}
}

// writeQairtZip builds a tiny zip with metadata.json + a fake bin shard,
// returning the zip path. Used to exercise the localfs single-zip flow.
func writeQairtZip(t *testing.T, path, metadataJSON string) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create zip: %v", err)
	}
	defer f.Close()
	w := zip.NewWriter(f)
	for name, content := range map[string]string{
		"metadata.json": metadataJSON,
		"model.bin":     "binary",
	} {
		fw, err := w.Create(name)
		if err != nil {
			t.Fatalf("zip create %s: %v", name, err)
		}
		if _, err := io.WriteString(fw, content); err != nil {
			t.Fatalf("zip write: %v", err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("zip close: %v", err)
	}
}

// LocalFS may be pointed at a single .zip file (instead of a directory).
// PostDownload should unzip it into outputDir before reading metadata.json.
func TestLocalFSPostDownload_ZipPathExtracts(t *testing.T) {
	tmp := t.TempDir()
	zipPath := filepath.Join(tmp, "model.zip")
	writeQairtZip(t, zipPath, `{"genie":{"supports_vision":true}}`)

	// Simulate what StartDownload does: it copies the zip into outputDir
	// before calling PostDownload.
	outDir := t.TempDir()
	dstZip := filepath.Join(outDir, "model.zip")
	src, _ := os.ReadFile(zipPath)
	if err := os.WriteFile(dstZip, src, 0o644); err != nil {
		t.Fatalf("copy zip: %v", err)
	}

	mf := &types.ModelManifest{
		PluginId: "qairt",
		ModelFile: map[string]types.ModelFileInfo{
			"N/A": {Name: "model.zip", Size: 1234, Downloaded: true},
		},
	}
	d := NewLocalFS(zipPath)
	if err := d.PostDownload(context.Background(), "x/y", outDir, mf); err != nil {
		t.Fatalf("PostDownload: %v", err)
	}

	// Zip itself was removed; flat-extracted files now live alongside.
	if _, err := os.Stat(dstZip); !os.IsNotExist(err) {
		t.Errorf("zip should have been removed: err=%v", err)
	}
	for _, name := range []string{"metadata.json", "model.bin"} {
		if _, err := os.Stat(filepath.Join(outDir, name)); err != nil {
			t.Errorf("expected %s extracted, got %v", name, err)
		}
	}
	if mf.ModelType != types.ModelTypeVLM {
		t.Errorf("ModelType = %q, want %q", mf.ModelType, types.ModelTypeVLM)
	}
	if len(mf.ExtraFiles) != 2 {
		t.Errorf("ExtraFiles = %d, want 2 (extracted entries)", len(mf.ExtraFiles))
	}
}

func TestLocalFSPostDownload_GGUFInfersLLM(t *testing.T) {
	dir := t.TempDir()
	mf := &types.ModelManifest{
		PluginId: "llama_cpp",
		ModelFile: map[string]types.ModelFileInfo{
			"Q4_K_M": {Name: "qwen3-q4_k_m.gguf", Size: 99, Downloaded: true},
		},
	}
	d := NewLocalFS(dir)
	if err := d.PostDownload(context.Background(), "x/y", dir, mf); err != nil {
		t.Fatalf("PostDownload: %v", err)
	}
	if _, ok := mf.ModelFile["Q4_K_M"]; !ok {
		t.Errorf("Q4_K_M entry was clobbered: %#v", mf.ModelFile)
	}
	if mf.ModelType != types.ModelTypeLLM {
		t.Errorf("ModelType = %q, want %q (no mmproj ⇒ LLM)", mf.ModelType, types.ModelTypeLLM)
	}
}

func TestLocalFSPostDownload_GGUFInfersVLMFromMMProj(t *testing.T) {
	dir := t.TempDir()
	mf := &types.ModelManifest{
		PluginId:   "llama_cpp",
		MMProjFile: types.ModelFileInfo{Name: "mmproj-f16.gguf"},
	}
	d := NewLocalFS(dir)
	if err := d.PostDownload(context.Background(), "x/y", dir, mf); err != nil {
		t.Fatalf("PostDownload: %v", err)
	}
	if mf.ModelType != types.ModelTypeVLM {
		t.Errorf("ModelType = %q, want %q", mf.ModelType, types.ModelTypeVLM)
	}
}

func TestLocalFSPostDownload_MalformedMetadataLeavesTypeEmpty(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "metadata.json"),
		[]byte("{not json"), 0o644); err != nil {
		t.Fatalf("write metadata.json: %v", err)
	}

	mf := &types.ModelManifest{}
	d := NewLocalFS(dir)
	if err := d.PostDownload(context.Background(), "x/y", dir, mf); err != nil {
		t.Fatalf("PostDownload: %v", err)
	}
	if mf.ModelType != "" {
		t.Fatalf("ModelType = %q, want empty so caller can prompt", mf.ModelType)
	}
}
