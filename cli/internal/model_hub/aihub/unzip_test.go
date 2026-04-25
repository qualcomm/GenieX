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

package aihub

import (
	"archive/zip"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// writeZip creates a test zip at path with the given entries. Entries with
// a trailing "/" are added as directory entries; otherwise their value
// becomes the file contents.
func writeZip(t *testing.T, path string, entries map[string]string) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create zip: %v", err)
	}
	defer f.Close()

	w := zip.NewWriter(f)
	for name, content := range entries {
		fw, err := w.Create(name)
		if err != nil {
			t.Fatalf("zip create %s: %v", name, err)
		}
		if _, err := fw.Write([]byte(content)); err != nil {
			t.Fatalf("zip write %s: %v", name, err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("zip close: %v", err)
	}
}

func TestExtractFlat_HappyPath(t *testing.T) {
	tmp := t.TempDir()
	zipPath := filepath.Join(tmp, "model.zip")
	dest := filepath.Join(tmp, "out")
	if err := os.MkdirAll(dest, 0o755); err != nil {
		t.Fatal(err)
	}

	writeZip(t, zipPath, map[string]string{
		"weights/shard_0.bin":   "binary-shard-0",
		"weights/shard_1.bin":   "binary-shard-1",
		"tokenizer.json":        "{}",
		"config/htp.json":       "{}",
	})

	res, err := ExtractFlat(zipPath, dest)
	if err != nil {
		t.Fatalf("ExtractFlat: %v", err)
	}

	if res.EntrypointBasename != "shard_0.bin" {
		t.Errorf("wrong entrypoint: %s", res.EntrypointBasename)
	}
	if len(res.Files) != 4 {
		t.Errorf("expected 4 files, got %d", len(res.Files))
	}

	// Every file is at destDir/<basename>.
	for _, f := range res.Files {
		if _, err := os.Stat(filepath.Join(dest, f.Name)); err != nil {
			t.Errorf("missing extracted file %s: %v", f.Name, err)
		}
	}
}

func TestExtractFlat_CollisionFails(t *testing.T) {
	tmp := t.TempDir()
	zipPath := filepath.Join(tmp, "collide.zip")
	dest := filepath.Join(tmp, "out")
	if err := os.MkdirAll(dest, 0o755); err != nil {
		t.Fatal(err)
	}

	writeZip(t, zipPath, map[string]string{
		"a/shared.bin": "A",
		"b/shared.bin": "B",
	})

	_, err := ExtractFlat(zipPath, dest)
	if !errors.Is(err, ErrFileCollision) {
		t.Fatalf("expected ErrFileCollision, got %v", err)
	}

	// No output files should have been written.
	if entries, _ := os.ReadDir(dest); len(entries) != 0 {
		t.Errorf("destDir should be empty on collision, got %d entries", len(entries))
	}
}

func TestExtractFlat_NoBinShardFails(t *testing.T) {
	tmp := t.TempDir()
	zipPath := filepath.Join(tmp, "nobin.zip")
	dest := filepath.Join(tmp, "out")
	if err := os.MkdirAll(dest, 0o755); err != nil {
		t.Fatal(err)
	}

	writeZip(t, zipPath, map[string]string{
		"tokenizer.json": "{}",
		"config.json":    "{}",
	})

	_, err := ExtractFlat(zipPath, dest)
	if !errors.Is(err, ErrNoBinShard) {
		t.Fatalf("expected ErrNoBinShard, got %v", err)
	}
}
