// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestDigestArtifactIsDeterministicAndContentBound(t *testing.T) {
	root := t.TempDir()
	firstPath := filepath.Join(root, "a.bin")
	secondPath := filepath.Join(root, "nested", "b.json")
	if err := os.MkdirAll(filepath.Dir(secondPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(firstPath, []byte("alpha"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(secondPath, []byte("beta"), 0o600); err != nil {
		t.Fatal(err)
	}

	absolute, first, err := digestArtifact(root)
	if err != nil {
		t.Fatal(err)
	}
	if !filepath.IsAbs(absolute) {
		t.Fatalf("artifact path is not absolute: %q", absolute)
	}
	_, second, err := digestArtifact(root)
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Fatalf("unchanged artifact digest changed: %q != %q", first, second)
	}

	if err := os.WriteFile(firstPath, []byte("omega"), 0o600); err != nil {
		t.Fatal(err)
	}
	changedTime := time.Unix(1_800_000_000, 123)
	if err := os.Chtimes(firstPath, changedTime, changedTime); err != nil {
		t.Fatal(err)
	}
	_, changed, err := digestArtifact(root)
	if err != nil {
		t.Fatal(err)
	}
	if changed == first {
		t.Fatal("changed artifact retained its old digest")
	}
}

func TestDigestArtifactIncludesRelativePaths(t *testing.T) {
	left := t.TempDir()
	right := t.TempDir()
	if err := os.WriteFile(filepath.Join(left, "left.bin"), []byte("same"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(right, "right.bin"), []byte("same"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, leftDigest, err := digestArtifact(left)
	if err != nil {
		t.Fatal(err)
	}
	_, rightDigest, err := digestArtifact(right)
	if err != nil {
		t.Fatal(err)
	}
	if leftDigest == rightDigest {
		t.Fatal("different artifact layouts produced the same digest")
	}
}

func TestDigestArtifactBindsTheRootKind(t *testing.T) {
	root := t.TempDir()
	fileRoot := filepath.Join(root, "flat", "leaf.bin")
	if err := os.MkdirAll(filepath.Dir(fileRoot), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fileRoot, []byte("same"), 0o600); err != nil {
		t.Fatal(err)
	}
	directoryRoot := filepath.Join(root, "tree")
	if err := os.MkdirAll(directoryRoot, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directoryRoot, "leaf.bin"), []byte("same"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, fileDigest, err := digestArtifact(fileRoot)
	if err != nil {
		t.Fatal(err)
	}
	_, directoryDigest, err := digestArtifact(directoryRoot)
	if err != nil {
		t.Fatal(err)
	}
	if fileDigest == directoryDigest {
		t.Fatal("file and directory artifact roots produced the same digest")
	}
}

func TestDigestArtifactFailsClosed(t *testing.T) {
	if _, _, err := digestArtifact(""); err == nil {
		t.Fatal("empty artifact path was accepted")
	}
	if _, _, err := digestArtifact(filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatal("missing artifact path was accepted")
	}
	if _, _, err := digestArtifact(t.TempDir()); err == nil {
		t.Fatal("empty artifact directory was accepted")
	}
}
