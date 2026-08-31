// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

type lineageArtifactIdentity struct {
	Runtime         string `json:"runtime"`
	SDKVersion      string `json:"sdk_version"`
	PluginVersion   string `json:"plugin_version"`
	ModelPath       string `json:"model_path"`
	ModelDigest     string `json:"model_digest"`
	TokenizerPath   string `json:"tokenizer_path"`
	TokenizerDigest string `json:"tokenizer_digest"`
}

type artifactFile struct {
	absolutePath string
	relativePath string
	size         int64
	modifiedNS   int64
	info         fs.FileInfo
}

type artifactDigestEntry struct {
	fingerprint string
	digest      string
}

var artifactDigests = struct {
	sync.Mutex
	entries map[string]artifactDigestEntry
}{entries: make(map[string]artifactDigestEntry)}

func resolveLineageArtifactIdentity(runtime, modelPath, tokenizerPath string) (lineageArtifactIdentity, error) {
	modelAbsolute, modelDigest, err := digestArtifact(modelPath)
	if err != nil {
		return lineageArtifactIdentity{}, fmt.Errorf("attest model artifact: %w", err)
	}
	tokenizerAbsolute := ""
	tokenizerDigest := "none"
	if strings.TrimSpace(tokenizerPath) != "" {
		tokenizerAbsolute, tokenizerDigest, err = digestArtifact(tokenizerPath)
		if err != nil {
			return lineageArtifactIdentity{}, fmt.Errorf("attest tokenizer artifact: %w", err)
		}
	}
	return lineageArtifactIdentity{
		Runtime:         runtime,
		SDKVersion:      geniex_sdk.Version(),
		PluginVersion:   geniex_sdk.GetPluginVersion(runtime),
		ModelPath:       modelAbsolute,
		ModelDigest:     modelDigest,
		TokenizerPath:   tokenizerAbsolute,
		TokenizerDigest: tokenizerDigest,
	}, nil
}

// digestArtifact hashes the full file or directory contents. The content hash
// is cached only while the path, file list, sizes, and modification times stay
// unchanged. This makes a large model pay the full hash cost once per process
// while preventing a reloaded, changed artifact from inheriting a lineage.
func digestArtifact(path string) (string, string, error) {
	if strings.TrimSpace(path) == "" {
		return "", "", fmt.Errorf("artifact path is empty")
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return "", "", err
	}
	absolute = filepath.Clean(absolute)
	files, rootKind, fingerprint, err := scanArtifact(absolute)
	if err != nil {
		return "", "", err
	}

	artifactDigests.Lock()
	if cached, ok := artifactDigests.entries[absolute]; ok && cached.fingerprint == fingerprint {
		artifactDigests.Unlock()
		return absolute, cached.digest, nil
	}
	artifactDigests.Unlock()

	hash := sha256.New()
	_, _ = hash.Write([]byte("geniex-managed-cache-artifact/1\x00"))
	if err := writeFramedString(hash, rootKind); err != nil {
		return "", "", err
	}
	if err := binary.Write(hash, binary.BigEndian, uint64(len(files))); err != nil {
		return "", "", err
	}
	for _, file := range files {
		if err := writeFramedString(hash, filepath.ToSlash(file.relativePath)); err != nil {
			return "", "", err
		}
		if err := binary.Write(hash, binary.BigEndian, uint64(file.size)); err != nil {
			return "", "", err
		}
		handle, err := os.Open(file.absolutePath)
		if err != nil {
			return "", "", err
		}
		observed, copyErr := io.Copy(hash, handle)
		closeErr := handle.Close()
		if copyErr != nil {
			return "", "", copyErr
		}
		if closeErr != nil {
			return "", "", closeErr
		}
		if observed != file.size {
			return "", "", fmt.Errorf("artifact file changed size while hashing: %s", file.absolutePath)
		}
		after, statErr := os.Lstat(file.absolutePath)
		if statErr != nil {
			return "", "", statErr
		}
		if after.Mode()&os.ModeSymlink != 0 || !after.Mode().IsRegular() ||
			!os.SameFile(file.info, after) || after.Size() != file.size ||
			after.ModTime().UnixNano() != file.modifiedNS {
			return "", "", fmt.Errorf("artifact file changed while hashing: %s", file.absolutePath)
		}
	}
	_, afterRootKind, afterFingerprint, err := scanArtifact(absolute)
	if err != nil {
		return "", "", err
	}
	if rootKind != afterRootKind || fingerprint != afterFingerprint {
		return "", "", fmt.Errorf("artifact changed while hashing: %s", absolute)
	}
	digest := "sha256:" + hex.EncodeToString(hash.Sum(nil))

	artifactDigests.Lock()
	artifactDigests.entries[absolute] = artifactDigestEntry{fingerprint: fingerprint, digest: digest}
	artifactDigests.Unlock()
	return absolute, digest, nil
}

func scanArtifact(root string) ([]artifactFile, string, string, error) {
	rootInfo, err := os.Lstat(root)
	if err != nil {
		return nil, "", "", err
	}
	if rootInfo.Mode()&os.ModeSymlink != 0 {
		return nil, "", "", fmt.Errorf("symbolic-link artifact root is not supported: %s", root)
	}

	files := make([]artifactFile, 0, 1)
	rootKind := ""
	if rootInfo.Mode().IsRegular() {
		rootKind = "file"
		files = append(files, artifactFile{
			absolutePath: root,
			relativePath: filepath.Base(root),
			size:         rootInfo.Size(),
			modifiedNS:   rootInfo.ModTime().UnixNano(),
			info:         rootInfo,
		})
	} else if rootInfo.IsDir() {
		rootKind = "directory"
		err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if path == root {
				return nil
			}
			info, infoErr := entry.Info()
			if infoErr != nil {
				return infoErr
			}
			if info.Mode()&os.ModeSymlink != 0 {
				return fmt.Errorf("symbolic links are not supported in managed-cache artifacts: %s", path)
			}
			if entry.IsDir() {
				return nil
			}
			if !info.Mode().IsRegular() {
				return fmt.Errorf("non-regular artifact entry is not supported: %s", path)
			}
			relative, relativeErr := filepath.Rel(root, path)
			if relativeErr != nil {
				return relativeErr
			}
			files = append(files, artifactFile{
				absolutePath: path,
				relativePath: relative,
				size:         info.Size(),
				modifiedNS:   info.ModTime().UnixNano(),
				info:         info,
			})
			return nil
		})
		if err != nil {
			return nil, "", "", err
		}
	} else {
		return nil, "", "", fmt.Errorf("artifact root is not a regular file or directory: %s", root)
	}
	if len(files) == 0 {
		return nil, "", "", fmt.Errorf("artifact contains no regular files: %s", root)
	}
	sort.Slice(files, func(i, j int) bool { return files[i].relativePath < files[j].relativePath })

	fingerprintHash := sha256.New()
	_, _ = fingerprintHash.Write([]byte("geniex-managed-cache-fingerprint/1\x00"))
	if err := writeFramedString(fingerprintHash, rootKind); err != nil {
		return nil, "", "", err
	}
	if err := binary.Write(fingerprintHash, binary.BigEndian, uint64(len(files))); err != nil {
		return nil, "", "", err
	}
	for _, file := range files {
		if err := writeFramedString(fingerprintHash, filepath.ToSlash(file.relativePath)); err != nil {
			return nil, "", "", err
		}
		if err := binary.Write(fingerprintHash, binary.BigEndian, uint64(file.size)); err != nil {
			return nil, "", "", err
		}
		if err := binary.Write(fingerprintHash, binary.BigEndian, file.modifiedNS); err != nil {
			return nil, "", "", err
		}
	}
	return files, rootKind, hex.EncodeToString(fingerprintHash.Sum(nil)), nil
}

func writeFramedString(writer io.Writer, value string) error {
	if err := binary.Write(writer, binary.BigEndian, uint64(len(value))); err != nil {
		return err
	}
	_, err := io.WriteString(writer, value)
	return err
}
