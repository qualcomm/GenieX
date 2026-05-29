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

package utils

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"os"
	"strings"
	"testing"
)

// TestSaveURIToTempFile_WebP tests WebP to PNG conversion
func TestSaveURIToTempFile_WebP(t *testing.T) {
	// Create a simple test WebP image (1x1 red pixel)
	// Note: This is a minimal valid WebP file in base64
	webpData := "UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAQAcJaQAA3AA/v3AgAA="

	webpBytes, err := base64.StdEncoding.DecodeString(webpData)
	if err != nil {
		t.Fatalf("Failed to decode test WebP data: %v", err)
	}

	// Create temp WebP file
	tmpWebP, err := os.CreateTemp("", "test-*.webp")
	if err != nil {
		t.Fatalf("Failed to create temp WebP file: %v", err)
	}
	defer os.Remove(tmpWebP.Name())

	if _, err := tmpWebP.Write(webpBytes); err != nil {
		t.Fatalf("Failed to write WebP data: %v", err)
	}
	tmpWebP.Close()

	// Test conversion via file path
	resultPath, err := SaveURIToTempFile(tmpWebP.Name())
	if err != nil {
		t.Fatalf("SaveURIToTempFile failed: %v", err)
	}
	defer os.Remove(resultPath)

	// Verify output is PNG
	if !strings.HasSuffix(resultPath, ".png") {
		t.Errorf("Expected PNG extension, got: %s", resultPath)
	}

	// Verify the file can be decoded as PNG
	resultFile, err := os.Open(resultPath)
	if err != nil {
		t.Fatalf("Failed to open result file: %v", err)
	}
	defer resultFile.Close()

	_, err = png.Decode(resultFile)
	if err != nil {
		t.Errorf("Failed to decode result as PNG: %v", err)
	}
}

// TestSaveURIToTempFile_WebP_DataURI tests WebP conversion from data URI
func TestSaveURIToTempFile_WebP_DataURI(t *testing.T) {
	// Minimal valid WebP as data URI
	webpData := "UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAQAcJaQAA3AA/v3AgAA="
	dataURI := "data:image/webp;base64," + webpData

	resultPath, err := SaveURIToTempFile(dataURI)
	if err != nil {
		t.Fatalf("SaveURIToTempFile failed for data URI: %v", err)
	}
	defer os.Remove(resultPath)

	// Verify it's converted to PNG
	if !strings.HasSuffix(resultPath, ".png") {
		t.Errorf("Expected PNG extension for converted WebP, got: %s", resultPath)
	}
}

// TestSaveURIToTempFile_PNG tests that PNG files are not converted
func TestSaveURIToTempFile_PNG(t *testing.T) {
	// Create a simple PNG image
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.Set(0, 0, color.RGBA{255, 0, 0, 255})

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("Failed to encode test PNG: %v", err)
	}

	// Create temp PNG file
	tmpPNG, err := os.CreateTemp("", "test-*.png")
	if err != nil {
		t.Fatalf("Failed to create temp PNG file: %v", err)
	}
	defer os.Remove(tmpPNG.Name())

	if _, err := tmpPNG.Write(buf.Bytes()); err != nil {
		t.Fatalf("Failed to write PNG data: %v", err)
	}
	tmpPNG.Close()

	// Test that PNG is saved without conversion
	resultPath, err := SaveURIToTempFile(tmpPNG.Name())
	if err != nil {
		t.Fatalf("SaveURIToTempFile failed for PNG: %v", err)
	}
	defer os.Remove(resultPath)

	// Verify output is still PNG
	if !strings.HasSuffix(resultPath, ".png") {
		t.Errorf("Expected PNG extension, got: %s", resultPath)
	}
}
