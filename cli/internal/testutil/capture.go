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

// Package testutil hosts shared helpers used only by *_test.go files.
package testutil

import (
	"io"
	"os"
	"testing"
)

// CaptureOutput swaps os.Stdout and os.Stderr with pipes for the duration of
// fn, then returns whatever fn wrote to each plus fn's own error. Use it to
// test code that reaches for fmt.Print* directly instead of an injectable
// io.Writer.
func CaptureOutput(t *testing.T, fn func() error) (stdout, stderr string, fnErr error) {
	t.Helper()

	origOut, origErr := os.Stdout, os.Stderr
	rOut, wOut, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe stdout: %v", err)
	}
	rErr, wErr, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe stderr: %v", err)
	}
	os.Stdout, os.Stderr = wOut, wErr

	outCh := make(chan string, 1)
	errCh := make(chan string, 1)
	go func() { b, _ := io.ReadAll(rOut); outCh <- string(b) }()
	go func() { b, _ := io.ReadAll(rErr); errCh <- string(b) }()

	fnErr = fn()

	wOut.Close()
	wErr.Close()
	os.Stdout, os.Stderr = origOut, origErr
	return <-outCh, <-errCh, fnErr
}
