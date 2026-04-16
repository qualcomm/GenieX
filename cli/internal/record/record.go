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

package record

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

type Recorder struct {
	cmd        *exec.Cmd
	outputFile string
}

func NewRecorder(outputFile string) (*Recorder, error) {
	var cmd *exec.Cmd
	var args []string

	if err := os.MkdirAll(filepath.Dir(outputFile), 0755); err != nil {
		return nil, err
	}

	switch runtime.GOOS {
	case "darwin", "linux":
		args = []string{"-d", "-t", "wav", outputFile}
		cmd = exec.Command("sox", args...)

	case "windows":
		args = []string{"-t", "waveaudio", "-c", "1", "-r", "16000", "-d", outputFile}
		cmd = exec.Command("sox", args...)

	default:
		return nil, fmt.Errorf("unsupported OS: %s", runtime.GOOS)
	}

	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	return &Recorder{
		cmd:        cmd,
		outputFile: outputFile,
	}, nil
}

func (r *Recorder) Run() error {
	return r.cmd.Run()
}

func (r *Recorder) Stop() error {
	if r.cmd.Process == nil {
		return fmt.Errorf("recording not started yet")
	}

	return r.cmd.Process.Kill()
	// _ = r.cmd.Wait()
}

func (r *Recorder) GetOutputFile() string {
	return r.outputFile
}
