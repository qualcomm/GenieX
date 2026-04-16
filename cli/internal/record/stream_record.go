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
	"encoding/binary"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os/exec"
	"runtime"
	"strings"
)

type StreamRecorder struct {
	cmd    *exec.Cmd
	stdout io.ReadCloser
}

func NewStreamRecorder() (*StreamRecorder, error) {
	var args []string

	sr := StreamRecorder{}

	switch runtime.GOOS {
	case "windows":
		args = []string{
			// input (device)
			"-t", "waveaudio",
			"-d",
			// output format options
			"-t", "raw",
			"-e", "float",
			"-b", "32",
			"-r", "16000",
			"-c", "1",
			"-", // OUTFILE = stdout
			"rate", "16000",
			"channels", "1",
		}
	case "darwin", "linux":
		args = []string{
			// input (device)
			"-d",
			// output format options
			"-t", "raw",
			"-e", "float",
			"-b", "32",
			"-r", "16000",
			"-c", "1",
			"-", // OUTFILE = stdout
			"rate", "16000",
			"channels", "1",
		}
	default:
		return nil, fmt.Errorf("unsupported OS: %s", runtime.GOOS)
	}

	slog.Debug("sox cmd", "cmd", "sox "+strings.Join(args, " "))
	sr.cmd = exec.Command("sox", args...)

	var err error
	sr.stdout, err = sr.cmd.StdoutPipe()
	//sr.cmd.Stderr = os.Stderr
	if err != nil {
		return nil, err
	}

	return &sr, nil
}

func (sr *StreamRecorder) Start() error {
	return sr.cmd.Start()
}

func (sr *StreamRecorder) ReadFloat32(buffer []float32) (int, error) {
	if sr.stdout == nil {
		return 0, fmt.Errorf("recorder not started")
	}

	rawBytes := make([]byte, len(buffer)*4)
	n, err := sr.stdout.Read(rawBytes)
	if err != nil {
		return 0, err
	}

	sampleCount := n / 4
	for i := range sampleCount {
		bits := binary.LittleEndian.Uint32(rawBytes[i*4 : (i+1)*4])
		buffer[i] = math.Float32frombits(bits)
	}

	return sampleCount, nil
}

func (sr *StreamRecorder) Stop() error {
	if sr.cmd != nil {
		return sr.cmd.Process.Kill()
	}
	return nil
}
