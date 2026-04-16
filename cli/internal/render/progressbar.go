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

package render

import (
	"os"
	"time"

	"github.com/schollz/progressbar/v3"
)

// ProgressBar wraps the progressbar functionality with huh-style interface
type ProgressBar struct {
	bar *progressbar.ProgressBar
}

// NewProgressBar creates a new progress bar with huh-style interface
func NewProgressBar(total int64, description string) *ProgressBar {
	bar := progressbar.NewOptions64(
		total,
		progressbar.OptionSetDescription(description),
		progressbar.OptionSetWriter(os.Stderr),
		progressbar.OptionShowBytes(true),
		progressbar.OptionShowTotalBytes(true),
		progressbar.OptionSetWidth(10),
		progressbar.OptionThrottle(65*time.Millisecond),
		progressbar.OptionShowCount(),
		progressbar.OptionOnCompletion(func() {
			os.Stderr.WriteString("\n")
		}),
		progressbar.OptionSpinnerType(14),
		progressbar.OptionFullWidth(),
		progressbar.OptionSetRenderBlankState(true),
		progressbar.OptionUseANSICodes(true),
	)
	return &ProgressBar{bar: bar}
}

// Set sets the current progress
func (p *ProgressBar) Set(current int64) {
	p.bar.Set64(current)
}

// Set sets the current progress
func (p *ProgressBar) Add(chunk int64) {
	p.bar.Add64(chunk)
}

// Exit finishes the progress bar
func (p *ProgressBar) Exit() {
	p.bar.Exit()
}

// Clear clears the progress bar
func (p *ProgressBar) Clear() {
	p.bar.Clear()
}
