// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

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
func NewProgressBar(total, startingBytes int64, description string) *ProgressBar {
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
		progressbar.OptionSetStartingBytes(startingBytes),
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
