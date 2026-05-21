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
	"fmt"
	"io"
	"log/slog"
	"sync"

	"github.com/gookit/color"
)

var theme *Theme
var themeInit sync.Once

type Theme struct {
	Normal  Style
	Info    Style
	Success Style
	Warning Style
	Error   Style

	Reference Style
	Heading   Style
	Command   Style
	Flag      Style

	Quant       Style
	Prompt      Style
	Placeholder Style
	AddFiles    Style
	ThinkOutput Style
	ModelOutput Style
	Profile     Style
}

func GetTheme() *Theme {
	themeInit.Do(func() {
		slog.Debug("Detect terminal theme")
		theme = defaultColorTheme()
	})
	return theme
}

type Style interface {
	Sprint(...any) string
	Sprintf(string, ...any) string
	Code() string
}

type NoColor struct{}

func (n NoColor) Sprint(a ...any) string                 { return fmt.Sprint(a...) }
func (n NoColor) Sprintf(format string, a ...any) string { return fmt.Sprintf(format, a...) }
func (n NoColor) Code() string                           { return "" }

func (t *Theme) Set(s Style) {
	color.SetTerminal(s.Code())
}

func (t *Theme) Reset() {
	color.ResetTerminal()
}

func noColorTheme() *Theme {
	return &Theme{
		Normal:  NoColor{},
		Info:    NoColor{},
		Success: NoColor{},
		Warning: NoColor{},
		Error:   NoColor{},

		Reference: NoColor{},
		Heading:   NoColor{},
		Command:   NoColor{},
		Flag:      NoColor{},

		Quant:       NoColor{},
		Prompt:      NoColor{},
		Placeholder: NoColor{},
		AddFiles:    NoColor{},
		ThinkOutput: NoColor{},
		ModelOutput: NoColor{},
		Profile:     NoColor{},
	}
}

func defaultColorTheme() *Theme {
	theme := noColorTheme()
	if color.SupportColor() {
		slog.Debug("apply 16 color")
		theme.Normal = NoColor{}
		theme.Info = color.Style{color.FgBlue}
		theme.Success = color.Style{color.FgGreen}
		theme.Warning = color.Style{color.FgYellow}
		theme.Error = color.Style{color.FgRed}

		theme.Reference = color.Style{color.OpFuzzy, color.FgDarkGray}
		theme.Heading = color.Style{color.FgCyan, color.Bold}
		theme.Command = color.Style{color.FgGreen}
		theme.Flag = color.Style{color.FgYellow}

		theme.Quant = color.Style{color.FgLightBlue}
		theme.Prompt = color.Style{color.FgGreen, color.Bold}
		theme.Placeholder = color.Style{color.FgDarkGray}
		theme.AddFiles = color.Style{color.FgWhite}
		theme.ThinkOutput = color.Style{color.FgDarkGray}
		theme.ModelOutput = NoColor{}
		theme.Profile = color.Style{color.FgLightCyan}
	}
	if color.Support256Color() {
		slog.Debug("apply 256 color")
		theme.Quant = color.S256(31)
		//theme.AddFiles = color.S256(7)
		//theme.ModelOutput = color.S256(15)
		theme.Profile = color.S256(44)
	}
	if color.SupportTrueColor() {
		slog.Debug("apply true color")
		theme.Quant = color.NewRGBStyle(color.RGB(0, 135, 175))
		//theme.AddFiles = color.NewRGBStyle(color.RGB(192, 192, 192))
		theme.Profile = color.NewRGBStyle(color.RGB(0, 215, 215))
	}
	return theme
}

// NewStyledWriter wraps w so every chunk written to it is rendered through s.
func NewStyledWriter(w io.Writer, s Style) io.Writer {
	return &styledWriter{w: w, s: s}
}

type styledWriter struct {
	w io.Writer
	s Style
}

func (sw *styledWriter) Write(p []byte) (int, error) {
	_, err := fmt.Fprint(sw.w, sw.s.Sprint(string(p)))
	return len(p), err
}
