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

package readline

import (
	"fmt"
	"log/slog"
	"os"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/mattn/go-runewidth"
	"golang.org/x/term"
)

type Buffer struct {
	// configuration
	prompt      string
	altPrompt   string
	placeholder string

	// state
	data         []rune
	cursorIndex  int
	cursorHeight int
}

func NewBuffer(prompt, altPrompt, placeholder string) *Buffer {
	return &Buffer{
		prompt:      prompt,
		altPrompt:   altPrompt,
		placeholder: placeholder,
		data:        make([]rune, 0),
	}
}

func (b *Buffer) insertRuneAtCursor(r rune) {
	b.data = append(b.data, 0) // extend slice
	copy(b.data[b.cursorIndex+1:], b.data[b.cursorIndex:])
	b.data[b.cursorIndex] = r
	b.cursorIndex++
}

func (b *Buffer) resetState() {
	b.data = b.data[:0]
	b.cursorIndex = 0
	b.cursorHeight = 1
}

func (b *Buffer) refresh() {
	width, _, err := term.GetSize(int(os.Stdout.Fd()))
	if err != nil {
		width = 80
		slog.Warn("failed to get terminal width", "error", err)
	}

	// check min width
	if width <= runewidth.StringWidth(b.prompt)+4 || width <= runewidth.StringWidth(b.altPrompt)+4 {
		print("\x1b[H\x1b[2J")
		print("terminal width is too small!")
		return
	}

	var buffer strings.Builder

	// move cursor to the top
	if b.cursorHeight != 1 {
		fmt.Fprintf(&buffer, "\x1b[%dA", b.cursorHeight-1)
	}

	// render lines

	curWidth := 0
	curHeight := 1
	cursorWidth := 0
	cursorHeight := 1

	buffer.WriteString("\x1b[1G") // move cursor to beginning
	buffer.WriteString("\x1b[J")  // clean after
	buffer.WriteString(b.prompt)
	curWidth += calcANSIWidth(b.prompt)
	cursorWidth = curWidth

	avail := width - curWidth - 1
	if len(b.data) == 0 && b.placeholder != "" && avail > 0 {
		buffer.WriteString(truncateANSI(b.placeholder, avail))
	}

	for i, r := range b.data {
		// line wrap
		rw := runewidth.RuneWidth(r)
		if r == CtrlJ {
			// new line
			buffer.WriteString("\n")
			buffer.WriteString(b.altPrompt)
			curHeight++
			curWidth = calcANSIWidth(b.altPrompt)
		} else if curWidth+rw == width {
			// exactly fit
			buffer.WriteString(string(r))
			buffer.WriteString("\n")
			buffer.WriteString(b.altPrompt)
			curHeight++
			curWidth = calcANSIWidth(b.altPrompt)
		} else if curWidth+rw > width {
			// over flow
			buffer.WriteString("\n")
			buffer.WriteString(b.altPrompt)
			curHeight++
			buffer.WriteString(string(r))
			curWidth = calcANSIWidth(b.altPrompt) + rw
		} else {
			// normal char
			buffer.WriteString(string(r))
			curWidth += rw
		}
		// record cursor position
		if i == b.cursorIndex-1 {
			cursorHeight = curHeight
			cursorWidth = curWidth
		}
	}

	// move cursor to the position

	b.cursorHeight = cursorHeight
	if curHeight > cursorHeight {
		fmt.Fprintf(&buffer, "\x1b[%dA", curHeight-cursorHeight)
	}
	buffer.WriteString("\x1b[1G") // move cursor to beginning
	fmt.Fprintf(&buffer, "\x1b[%dC", cursorWidth)

	print(buffer.String())
}

var ansiRegexp = regexp.MustCompile(`\x1b\[[0-9;?]*[ -/]*[@-~]`)

func calcANSIWidth(s string) int {
	return runewidth.StringWidth(ansiRegexp.ReplaceAllString(s, ""))
}

func truncateANSI(s string, limit int) string {
	if limit <= 0 {
		return ""
	}
	if calcANSIWidth(s) <= limit {
		return s
	}
	locs := ansiRegexp.FindAllStringIndex(s, -1)

	var out strings.Builder
	width := 0
	hasSGR := false
	i := 0
	for i < len(s) {
		if len(locs) > 0 && locs[0][0] == i {
			out.WriteString(s[locs[0][0]:locs[0][1]])
			hasSGR = true
			i = locs[0][1]
			locs = locs[1:]
			continue
		}
		r, sz := utf8.DecodeRuneInString(s[i:])
		rw := runewidth.RuneWidth(r)
		if width+rw > limit-1 {
			break
		}
		out.WriteRune(r)
		width += rw
		i += sz
	}
	out.WriteRune('…')
	if hasSGR {
		out.WriteString("\x1b[0m")
	}
	return out.String()
}
