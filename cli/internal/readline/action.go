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
	"errors"
	"io"
)

var (
	ErrComplete = errors.New("Complete") // readline complete

	ErrInterrupt = errors.New("Interrupt")
)

const (
	Null      = 0
	CtrlA     = 1
	CtrlB     = 2
	CtrlC     = 3
	CtrlD     = 4
	CtrlE     = 5
	CtrlF     = 6
	Bell      = 7
	CtrlH     = 8
	Tab       = 9
	CtrlJ     = 10
	Kill      = 11
	CtrlL     = 12
	Enter     = 13
	Next      = 14
	Prev      = 16
	BckSearch = 18
	FwdSearch = 19
	Transpose = 20
	CtrlU     = 21
	CtrlW     = 23
	CtrlY     = 25
	CtrlZ     = 26
	Esc       = 27
	Backspace = 127
)

func (rl *Readline) initializeEventMaps() {
	rl.eventMap = map[rune]func() error{
		Null:  rl.noop,
		CtrlA: rl.begin,
		CtrlB: rl.left,
		CtrlC: rl.interrupt,
		CtrlD: rl.eof,
		CtrlE: rl.end,
		CtrlF: rl.right,
		Bell:  rl.noop,
		CtrlH: rl.noop,
		Tab:   rl.tab,
		// CtrlJ:     rl.lf,
		Kill:      rl.noop,
		CtrlL:     rl.clear,
		Enter:     rl.enter,
		Next:      rl.noop,
		Prev:      rl.noop,
		BckSearch: rl.noop,
		FwdSearch: rl.noop,
		Transpose: rl.noop,
		CtrlU:     rl.noop,
		CtrlW:     rl.noop,
		CtrlY:     rl.noop,
		CtrlZ:     rl.noop,
		Esc:       rl.esc,
		Backspace: rl.backspace,
	}
	rl.escEventMap = map[string]func() error{
		"[200~": rl.pasteStart, // start paste
		"[201~": rl.pasteEnd,   // end paste

		"[2~": rl.noop,   // insert
		"[3~": rl.delete, // delete
		"[5~": rl.noop,   // page up
		"[6~": rl.noop,   // page down
		"[H":  rl.noop,   // home
		"[F":  rl.noop,   // end

		"[A": rl.prevHistory, // up
		"[B": rl.nextHistory, // down
		"[C": rl.right,       // right
		"[D": rl.left,        // left

		"OP":   rl.noop, // F1
		"OQ":   rl.noop, // F2
		"OR":   rl.noop, // F3
		"OS":   rl.noop, // F4
		"[15~": rl.noop, // F5
		"[17~": rl.noop, // F6
		"[18~": rl.noop, // F7
		"[19~": rl.noop, // F8
		"[20~": rl.noop, // F9
		"[21~": rl.noop, // F10
		"[23~": rl.noop, // F11
		"[24~": rl.noop, // F12
	}
}

func (rl *Readline) noop() error {
	return nil
}

// control actions

func (rl *Readline) interrupt() error {
	if len(rl.buf.data) == 0 {
		println("^C")
		return ErrInterrupt
	}

	rl.buf.resetState()
	println()
	return nil
}

func (rl *Readline) eof() error {
	if len(rl.buf.data) == 0 {
		return io.EOF
	}
	return nil
}

func (rl *Readline) esc() error {
	rl.isEsc = true
	return nil
}

func (rl *Readline) pasteStart() error {
	rl.isPaste = true
	return nil
}

func (rl *Readline) pasteEnd() error {
	rl.isPaste = false
	return nil
}

// cursor move

func (rl *Readline) left() error {
	if rl.buf.cursorIndex > 0 {
		rl.buf.cursorIndex--
	}
	return nil
}

func (rl *Readline) right() error {
	if rl.buf.cursorIndex < len(rl.buf.data) {
		rl.buf.cursorIndex++
	}
	return nil
}

func (rl *Readline) begin() error {
	rl.buf.cursorIndex = 0
	return nil
}

func (rl *Readline) end() error {
	rl.buf.cursorIndex = len(rl.buf.data)
	return nil
}

// edit actions

func (rl *Readline) backspace() error {
	if len(rl.buf.data) > 0 && rl.buf.cursorIndex > 0 {
		rl.buf.data = append(rl.buf.data[:rl.buf.cursorIndex-1], rl.buf.data[rl.buf.cursorIndex:]...)
		rl.buf.cursorIndex--
	}
	return nil
}

func (rl *Readline) delete() error {
	if rl.buf.cursorIndex < len(rl.buf.data) {
		rl.buf.data = append(rl.buf.data[:rl.buf.cursorIndex], rl.buf.data[rl.buf.cursorIndex+1:]...)
	}
	return nil
}

func (rl *Readline) clear() error {
	print("\x1b[H\x1b[2J") // clear screen
	return nil
}

func (rl *Readline) tab() error {
	rl.buf.insertRuneAtCursor(' ')
	rl.buf.insertRuneAtCursor(' ')
	rl.buf.insertRuneAtCursor(' ')
	rl.buf.insertRuneAtCursor(' ')
	return nil
}

func (rl *Readline) enter() error {
	if rl.isPaste {
		rl.buf.insertRuneAtCursor(CtrlJ)
		return nil
	} else {
		println()
		rl.history.Add(rl.buf.data)
		return ErrComplete
	}
}

func (rl *Readline) prevHistory() error {
	hist := rl.history.Prev()
	if hist != nil {
		rl.buf.data = []rune(hist)
		rl.buf.cursorIndex = len(rl.buf.data)
	}
	return nil
}

func (rl *Readline) nextHistory() error {
	hist := rl.history.Next()
	if hist != nil {
		rl.buf.data = []rune(hist)
		rl.buf.cursorIndex = len(rl.buf.data)
	}
	return nil
}
