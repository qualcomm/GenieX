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
)

type Config struct {
	Prompt      string
	AltPrompt   string
	HistoryFile string
}

type Readline struct {
	term    *Terminal
	buf     *Buffer
	history *History

	eventMap map[rune]func() error

	// csi
	isEsc       bool
	isEscEx     bool
	escBuf      string
	escEventMap map[string]func() error
	isPaste     bool
}

func New(config *Config) (*Readline, error) {
	term, err := NewTerminal()
	if err != nil {
		return nil, err
	}

	buf := NewBuffer(
		config.Prompt,
		config.AltPrompt,
	)

	hist := NewHistory(config.HistoryFile)

	rl := Readline{
		term:    term,
		buf:     buf,
		history: hist,
	}
	rl.initializeEventMaps()
	return &rl, nil
}

func (rl *Readline) Read() (string, error) {
	rl.buf.resetState()
	rl.buf.refresh()

	if err := rl.term.EnterRaw(); err != nil {
		return "", err
	}
	defer rl.term.ExitRaw()

	for {
		r, err := rl.term.Read()
		if err != nil {
			return "", err
		}

		if err := rl.parse(r); err != nil {
			if errors.Is(err, ErrComplete) {
				return string(rl.buf.data), nil
			}
			return "", err
		}
	}
}

func (rl *Readline) parse(r rune) error {
	if rl.isEsc {
		// escape sequence

		rl.isEsc = false

		switch r {
		case 'O': // VT100
			rl.isEscEx = true
			rl.escBuf = "O"
			return nil
		case '[': // CSI
			rl.isEscEx = true
			rl.escBuf = "["
			return nil
		}

	} else if rl.isEscEx {
		// escape ex sequence

		if r < 0x20 || r >= 0x80 {
			// invalid char, end escape ex
			rl.isEscEx = false
			return nil
		}
		rl.escBuf += string(r)
		if r >= 0x40 {
			// end of escape ex
			rl.isEscEx = false
			if event, ok := rl.escEventMap[rl.escBuf]; ok {
				if err := event(); err != nil {
					return err
				}
			} else {
				// print("unknown escape sequence: " + rl.escBuf + "\n") // debug
			}
		}

	} else {
		// single char

		if event, ok := rl.eventMap[r]; !ok {
			rl.buf.insertRuneAtCursor(r)
		} else if err := event(); err != nil {
			return err
		}
	}

	rl.buf.refresh()
	return nil
}

func (rl *Readline) Close() error {
	return rl.term.Close()
}
