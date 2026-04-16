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
	"bufio"
	"os"
)

type Terminal struct {
	oldState Termios
	state    Termios
	r        *bufio.Reader
}

func NewTerminal() (*Terminal, error) {
	t := &Terminal{}

	termios, err := getTermios()
	if err != nil {
		return nil, err
	}

	t.oldState = *termios
	t.state = *termios
	applyRawMode(&t.state)

	t.r = bufio.NewReader(os.Stdin)

	return t, nil
}

func (t *Terminal) Read() (rune, error) {
	r, _, err := t.r.ReadRune()
	return r, err
}

func (t *Terminal) Close() error {
	return t.ExitRaw()
}

func (t *Terminal) EnterRaw() error {
	err := setTermios(&t.state)
	if err != nil {
		return err
	}
	print("\x1b[?2004h") // enable bracketed paste mode
	return nil
}

func (t *Terminal) ExitRaw() error {
	print("\x1b[?2004l") // disable bracketed paste mode
	return setTermios(&t.oldState)
}
