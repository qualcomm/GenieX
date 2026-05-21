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

package common

import (
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/readline"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/render"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/store"
)

var baseHelp = [][2]string{
	{"/?, /h, /help", "Show this help message"},
	{"/exit", "Exit the REPL"},
	{"/clear", "Clear the screen and conversation history"},
}

var micHelp = [2]string{"/mic", "Record audio for transcription"}

type Repl struct {
	Reset func() error

	Record          func() (*string, error)
	RecordImmediate bool

	init bool
	rl   *readline.Readline
}

// ========= repl tool ========

func (r *Repl) GetPrompt() (string, error) {
	if !r.init {
		// fill default functions
		if r.Reset == nil {
			r.Reset = func() error { return nil }
		}

		// init readline
		config := &readline.Config{
			Prompt:      render.GetTheme().Prompt.Sprint("> "),
			AltPrompt:   render.GetTheme().Prompt.Sprint(". "),
			Placeholder: render.GetTheme().Placeholder.Sprint("Send a message (/? for help)"),
			HistoryFile: filepath.Join(store.Get().DataPath(), "history"),
		}
		rl, err := readline.New(config)
		if err != nil {
			return "", fmt.Errorf("init readline: %w", err)
		}
		r.rl = rl

		r.init = true
	}

	var recordAudios []string

	for {
		// print stashed content
		if len(recordAudios) > 0 {
			fmt.Println(render.GetTheme().Info.Sprintf("Current stash audios: %s", strings.Join(recordAudios, ", ")))
		}

		line, err := r.rl.Read()

		// check err or exit
		switch {
		case errors.Is(err, io.EOF):
			fmt.Println()
			return "", io.EOF
		case errors.Is(err, readline.ErrInterrupt):
			if line == "" {
				fmt.Println("\nUse Ctrl + d or /exit to exit.")
				fmt.Println()
			}
			continue
		case err != nil:
			return "", err
		}

		// check if it's a command
		var fields []string
		if !strings.HasPrefix(line, "/") {
			if len(recordAudios) > 0 {
				line += " " + strings.Join(recordAudios, " ")
			}
			recordAudios = nil // clear stashed audios after use
			return line, nil
		}
		fields = strings.Fields(strings.TrimSpace(line))
		if strings.Contains(fields[0][1:], "/") || strings.Contains(fields[0], ".") {
			if len(recordAudios) > 0 {
				line += " " + strings.Join(recordAudios, " ")
			}
			recordAudios = nil // clear stashed audios after use
			return line, nil
		}

		switch fields[0] {
		case "/?", "/h", "/help":
			fmt.Println("Commands:")
			cmds := baseHelp
			if r.Record != nil {
				cmds = append(cmds, micHelp)
			}
			for _, h := range cmds {
				fmt.Printf("  %-25s %s\n", h[0], h[1])
			}
			fmt.Println()
			continue

		case "/exit":
			return "", io.EOF

		case "/clear":
			r.Reset()
			recordAudios = nil
			fmt.Print("\033[H\033[2J")
			continue

		case "/mic":
			if r.Record == nil {
				fmt.Println(render.GetTheme().Error.Sprintf("Unknown command: %s", fields[0]))
				fmt.Println()
				continue
			}
			outputFile, err := r.Record()
			if err != nil {
				fmt.Println(render.GetTheme().Error.Sprintf("Error: %s", err))
				fmt.Println()
			}
			if outputFile != nil {
				if r.RecordImmediate {
					return *outputFile, nil
				}
				recordAudios = append(recordAudios, *outputFile)
			}
			continue

		default:
			fmt.Println(render.GetTheme().Error.Sprintf("Unknown command: %s", fields[0]))
			fmt.Println()
			continue
		}
	}
}

func (r *Repl) Close() {
	if r.init && r.rl != nil {
		r.rl.Close()
	}
}
