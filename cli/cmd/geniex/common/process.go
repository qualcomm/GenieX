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
	"log/slog"
	"os"
	"os/signal"
	"path"
	"regexp"
	"strings"

	"github.com/bytedance/sonic"

	geniex_sdk "github.com/qcom-it-nexa-ai/geniex/bindings/go"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/render"
)

var (
	ErrNoAudio               = errors.New("no audio file provided")
	ErrNoImage               = errors.New("no image file provided")
	ErrContextLengthExceeded = errors.New("model context length exceeded; output is truncated")
)

type Processor struct {
	ParseFile bool

	Verbose  bool
	TestMode bool

	GetPrompt func() (string, error)
	Run       func(prompt string, images, audios []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error)

	fsm      map[[2]any][2]any
	fsmState int
}

func (p *Processor) Process() error {
	var stopGen bool
	go func() {
		cSignal := make(chan os.Signal, 1)
		signal.Notify(cSignal, os.Interrupt)
		for range cSignal {
			slog.Warn("interrupt signal received")
			stopGen = true
		}
	}()

	for {
		line, err := p.GetPrompt()
		slog.Debug("GetPrompt", "line", line, "err", err)

		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}

		var prompt string
		var images, audios []string
		if p.ParseFile {
			var parseErr error
			prompt, images, audios, parseErr = p.parseFiles(line)
			if parseErr != nil {
				fmt.Println(render.GetTheme().Error.Sprintf("Error: %s", parseErr))
				fmt.Println()
				continue
			}
		} else {
			prompt = line
		}

		// run async
		firstToken := true
		spin := render.NewSpinner("encoding...") // merge into fsm
		spin.Start()

		p.fsmInit()
		stopGen = false
		output, profileData, err := p.Run(prompt, images, audios, func(token string) bool {
			if firstToken {
				spin.Stop()
				firstToken = false
			}

			p.fsmEvent(token)
			return !stopGen
		})

		slog.Debug("profileData", "profileData", profileData)
		if p.TestMode {
			pd, _ := sonic.MarshalString(map[string]any{
				"Input":   line,
				"Output":  output,
				"Profile": profileData,
				"Error":   err,
			})
			fmt.Fprintln(os.Stderr, pd)
		}

		// reset spin when no token received
		if firstToken {
			spin.Stop()
		}

		// reset color
		render.GetTheme().Reset()
		fmt.Println()
		fmt.Println()
		p.printProfile(profileData)

		switch {
		case err == nil:
		case errors.Is(err, ErrNoAudio):
			fmt.Println(render.GetTheme().Error.Sprintf("No audio file provided, please provide an audio file or use /mic command"))
			fmt.Println()
		case errors.Is(err, ErrNoImage):
			fmt.Println(render.GetTheme().Error.Sprintf("No image file provided, please provide an image file"))
			fmt.Println()
		case errors.Is(err, ErrContextLengthExceeded):
			fmt.Println(render.GetTheme().Error.Sprintf("Model context length exceeded; output above is truncated"))
			fmt.Println()
		default:
			return err
		}
	}
}

// file name parse

var fileRegex = regexp.MustCompile(`(?:[a-zA-Z]:)?(?:\./|/|\\)[\S\\ ]+?\.(?i:jpg|jpeg|png|webp|mp3|wav)\b`)

var shellUnescaper = strings.NewReplacer(
	`\ `, ` `,
	`\(`, `(`,
	`\)`, `)`,
	`\[`, `[`,
	`\]`, `]`,
	`\{`, `{`,
	`\}`, `}`,
	`\$`, `$`,
	`\&`, `&`,
	`\;`, `;`,
	`\'`, `'`,
	`\\`, `\`,
	`\*`, `*`,
	`\?`, `?`,
	`\~`, `~`,
)

func (p *Processor) parseFiles(prompt string) (string, []string, []string, error) {
	files := fileRegex.FindAllString(prompt, -1)
	images := make([]string, 0, len(files))
	audios := make([]string, 0, len(files))

	for _, file := range files {
		realFile := shellUnescaper.Replace(file)

		if _, err := os.Stat(realFile); err != nil {
			return "", nil, nil, fmt.Errorf("file not found: %s", realFile)
		}

		switch path.Ext(realFile) {
		case ".mp3", ".wav":
			audios = append(audios, realFile)
			slog.Debug("add audio", "file", realFile)
		default:
			images = append(images, realFile)
			slog.Debug("add image", "file", realFile)
		}

		for _, quoted := range []string{`"` + file + `"`, `'` + file + `'`, `"` + realFile + `"`, `'` + realFile + `'`} {
			prompt = strings.ReplaceAll(prompt, quoted, "")
		}
		prompt = strings.ReplaceAll(prompt, file, "")
	}
	return strings.TrimSpace(prompt), images, audios, nil
}

// output parse

const (
	STATE_ASSISTANT = iota // init state
	STATE_THINK
	STATE_NORMAL

	STATE_START
	STATE_CHANNEL
	STATE_ANALYSIS
	STATE_FINAL
	STATE_END
)

func (p *Processor) fsmInit() {
	thinkStart := func(extraLine bool) func() {
		return func() {
			render.GetTheme().Set(render.GetTheme().ThinkOutput)
			if extraLine {
				fmt.Print("<think>\n")
			} else {
				fmt.Print("<think>")
			}
		}
	}
	thinkEnd := func(extraLine bool) func() {
		return func() {
			if extraLine {
				fmt.Print("\n</think>\n\n")
			} else {
				fmt.Print("</think>")
			}
			render.GetTheme().Set(render.GetTheme().ModelOutput)
		}
	}
	p.fsm = map[[2]any][2]any{
		// normal
		{STATE_ASSISTANT, "<think>"}: {STATE_THINK, thinkStart(false)},
		{STATE_THINK, "</think>"}:    {STATE_NORMAL, thinkEnd(false)},

		// gpt-oss
		{STATE_ASSISTANT, "<|channel|>"}: {STATE_CHANNEL, nil},
		{STATE_CHANNEL, "analysis"}:      {STATE_ANALYSIS, nil},
		{STATE_CHANNEL, "final"}:         {STATE_FINAL, nil},
		{STATE_ANALYSIS, "<|message|>"}:  {STATE_THINK, thinkStart(true)},
		{STATE_FINAL, "<|message|>"}:     {STATE_NORMAL, nil},
		{STATE_THINK, "<|end|>"}:         {STATE_END, thinkEnd(true)},
		{STATE_NORMAL, "<|end|>"}:        {STATE_END, nil},
		{STATE_END, "<|start|>"}:         {STATE_START, nil},
		{STATE_START, "assistant"}:       {STATE_ASSISTANT, nil},
	}
	p.fsmState = STATE_ASSISTANT
}

func (p *Processor) fsmEvent(token string) {
	next, ok := p.fsm[[2]any{p.fsmState, token}]
	if ok {
		p.fsmState = next[0].(int)
		if next[1] != nil {
			next[1].(func())()
		}
		return
	}

	fmt.Print(token)
}

// print profile data

func (p *Processor) printProfile(pd geniex_sdk.ProfileData) {
	var text string

	if p.Verbose {
		text = fmt.Sprintf(
			strings.TrimSpace(`
total time:     %fs
ttft:           %fs
prompt time:    %fs
prompt tokens:  %d token(s)
prompt speed:   %f tok/s
decode time:    %fs
decode tokens:  %d token(s)
decode speed:   %f tok/s
stop reason:    %s
			`),
			float64(pd.TotalTimeUs())/1e6,
			float64(pd.TTFT)/1e6,
			float64(pd.PromptTime)/1e6,
			pd.PromptTokens,
			pd.PrefillSpeed,
			float64(pd.DecodeTime)/1e6,
			pd.GeneratedTokens,
			pd.DecodingSpeed,
			pd.StopReason,
		)

	} else {
		if pd.AudioDuration > 0 { // ASR TTS
			text = fmt.Sprintf("processing_time %.2fs  |  audio_duration %.2fs  |  RTF %.2f (%.1fx realtime)",
				float64(pd.TotalTimeUs())/1e6,
				float64(pd.AudioDuration)/1e6,
				pd.RealTimeFactor,
				1.0/pd.RealTimeFactor)

		} else if pd.DecodingSpeed != 0 {
			text = fmt.Sprintf("— %.1f tok/s • %d tok • %.1f s first token —",
				pd.DecodingSpeed,
				pd.GeneratedTokens,
				float64(pd.TTFT)/1e6)

		} else {
			if pd.TotalTimeUs() != 0 {
				text = fmt.Sprintf("- %.1f s -",
					float64(pd.TotalTimeUs())/1e6,
				)
			}
		}
	}
	if text == "" {
		return
	}

	fmt.Println(render.GetTheme().Profile.Sprint(text))
	fmt.Println()
}
