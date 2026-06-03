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

package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/spf13/cobra"

	geniex_sdk "github.com/qcom-it-nexa-ai/geniex/bindings/go"
	"github.com/qcom-it-nexa-ai/geniex/cli/cmd/geniex/common"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/config"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/render"
)

// tagServerError tags transport-layer dial errors as ErrServerUnreachable
// so PrintError can render the "is geniex serve running?" hint. HTTP-level
// errors (4xx/5xx) flow through untouched.
func tagServerError(err error) error {
	var ne *net.OpError
	if errors.As(err, &ne) {
		return fmt.Errorf("%w: %v", common.ErrServerUnreachable, err)
	}
	return err
}

var client openai.Client

func run() *cobra.Command {
	runCmd := &cobra.Command{
		GroupID: "inference",
		Use:     "run <model-name>[:<precision>]",
		Short:   "Infer a model with server",
		Long:    "Infer a model with server. The server must be running and the model should be downloaded and cached locally. Append ':<precision>' to pick a specific precision.",
	}

	runCmd.Args = cobra.MatchAll(cobra.ExactArgs(1), cobra.OnlyValidArgs)
	for _, flags := range flagGroups {
		runCmd.Flags().AddFlagSet(flags)
	}

	runCmd.SetUsageFunc(flagGroupedUsage)

	runCmd.RunE = func(cmd *cobra.Command, args []string) error {
		name, quant := geniex_sdk.SplitNameQuant(args[0])
		fullName := name
		if quant != "" {
			fullName = name + ":" + quant
		}

		client = openai.NewClient(
			option.WithBaseURL(fmt.Sprintf("http://%s/v1", config.Get().Host)),
			// option.WithRequestTimeout(time.Second*15),
		)

		ctx := cmd.Context()
		// check the server is reachable before opening the REPL
		if _, err := client.Models.Get(ctx, fullName); err != nil {
			return tagServerError(err)
		}

		modelType, err := geniex_sdk.ModelGetType(name)
		if err != nil {
			return err
		}

		switch modelType {
		case geniex_sdk.ModelTypeLLM, geniex_sdk.ModelTypeVLM:
			return runCompletions(ctx, fullName, modelType)
		default:
			return fmt.Errorf("unsupported model type: %s", modelType)
		}
	}
	return runCmd
}

func runCompletions(ctx context.Context, name string, modelType geniex_sdk.ModelType) error {

	// warm up
	spin := render.NewSpinner("loading model...")
	spin.Start()
	warmUpRequest := openai.ChatCompletionNewParams{
		Messages: nil,
		Model:    name,
	}
	if systemPrompt != "" {
		warmUpRequest.Messages = append(warmUpRequest.Messages, openai.SystemMessage(systemPrompt))
	}
	_, err := client.Chat.Completions.New(ctx,
		warmUpRequest,
		option.WithJSONSet("ngl", ngl),
		option.WithJSONSet("nctx", nctx),
	)
	spin.Stop()

	if err != nil {
		return tagServerError(err)
	}

	// repl
	var history []openai.ChatCompletionMessageParamUnion
	if systemPrompt != "" {
		history = append(history, openai.SystemMessage(systemPrompt))
	}

	processor := &common.Processor{
		ParseFile: modelType == geniex_sdk.ModelTypeVLM,
		Verbose:   verbose,
		TestMode:  testMode,
		Run: func(prompt string, images, audios []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
			if len(images) > 0 || len(audios) > 0 {
				contents := make([]openai.ChatCompletionContentPartUnionParam, 0)
				contents = append(contents, openai.ChatCompletionContentPartUnionParam{
					OfText: &openai.ChatCompletionContentPartTextParam{
						Text: prompt,
					},
				})
				for _, image := range images {
					contents = append(contents, openai.ChatCompletionContentPartUnionParam{
						OfImageURL: &openai.ChatCompletionContentPartImageParam{
							ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
								URL: image,
							},
						},
					})
				}
				for _, audio := range audios {
					contents = append(contents, openai.ChatCompletionContentPartUnionParam{
						OfInputAudio: &openai.ChatCompletionContentPartInputAudioParam{
							InputAudio: openai.ChatCompletionContentPartInputAudioInputAudioParam{
								Data: audio,
							},
						},
					})
				}
				history = append(history, openai.UserMessage(contents))
			} else {
				history = append(history, openai.UserMessage(prompt))
			}

			start := time.Now()
			acc := openai.ChatCompletionAccumulator{}
			stream := client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
				Messages:            history,
				Model:               name,
				StreamOptions:       openai.ChatCompletionStreamOptionsParam{IncludeUsage: openai.Opt(true)},
				Temperature:         openai.Float(float64(temperature)),
				TopP:                openai.Float(float64(topP)),
				PresencePenalty:     openai.Float(float64(presencePenalty)),
				FrequencyPenalty:    openai.Float(float64(frequencyPenalty)),
				Seed:                openai.Int(int64(seed)),
				MaxCompletionTokens: openai.Int(int64(maxTokens)),
			},

				option.WithJSONSet("enable_think", enableThink),
				option.WithJSONSet("top_k", topK),
				option.WithJSONSet("min_p", minP),
				option.WithJSONSet("repetition_penalty", repetitionPenalty),
				option.WithJSONSet("grammar_path", grammarPath),
				option.WithJSONSet("grammar_string", grammarString),
				option.WithJSONSet("ngl", ngl),
				option.WithJSONSet("nctx", nctx),
				option.WithHeaderAdd("GenieX-KeepCache", "true"))

			var firstToken time.Time
			var profileData geniex_sdk.ProfileData
			for stream.Next() {
				if firstToken.IsZero() {
					firstToken = time.Now()
				}

				chunk := stream.Current()
				acc.AddChunk(chunk)
				if len(chunk.Choices) > 0 {
					if !onToken(chunk.Choices[0].Delta.Content) {
						stream.Close()
						break
					}
				}
				if chunk.Usage.PromptTokens > 0 {
					profileData.PromptTokens = chunk.Usage.PromptTokens
					profileData.GeneratedTokens = chunk.Usage.CompletionTokens
				}
			}

			// zero token generated
			if firstToken.IsZero() {
				firstToken = time.Now()
			}

			end := time.Now()
			profileData.TTFT = firstToken.Sub(start).Microseconds()
			profileData.PromptTime = profileData.TTFT
			profileData.DecodeTime = end.Sub(firstToken).Microseconds()
			profileData.DecodingSpeed = float64(profileData.GeneratedTokens) / float64(end.Sub(firstToken).Seconds())

			if stream.Err() != nil {
				return "", profileData, tagServerError(stream.Err())
			}

			if len(acc.Choices) > 0 {
				history = append(history, openai.AssistantMessage(acc.Choices[0].Message.Content))
				return acc.Choices[0].Message.Content, profileData, nil
			}

			return "", profileData, nil
		},
	}
	if len(prompt) > 0 || input != "" {
		processor.GetPrompt = getPromptOrInput
	} else {
		repl := common.Repl{
			Reset: func() error {
				history = nil
				_, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
					Messages: nil,
					Model:    name,
				})
				return err
			},
		}
		defer repl.Close()
		processor.GetPrompt = repl.GetPrompt
	}
	return processor.Process()
}
