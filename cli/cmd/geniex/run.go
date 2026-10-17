// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"time"

	"github.com/bytedance/sonic"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/spf13/cobra"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/cmd/geniex/common"
	"github.com/qualcomm/GenieX/cli/internal/config"
	"github.com/qualcomm/GenieX/cli/internal/render"
)

// tagServerError tags transport-layer dial errors as ErrServerUnreachable so
// PrintError renders the "is geniex serve running?" hint. HTTP errors pass through.
func tagServerError(err error) error {
	var ne *net.OpError
	if errors.As(err, &ne) {
		return fmt.Errorf("%w: %v", common.ErrServerUnreachable, err)
	}
	return err
}

// tagStreamError recovers the source SDKError from the `code` extension the
// server attaches to an SSE error event, so Processor can react to e.g.
// ErrLlmTokenizationContextLength. Transport errors fall back to tagServerError.
func tagStreamError(err error) error {
	var se *ssestream.StreamError
	if errors.As(err, &se) {
		var body struct {
			Code *int32 `json:"code"`
		}
		if sonic.Unmarshal(se.Event.Data, &body) == nil && body.Code != nil && *body.Code != -1 {
			return geniex_sdk.SDKError(*body.Code)
		}
	}
	return tagServerError(err)
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
		name, quant := geniex_sdk.SplitNamePrecision(args[0])
		fullName := geniex_sdk.JoinNamePrecision(name, quant)

		client = openai.NewClient(
			option.WithBaseURL(fmt.Sprintf("http://%s/v1", config.Get().Host)),
			// option.WithRequestTimeout(time.Second*15),
		)

		ctx := cmd.Context()
		// Doubles as the reachability probe. Only the server knows a remote model's type.
		var model struct {
			ModelType string `json:"model_type"`
		}
		if err := client.Get(ctx, "models/"+fullName, nil, &model); err != nil {
			return tagServerError(err)
		}
		modelType, ok := geniex_sdk.ParseModelType(model.ModelType)
		if !ok {
			return fmt.Errorf("server reported an unsupported model type %q for %s", model.ModelType, fullName)
		}

		return runCompletions(ctx, fullName, modelType)
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
		option.WithJSONSet("compute", computeUnit),
		option.WithJSONSet("vit_compute", vitComputeUnit),
		option.WithJSONSet("spec_type", specType),
		option.WithJSONSet("spec_draft_model", draftModel),
		option.WithJSONSet("spec_n_max", draftTokens),
		option.WithJSONSet("spec_n_min", draftMin),
		option.WithJSONSet("spec_p_min", draftPMin),
		option.WithJSONSet("power_mode", powerMode),
	)
	spin.Stop()

	if err != nil {
		return tagServerError(err)
	}

	// repl
	var history []openai.ChatCompletionMessageParamUnion
	protected := 0
	if systemPrompt != "" {
		history = append(history, openai.SystemMessage(systemPrompt))
		protected = 1
	}

	processor := &common.Processor{
		ParseFile: modelType == geniex_sdk.ModelTypeVLM,
		Verbose:   verbose,
		TestMode:  testMode,
		Reset: func() error {
			_, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
				Messages: nil,
				Model:    name,
			})
			if err == nil {
				history = history[:0]
				if protected == 1 {
					history = append(history, openai.SystemMessage(systemPrompt))
				}
			}
			return err
		},
		Run: func(prompt string, images, audios []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
			var userMessage openai.ChatCompletionMessageParamUnion
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
				userMessage = openai.UserMessage(contents)
			} else {
				userMessage = openai.UserMessage(prompt)
			}
			pending := append(append([]openai.ChatCompletionMessageParamUnion(nil), history...), userMessage)
			type attemptResult struct {
				text     string
				profile  geniex_sdk.ProfileData
				complete bool
			}
			var emitted bool
			result, kept, dropped, err := retryChatHistory(pending, protected,
				func(messages []openai.ChatCompletionMessageParamUnion) (attemptResult, error) {
					emitted = false

					start := time.Now()
					acc := openai.ChatCompletionAccumulator{}
					stream := client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
						Messages:            messages,
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
						option.WithJSONSet("compute", computeUnit),
						option.WithJSONSet("vit_compute", vitComputeUnit),
						option.WithJSONSet("spec_type", specType),
						option.WithJSONSet("spec_draft_model", draftModel),
						option.WithJSONSet("spec_n_max", draftTokens),
						option.WithJSONSet("spec_n_min", draftMin),
						option.WithJSONSet("spec_p_min", draftPMin),
						option.WithJSONSet("power_mode", powerMode))
					defer stream.Close()

					var firstToken time.Time
					var profileData geniex_sdk.ProfileData
					for stream.Next() {
						if firstToken.IsZero() {
							firstToken = time.Now()
						}

						chunk := stream.Current()
						acc.AddChunk(chunk)
						if len(chunk.Choices) > 0 {
							content := chunk.Choices[0].Delta.Content
							if content != "" {
								emitted = true
							}
							if !onToken(content) {
								stream.Close()
								break
							}
						}
						if chunk.Usage.PromptTokens > 0 {
							profileData.PromptTokens = chunk.Usage.PromptTokens
							profileData.GeneratedTokens = chunk.Usage.CompletionTokens
							det := chunk.Usage.CompletionTokensDetails
							profileData.DraftNAccepted = det.AcceptedPredictionTokens
							profileData.DraftNTotal = det.AcceptedPredictionTokens + det.RejectedPredictionTokens
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
						return attemptResult{profile: profileData}, tagStreamError(stream.Err())
					}

					if len(acc.Choices) > 0 {
						return attemptResult{text: acc.Choices[0].Message.Content, profile: profileData, complete: true}, nil
					}

					return attemptResult{profile: profileData}, nil
				}, func(err error) bool {
					return !emitted && errors.Is(err, geniex_sdk.ErrLlmGenerationPromptTooLong)
				})
			if err != nil {
				return result.text, result.profile, err
			}
			if dropped > 0 {
				fmt.Println(render.GetTheme().Info.Sprint("Older messages were removed to fit the context window."))
			}
			if result.complete {
				history = append(kept, openai.AssistantMessage(result.text))
			}
			return result.text, result.profile, nil
		},
	}
	if len(prompt) > 0 || input != "" {
		processor.GetPrompt = getPromptOrInput
	} else {
		repl := common.Repl{}
		repl.Reset = processor.Reset
		defer repl.Close()
		processor.GetPrompt = repl.GetPrompt
	}
	return processor.Process()
}
