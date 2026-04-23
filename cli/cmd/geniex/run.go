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
	// "encoding/base64"
	// "errors"
	"fmt"
	// "io"
	"net"
	"net/http"
	"os"
	// "path/filepath"
	// "strconv"
	"strings"
	"time"
	"unicode"

	"github.com/bytedance/sonic"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/spf13/cobra"

	geniex_sdk "github.com/qcom-it-nexa-ai/geniex/bindings/go"
	"github.com/qcom-it-nexa-ai/geniex/cli/cmd/geniex/common"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/config"
	// "github.com/qcom-it-nexa-ai/geniex/cli/internal/record"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/render"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/types"
)

var client openai.Client

func run() *cobra.Command {
	runCmd := &cobra.Command{
		GroupID: "inference",
		Use:     "run <model-name>",
		Short:   "Infer a model with server",
		Long:    "Infer a model with server. The server must be running and the model should be downloaded and cached locally.",
	}

	runCmd.Args = cobra.MatchAll(cobra.ExactArgs(1), cobra.OnlyValidArgs)
	for _, flags := range flagGroups {
		runCmd.Flags().AddFlagSet(flags)
	}

	runCmd.SetUsageFunc(func(c *cobra.Command) error {
		w := c.OutOrStdout()
		fmt.Fprint(w, "Usage:")
		if c.Runnable() {
			fmt.Fprintf(w, "\n  %s", c.UseLine())
		}
		if len(c.Aliases) > 0 {
			fmt.Fprintf(w, "\n\nAliases:\n")
			fmt.Fprintf(w, "  %s", c.NameAndAliases())
		}

		for _, flags := range flagGroups {
			fmt.Fprintf(w, "\n\n%s Flags:\n", flags.Name())
			fmt.Fprint(w, strings.TrimRightFunc(flags.FlagUsages(), unicode.IsSpace))
		}

		if c.HasAvailableInheritedFlags() {
			fmt.Fprintf(w, "\n\nGlobal Flags:\n")
			fmt.Fprint(w, strings.TrimRightFunc(c.InheritedFlags().FlagUsages(), unicode.IsSpace))
		}
		fmt.Fprintln(w)
		return nil
	})

	runCmd.Run = func(cmd *cobra.Command, args []string) {
		name, quant := normalizeModelName(args[0])
		if quant != "" {
			name = name + ":" + quant
		}

		client = openai.NewClient(
			option.WithBaseURL(fmt.Sprintf("http://%s/v1", config.Get().Host)),
			// option.WithRequestTimeout(time.Second*15),
		)

		// check
		modelInfo, err := client.Models.Get(context.TODO(), name)
		if err != nil {
			if _, ok := err.(net.Error); ok {
				fmt.Println(render.GetTheme().Error.Sprintf("Is server running? Please check your network. \n\t%s", err))
				os.Exit(1)
			}
			if e, ok := err.(*openai.Error); ok && e.StatusCode == http.StatusNotFound {
				fmt.Println(render.GetTheme().Error.Sprintf("Model or quant not found: %s, Please download first", name))
				os.Exit(1)
			} else {
				fmt.Println(render.GetTheme().Error.Sprintf("get model error: %s", err.Error()))
				os.Exit(1)
			}
		}

		var manifest types.ModelManifest
		sonic.UnmarshalString(modelInfo.RawJSON(), &manifest)

		switch manifest.ModelType {
		case types.ModelTypeLLM, types.ModelTypeVLM:
			err = runCompletions(manifest, quant)
		// case types.ModelTypeEmbedder:
		// 	err = runEmbeddings(manifest, quant)
		// case types.ModelTypeReranker:
		// 	err = runReranking(manifest, quant)
		// case types.ModelTypeTTS:
		// 	err = runAudioSpeech(manifest, quant)
		// case types.ModelTypeASR:
		// 	err = runAudioTranscription(manifest, quant)
		// case types.ModelTypeDiarize:
		// 	err = runAudioDiarize(manifest, quant)
		// case types.ModelTypeCV:
		// 	err = runCV(manifest, quant)
		// case types.ModelTypeImageGen:
		// 	// ImageGen model is a directory, not a file
		// 	err = runImagesGenerations(manifest, quant)
		default:
			panic("not support model type")
		}

		switch err {
		case nil:
			os.Exit(0)
		default:
			fmt.Println(render.GetTheme().Error.Sprintf("Error: %s", err))
			os.Exit(1)
		}
	}
	return runCmd
}

func runCompletions(manifest types.ModelManifest, quant string) error {
	name := manifest.Name
	if quant != "" {
		name = name + ":" + quant
	}

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
	_, err := client.Chat.Completions.New(context.TODO(),
		warmUpRequest,
		option.WithJSONSet("ngl", ngl),
		option.WithJSONSet("nctx", nctx),
	)
	spin.Stop()

	if err != nil {
		return err
	}

	// repl
	var history []openai.ChatCompletionMessageParamUnion
	if systemPrompt != "" {
		history = append(history, openai.SystemMessage(systemPrompt))
	}

	processor := &common.Processor{
		HideThink: hideThink,
		ParseFile: manifest.ModelType == types.ModelTypeVLM,
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
			stream := client.Chat.Completions.NewStreaming(context.Background(), openai.ChatCompletionNewParams{
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
				option.WithJSONSet("enable_json", enableJson),

				option.WithJSONSet("ngl", ngl),
				option.WithJSONSet("nctx", nctx),
				option.WithJSONSet("enable_think", enableThink),

				option.WithJSONSet("image_max_length", imageMaxLength),

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
				return "", profileData, stream.Err()
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
				_, err := client.Chat.Completions.New(context.TODO(), openai.ChatCompletionNewParams{
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

// func runEmbeddings(manifest types.ModelManifest, quant string) error {
// 	name := manifest.Name
// 	if quant != "" {
// 		name = name + ":" + quant
// 	}
//
// 	// warm up
// 	spin := render.NewSpinner("loading model...")
// 	spin.Start()
// 	warmUpRequest := openai.EmbeddingNewParams{
// 		Model: name,
// 		Input: openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: []string{}},
// 	}
// 	_, err := client.Embeddings.New(context.TODO(), warmUpRequest)
// 	spin.Stop()
//
// 	if err != nil {
// 		return err
// 	}
//
// 	processor := &common.Processor{
// 		ParseFile: manifest.ModelType == types.ModelTypeVLM,
// 		Verbose:   verbose,
// 		TestMode:  testMode,
// 		Run: func(prompt string, _, _ []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
// 			start := time.Now()
//
// 			res, err := client.Embeddings.New(context.TODO(), openai.EmbeddingNewParams{
// 				Model: name,
// 				Input: openai.EmbeddingNewParamsInputUnion{
// 					OfString: openai.String(prompt),
// 				},
// 			},
// 				option.WithJSONSet("task_type", taskType),
// 			)
//
// 			duration := time.Since(start).Microseconds()
// 			profileData := geniex_sdk.ProfileData{
// 				TTFT:            duration,
// 				PromptTime:      0,
// 				DecodeTime:      duration,
// 				PromptTokens:    res.Usage.PromptTokens,
// 				GeneratedTokens: res.Usage.TotalTokens - res.Usage.PromptTokens,
// 				AudioDuration:   0,
// 				PrefillSpeed:    0,
// 				DecodingSpeed:   float64(res.Usage.TotalTokens-res.Usage.PromptTokens) / (float64(duration) / 1e6),
// 			}
//
// 			if err != nil {
// 				return "", profileData, err
// 			}
//
// 			emb := res.Data[0].Embedding
// 			n := len(emb)
// 			info := render.GetTheme().Info.Sprintf("Embedding")
// 			var out string
// 			if len(emb) > 6 {
// 				emb := res.Data[0].Embedding
// 				out = render.GetTheme().Success.Sprintf(
// 					"[%.6f, %.6f, %.6f, ..., %.6f, %.6f, %.6f] (length: %d)",
// 					emb[0], emb[1], emb[2],
// 					emb[n-3], emb[n-2], emb[n-1], n,
// 				)
// 			} else {
// 				out = render.GetTheme().Success.Sprintf("%v (length: %d)", emb, n)
// 			}
//
// 			data := fmt.Sprintf("%s: %s", info, out)
// 			onToken(data)
// 			return data, profileData, err
//
// 		},
// 	}
// 	if len(prompt) > 0 || input != "" {
// 		processor.GetPrompt = getPromptOrInput
// 	} else {
// 		repl := common.Repl{}
// 		defer repl.Close()
// 		processor.GetPrompt = repl.GetPrompt
// 	}
// 	return processor.Process()
// }

// func runReranking(manifest types.ModelManifest, quant string) error {
// 	name := manifest.Name
// 	if quant != "" {
// 		name = name + ":" + quant
// 	}
//
// 	// warm up
// 	spin := render.NewSpinner("loading model...")
// 	spin.Start()
// 	warmUpRequest := map[string]any{
// 		"model": name,
// 	}
// 	err := client.Post(context.TODO(), "reranking", warmUpRequest, nil)
// 	spin.Stop()
//
// 	if err != nil {
// 		return err
// 	}
//
// 	const SEP = "\\n"
// 	processor := &common.Processor{
// 		ParseFile: manifest.ModelType == types.ModelTypeVLM,
// 		Verbose:   verbose,
// 		TestMode:  testMode,
// 		Run: func(prompt string, _, _ []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
// 			parsedPrompt := strings.Split(prompt, SEP)
// 			if len(parsedPrompt) < 2 {
// 				return "", geniex_sdk.ProfileData{}, fmt.Errorf("parsed prompt failed, query and document are required for reranking")
// 			}
//
// 			query := parsedPrompt[0]
// 			document := parsedPrompt[1:]
// 			fmt.Println(render.GetTheme().Info.Sprintf("Query: %s", query))
// 			fmt.Println(render.GetTheme().Info.Sprintf("Processing %d documents", len(document)))
//
// 			start := time.Now()
//
// 			res := struct {
// 				Result []float32 `json:"result"`
// 			}{}
// 			err := client.Post(context.TODO(), "reranking", map[string]any{
// 				"model":     name,
// 				"query":     query,
// 				"documents": document,
// 			}, &res)
//
// 			duration := time.Since(start).Microseconds()
// 			profileData := geniex_sdk.ProfileData{
// 				TTFT:       duration,
// 				PromptTime: 0,
// 				DecodeTime: duration,
// 				// PromptTokens:    res.Usage.PromptTokens,
// 				// GeneratedTokens: res.Usage.TotalTokens - res.Usage.PromptTokens,
// 				AudioDuration: 0,
// 				PrefillSpeed:  0,
// 				// DecodingSpeed:   float64(res.Usage.TotalTokens-res.Usage.PromptTokens) / (float64(duration) / 1e6),
// 			}
//
// 			if err != nil {
// 				return "", profileData, err
// 			}
//
// 			fmt.Println(render.GetTheme().Success.Sprintf("✓ Reranking completed successfully. Generated %d scores", len(res.Result)))
//
// 			// Display results
// 			var data strings.Builder
// 			for i, doc := range document {
// 				if i < len(res.Result) {
// 					line := fmt.Sprintf("\n%s [%d]: %s\n", render.GetTheme().Info.Sprintf("Document"), i+1, doc)
// 					onToken(line)
// 					data.WriteString(line)
// 					line = fmt.Sprintf("%s: %.6f\n", render.GetTheme().Info.Sprintf("Score"), res.Result[i])
// 					onToken(line)
// 					data.WriteString(line)
// 				}
// 			}
// 			return data.String(), profileData, err
// 		},
// 	}
//
// 	if query != "" || len(document) > 0 {
// 		if query == "" || len(document) == 0 {
// 			fmt.Println(render.GetTheme().Error.Sprintf("query and document are required for reranking"))
// 			return errors.New("query and document are required for reranking")
// 		}
// 		processor.GetPrompt = func() (string, error) {
// 			if query == "" || len(document) == 0 {
// 				return "", io.EOF
// 			}
// 			prompt := strings.Join(append([]string{query}, document...), SEP)
// 			query, document = "", nil
// 			fmt.Print(render.GetTheme().Prompt.Sprintf("> "))
// 			fmt.Println(render.GetTheme().Normal.Sprint(prompt))
// 			return prompt, nil
// 		}
// 	} else {
// 		repl := common.Repl{}
// 		defer repl.Close()
// 		processor.GetPrompt = repl.GetPrompt
// 	}
// 	return processor.Process()
// }

// func runAudioSpeech(manifest types.ModelManifest, quant string) error {
// 	name := manifest.Name
// 	if quant != "" {
// 		name = name + ":" + quant
// 	}
//
// 	// warm up
// 	spin := render.NewSpinner("loading model...")
// 	spin.Start()
// 	warmUpRequest := openai.AudioSpeechNewParams{
// 		Model: name,
// 		Input: "",
// 	}
// 	_, err := client.Audio.Speech.New(context.TODO(), warmUpRequest)
// 	spin.Stop()
//
// 	if err != nil {
// 		return err
// 	}
//
// 	// TODO: support list voice over server
// 	if listVoice {
// 		return fmt.Errorf("not implemented")
// 	}
//
// 	processor := &common.Processor{
// 		Verbose:  verbose,
// 		TestMode: testMode,
// 		Run: func(prompt string, _, _ []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
//
// 			start := time.Now()
//
// 			textToSynthesize := strings.TrimSpace(prompt)
// 			if textToSynthesize == "" {
// 				return "", geniex_sdk.ProfileData{}, fmt.Errorf("prompt cannot be empty")
// 			}
//
// 			// Generate output filename if not specified
// 			outputFile := output
// 			if outputFile == "" {
// 				outputFile = fmt.Sprintf("tts_output_%d.wav", time.Now().Unix())
// 			}
//
// 			res, err := client.Audio.Speech.New(context.TODO(), openai.AudioSpeechNewParams{
// 				Model: name,
// 				Voice: openai.AudioSpeechNewParamsVoice(voice),
// 				Speed: openai.Float(float64(speechSpeed)),
// 			})
//
// 			duration := time.Since(start).Microseconds()
// 			profileData := geniex_sdk.ProfileData{
// 				TTFT:       duration,
// 				DecodeTime: duration,
// 			}
//
// 			if err != nil {
// 				return "", profileData, err
// 			}
//
// 			// Save audio to filename
// 			audioData, err := io.ReadAll(res.Body)
// 			if err != nil {
// 				return "", profileData, err
// 			}
// 			err = os.WriteFile(outputFile, audioData, 0644)
// 			if err != nil {
// 				return "", profileData, err
// 			}
//
// 			data := render.GetTheme().Success.Sprintf("✓ Audio saved: %s", outputFile)
// 			onToken(data)
// 			return data, profileData, err
//
// 		},
// 	}
// 	if len(prompt) > 0 || input != "" {
// 		processor.GetPrompt = getPromptOrInput
// 	} else {
// 		repl := common.Repl{}
// 		defer repl.Close()
// 		processor.GetPrompt = repl.GetPrompt
// 	}
// 	return processor.Process()
// }

// func runAudioTranscription(manifest types.ModelManifest, quant string) error {
// 	name := manifest.Name
// 	if quant != "" {
// 		name = name + ":" + quant
// 	}
//
// 	// warm up
// 	spin := render.NewSpinner("loading model...")
// 	spin.Start()
// 	warmUpRequest := openai.AudioTranscriptionNewParams{
// 		Model: name,
// 	}
// 	_, err := client.Audio.Transcriptions.New(context.TODO(), warmUpRequest)
// 	spin.Stop()
//
// 	if err != nil {
// 		return err
// 	}
//
// 	processor := &common.Processor{
// 		ParseFile: true,
// 		Verbose:   verbose,
// 		TestMode:  testMode,
// 		Run: func(prompt string, _, audios []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
// 			if len(audios) == 0 {
// 				return "", geniex_sdk.ProfileData{}, common.ErrNoAudio
// 			}
// 			if len(audios) > 1 {
// 				return "", geniex_sdk.ProfileData{}, fmt.Errorf("only one audio file is supported")
// 			}
//
// 			// send request
// 			file, err := os.Open(audios[0])
// 			if err != nil {
// 				return "", geniex_sdk.ProfileData{}, fmt.Errorf("open audio file error: %s", err.Error())
// 			}
// 			defer file.Close()
//
// 			start := time.Now()
// 			res, err := client.Audio.Transcriptions.New(context.TODO(), openai.AudioTranscriptionNewParams{
// 				Model: name,
// 				File:  file,
// 			})
// 			duration := time.Since(start).Microseconds()
//
// 			profileData := geniex_sdk.ProfileData{
// 				TTFT:       duration,
// 				PromptTime: 0,
// 				DecodeTime: duration,
// 			}
//
// 			if err != nil {
// 				return "", profileData, err
// 			}
// 			onToken(res.Text)
// 			return res.Text, profileData, err
// 		},
// 	}
// 	if input != "" {
// 		processor.GetPrompt = func() (string, error) {
// 			if input == "" {
// 				return "", io.EOF
// 			}
// 			audioPath := input
// 			input = ""
// 			fmt.Print(render.GetTheme().Prompt.Sprintf("> "))
// 			fmt.Println(render.GetTheme().Normal.Sprint(audioPath))
// 			return audioPath, nil
// 		}
// 	} else {
// 		repl := common.Repl{
// 			RecordImmediate: true,
// 			Record: func() (*string, error) {
// 				t := strconv.Itoa(int(time.Now().Unix()))
// 				outputFile := filepath.Join(os.TempDir(), "geniex-cli", t+".wav")
// 				rec, err := record.NewRecorder(outputFile)
// 				if err != nil {
// 					return nil, err
// 				}
//
// 				fmt.Println(render.GetTheme().Info.Sprint("Recording is going on, press Ctrl-C to stop"))
//
// 				err = rec.Run()
// 				if err != nil {
// 					return nil, err
// 				}
//
// 				return &outputFile, nil
// 			},
// 		}
// 		defer repl.Close()
// 		processor.GetPrompt = repl.GetPrompt
// 	}
// 	return processor.Process()
// }

// func runAudioDiarize(manifest types.ModelManifest, quant string) error {
// 	name := manifest.Name
// 	if quant != "" {
// 		name = name + ":" + quant
// 	}
//
// 	// warm up
// 	spin := render.NewSpinner("loading model...")
// 	spin.Start()
// 	warmUpRequest := map[string]any{
// 		"model": name,
// 	}
// 	err := client.Post(context.TODO(), "audio/diarize", warmUpRequest, nil)
// 	spin.Stop()
//
// 	if err != nil {
// 		return err
// 	}
//
// 	processor := &common.Processor{
// 		ParseFile: true,
// 		Verbose:   verbose,
// 		TestMode:  testMode,
// 		Run: func(_ string, _, audios []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
// 			if len(audios) == 0 {
// 				return "", geniex_sdk.ProfileData{}, common.ErrNoAudio
// 			}
// 			if len(audios) > 1 {
// 				return "", geniex_sdk.ProfileData{}, fmt.Errorf("only one audio file is supported")
// 			}
//
// 			// base64 encode audio
// 			audioData, err := os.ReadFile(audios[0])
// 			if err != nil {
// 				return "", geniex_sdk.ProfileData{}, fmt.Errorf("read audio file error: %s", err.Error())
// 			}
// 			mimeType := http.DetectContentType(audioData)
// 			b64Audio := base64.StdEncoding.EncodeToString(audioData)
// 			audioStr := fmt.Sprintf("data:%s;base64,%s", mimeType, b64Audio)
//
// 			// send request
// 			res := geniex_sdk.DiarizeInferOutput{}
// 			err = client.Post(context.TODO(), "audio/diarize", map[string]any{
// 				"model": name,
// 				"audio": audioStr,
// 			}, &res)
//
// 			profileData := res.ProfileData
// 			if err != nil {
// 				return "", profileData, err
// 			}
//
// 			// Format the diarization output
// 			output := fmt.Sprint(render.GetTheme().Success.Sprintf("Detected %d speaker(s) in %.2f seconds of audio:\n\n", res.NumSpeakers, res.Duration))
// 			for i, segment := range res.Segments {
// 				output += fmt.Sprintf("%s %s\n",
// 					render.GetTheme().Info.Sprintf("[%d]", i+1),
// 					render.GetTheme().Success.Sprintf("%.2fs - %.2fs: %s", segment.StartTime, segment.EndTime, segment.SpeakerLabel))
// 			}
// 			onToken(output)
// 			return output, profileData, err
// 		},
// 	}
// 	if input != "" {
// 		processor.GetPrompt = func() (string, error) {
// 			if input == "" {
// 				return "", io.EOF
// 			}
// 			audioPath := input
// 			input = ""
// 			fmt.Print(render.GetTheme().Prompt.Sprintf("> "))
// 			fmt.Println(render.GetTheme().Normal.Sprint(audioPath))
// 			return audioPath, nil
// 		}
// 	} else {
// 		repl := common.Repl{
// 			RecordImmediate: true,
// 			Record: func() (*string, error) {
// 				t := strconv.Itoa(int(time.Now().Unix()))
// 				outputFile := filepath.Join(os.TempDir(), "geniex-cli", t+".wav")
// 				rec, err := record.NewRecorder(outputFile)
// 				if err != nil {
// 					return nil, err
// 				}
//
// 				fmt.Println(render.GetTheme().Info.Sprint("Recording is going on, press Ctrl-C to stop"))
//
// 				err = rec.Run()
// 				if err != nil {
// 					return nil, err
// 				}
// 				return &outputFile, nil
// 			},
// 		}
// 		defer repl.Close()
// 		processor.GetPrompt = repl.GetPrompt
// 	}
// 	return processor.Process()
//
// }

// func runCV(manifest types.ModelManifest, quant string) error {
// 	name := manifest.Name
// 	if quant != "" {
// 		name = name + ":" + quant
// 	}
//
// 	// warm up
// 	spin := render.NewSpinner("loading model...")
// 	spin.Start()
// 	warmUpRequest := map[string]any{
// 		"model": name,
// 	}
// 	err := client.Post(context.TODO(), "cv", warmUpRequest, nil)
// 	spin.Stop()
//
// 	if err != nil {
// 		return err
// 	}
//
// 	processor := &common.Processor{
// 		ParseFile: true,
// 		Verbose:   verbose,
// 		TestMode:  testMode,
// 		Run: func(_ string, images, _ []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
// 			start := time.Now()
//
// 			if len(images) == 0 {
// 				return "", geniex_sdk.ProfileData{}, common.ErrNoImage
// 			}
// 			if len(images) > 1 {
// 				return "", geniex_sdk.ProfileData{}, fmt.Errorf("only one image is supported")
// 			}
//
// 			// base64 encode image
// 			imageData, err := os.ReadFile(images[0])
// 			if err != nil {
// 				return "", geniex_sdk.ProfileData{}, fmt.Errorf("read image file error: %s", err.Error())
// 			}
// 			mimeType := http.DetectContentType(imageData)
// 			b64Image := base64.StdEncoding.EncodeToString(imageData)
// 			imageStr := fmt.Sprintf("data:%s;base64,%s", mimeType, b64Image)
//
// 			// send request
// 			res := struct {
// 				Results []geniex_sdk.CVResult `json:"results"`
// 			}{}
// 			err = client.Post(context.TODO(), "cv", map[string]any{
// 				"model": name,
// 				"image": imageStr,
// 			}, &res)
//
// 			duration := time.Since(start).Microseconds()
// 			profileData := geniex_sdk.ProfileData{
// 				TTFT:       duration,
// 				DecodeTime: duration,
// 			}
// 			if err != nil {
// 				return "", profileData, err
// 			}
//
// 			onToken(render.GetTheme().Success.Sprintf("✓ CV inference completed successfully"))
// 			onToken("\n")
// 			onToken(render.GetTheme().Info.Sprintf("  Found %d results\n", len(res.Results)))
//
// 			var data strings.Builder
// 			for _, cvResult := range res.Results {
// 				result := fmt.Sprintf("[%s] %s\n",
// 					render.GetTheme().Info.Sprintf("%.3f", cvResult.Confidence),
// 					render.GetTheme().Success.Sprintf("\"%s\"", cvResult.Text))
// 				onToken(result)
// 				data.WriteString(result)
// 			}
// 			return data.String(), profileData, err
// 		},
// 	}
// 	if input != "" {
// 		processor.GetPrompt = func() (string, error) {
// 			if input == "" {
// 				return "", io.EOF
// 			}
// 			imagePath := input
// 			input = ""
// 			fmt.Print(render.GetTheme().Prompt.Sprintf("> "))
// 			fmt.Println(render.GetTheme().Normal.Sprint(imagePath))
// 			return imagePath, nil
// 		}
// 	} else {
// 		repl := common.Repl{}
// 		defer repl.Close()
// 		processor.GetPrompt = repl.GetPrompt
// 	}
// 	return processor.Process()
// }

// func runImagesGenerations(manifest types.ModelManifest, quant string) error {
// 	name := manifest.Name
// 	if quant != "" {
// 		name = name + ":" + quant
// 	}
//
// 	// warm up
// 	spin := render.NewSpinner("loading model...")
// 	spin.Start()
// 	warmUpRequest := openai.ImageGenerateParams{
// 		Model: name,
// 	}
// 	_, err := client.Images.Generate(context.TODO(), warmUpRequest)
// 	spin.Stop()
//
// 	if err != nil {
// 		return err
// 	}
//
// 	processor := &common.Processor{
// 		Verbose:  verbose,
// 		TestMode: testMode,
// 		Run: func(prompt string, _, _ []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
// 			start := time.Now()
//
// 			textPrompt := strings.TrimSpace(prompt)
// 			if textPrompt == "" {
// 				return "", geniex_sdk.ProfileData{}, fmt.Errorf("prompt cannot be empty")
// 			}
//
// 			// Generate output filename if not specified
// 			outputFile := output
// 			if outputFile == "" {
// 				outputFile = fmt.Sprintf("image_gen_output_%d.png", time.Now().Unix())
// 			}
// 			if !strings.HasSuffix(strings.ToLower(outputFile), ".png") {
// 				return "", geniex_sdk.ProfileData{}, fmt.Errorf("output file must have .png extension")
// 			}
//
// 			res, err := client.Images.Generate(context.TODO(), openai.ImageGenerateParams{
// 				Model:  name,
// 				Prompt: prompt,
// 				N:      openai.Int(1),
// 			},
// 				option.WithJSONSet("task_type", taskType),
// 			)
//
// 			duration := time.Since(start).Microseconds()
// 			profileData := geniex_sdk.ProfileData{
// 				DecodeTime: duration,
// 			}
// 			if err != nil {
// 				return "", profileData, err
// 			}
//
// 			bstr := strings.SplitN(res.Data[0].B64JSON, ",", 2)
// 			if len(bstr) != 2 {
// 				return "", profileData, fmt.Errorf("invalid base64 image data")
// 			}
// 			bdata, err := base64.StdEncoding.DecodeString(bstr[1])
// 			if err != nil {
// 				return "", profileData, err
// 			}
// 			err = os.WriteFile(outputFile, bdata, 0644)
// 			if err != nil {
// 				return "", profileData, err
// 			}
//
// 			data := render.GetTheme().Success.Sprintf("✓ Image saved to: %s", outputFile)
// 			onToken(data)
// 			return data, profileData, err
//
// 		},
// 	}
// 	if len(prompt) > 0 || input != "" {
// 		processor.GetPrompt = getPromptOrInput
// 	} else {
// 		repl := common.Repl{}
// 		defer repl.Close()
// 		processor.GetPrompt = repl.GetPrompt
// 	}
// 	return processor.Process()
// }
