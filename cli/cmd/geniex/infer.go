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
	"io"
	"log/slog"
	"os"
	"path/filepath"
	// "reflect"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/charmbracelet/huh"
	"github.com/dustin/go-humanize"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	geniex_sdk "github.com/qcom-it-nexa-ai/geniex/bindings/go"
	"github.com/qcom-it-nexa-ai/geniex/cli/cmd/geniex/common"
	// "github.com/qcom-it-nexa-ai/geniex/cli/cmd/geniex/logic"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub/aihub"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/record"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/render"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/store"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/types"
)

var (
	// disableStream *bool // reuse in run.go
	ngl            int32
	nctx           int32
	maxTokens      int32
	stop           []string
	stopFile       string
	imageMaxLength int32
	enableThink    bool
	hideThink      bool
	prompt         []string
	tokenFile      string
	input        string
	systemPrompt string

	// sampler config
	temperature       float32
	topP              float32
	topK              int32
	minP              float32
	repetitionPenalty float32
	presencePenalty   float32
	frequencyPenalty  float32
	seed              int32
	grammarPath       string
	grammarString     string
	enableJson        bool
)

// NOTE: flagset use same flag name will be ignored, but usage is different, so we keep them in different flagset
var (
	samplerFlags = func() *pflag.FlagSet {
		samplerFlags := pflag.NewFlagSet("LLM/VLM Sampler", pflag.ExitOnError)
		samplerFlags.SortFlags = false
		samplerFlags.Float32VarP(&temperature, "temperature", "", 0.0, "sampling temperature")
		samplerFlags.Float32VarP(&topP, "top-p", "", 0.0, "top-p sampling")
		samplerFlags.Int32VarP(&topK, "top-k", "", 0, "top-k sampling")
		samplerFlags.Float32VarP(&minP, "min-p", "", 0.0, "min-p sampling")
		samplerFlags.Float32VarP(&repetitionPenalty, "repetition-penalty", "", 1.0, "repetition penalty")
		samplerFlags.Float32VarP(&presencePenalty, "presence-penalty", "", 0.0, "presence penalty")
		samplerFlags.Float32VarP(&frequencyPenalty, "frequency-penalty", "", 0.0, "frequency penalty")
		samplerFlags.Int32VarP(&seed, "seed", "", 0, "random seed")
		samplerFlags.StringVarP(&grammarPath, "grammar-path", "", "", "path to grammar file")
		samplerFlags.StringVarP(&grammarString, "grammar-string", "", "", "grammar in string format")
		samplerFlags.BoolVarP(&enableJson, "enable-json", "", false, "enable json output")
		return samplerFlags
	}()
	llmFlags = func() *pflag.FlagSet {
		llmFlags := pflag.NewFlagSet("LLM/VLM Model", pflag.ExitOnError)
		llmFlags.SortFlags = false
		llmFlags.Int32VarP(&ngl, "ngl", "n", 999, "num of layers pass to gpu")
		llmFlags.Int32VarP(&nctx, "nctx", "", 4096, "context window size")
		llmFlags.Int32VarP(&maxTokens, "max-tokens", "", 2048, "max tokens")
		llmFlags.StringArrayVarP(&stop, "stop", "", nil, "stop sequences")
		llmFlags.StringVarP(&stopFile, "stop-file", "", "", "file containing stop sequences")
		llmFlags.BoolVarP(&enableThink, "enable-think", "", true, "enable thinking mode")
		llmFlags.BoolVarP(&hideThink, "hide-think", "", false, "hide thinking output")
		llmFlags.StringVarP(&systemPrompt, "system-prompt", "s", "", "system prompt to set model behavior")
		llmFlags.StringVarP(&input, "input", "i", "", "prompt txt file")
		llmFlags.StringArrayVarP(&prompt, "prompt", "p", nil, "pass prompt")
		llmFlags.StringVarP(&tokenFile, "token-file", "t", "", "path to token file (space-separated token IDs)")
		return llmFlags
	}()
	vlmFlags = func() *pflag.FlagSet {
		vlmFlags := pflag.NewFlagSet("VLM Specific", pflag.ExitOnError)
		vlmFlags.SortFlags = false
		vlmFlags.StringArrayVarP(&prompt, "prompt", "p", nil, "pass prompt")
		vlmFlags.Int32VarP(&imageMaxLength, "image-max-length", "", 512, "max image length")
		return vlmFlags
	}()
	flagGroups = []*pflag.FlagSet{
		samplerFlags, llmFlags, vlmFlags,
	}
)

func infer() *cobra.Command {
	inferCmd := &cobra.Command{
		GroupID: "inference",
		Use:     "infer <model-name>",
		Short:   "Infer with a model",
		Long:    "Run inference with a specified model. The model must be downloaded and cached locally.",
	}

	inferCmd.Args = cobra.MatchAll(cobra.ExactArgs(1), cobra.OnlyValidArgs)
	for _, flags := range flagGroups {
		inferCmd.Flags().AddFlagSet(flags)
	}

	inferCmd.SetUsageFunc(func(c *cobra.Command) error {
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

	inferCmd.Run = func(cmd *cobra.Command, args []string) {
		s := store.Get()

		name, quant := normalizeModelName(args[0])
		manifest, err := ensureModelAvailable(s, name, quant)
		if err != nil {
			fmt.Println(render.GetTheme().Error.Sprintf("Error: %s", err))
			os.Exit(1)
		}

		if quant != "" {
			if fileinfo, exist := manifest.ModelFile[quant]; !exist {
				fmt.Println(render.GetTheme().Error.Sprintf("Error: quant %s not found", quant))
				os.Exit(1)
			} else if !fileinfo.Downloaded {
				fmt.Println(render.GetTheme().Error.Sprintf("Error: quant %s not downloaded", quant))
				os.Exit(1)
			}
		} else {
			sq, err := selectQuant(manifest)
			if err != nil {
				fmt.Println(render.GetTheme().Error.Sprintf("Error: %s", err))
				os.Exit(1)
			}
			quant = sq
		}

		geniex_sdk.Init()
		defer geniex_sdk.DeInit()

		switch manifest.ModelType {
		case types.ModelTypeLLM:
			err = inferLLM(manifest, quant)
		case types.ModelTypeVLM:
			checkDependency()
			err = inferVLM(manifest, quant)
		default:
			panic("not support model type")
		}

		switch err {
		case nil:
			os.Exit(0)
		case geniex_sdk.ErrCommonNotSupport:
			fmt.Println(render.GetTheme().Error.Sprint(`
⚠️ Oops. This model type is not supported yet.

👉 Try these:
- Check back later for updates.
- See help in our discord or slack.`))
		case geniex_sdk.ErrCommonModelLoad:
			fmt.Println(render.GetTheme().Error.Sprint(`
⚠️ Oops. Model failed to load.

👉 Try these:
- Redownload the model.
- Verify your system meets the model's requirements.
- See help in our discord or slack.`))
		case geniex_sdk.ErrCommonPluginLoad:
			fmt.Println(render.GetTheme().Error.Sprint(`
⚠️ Oops. Plugin failed to load.

👉 Try these:
- Ensure all plugin dependencies are correct.
- See help in our discord or slack.`))
		case geniex_sdk.ErrCommonPluginInvalid:
			fmt.Println(render.GetTheme().Error.Sprint(`
⚠️ Oops. Plugin is invalid.

👉 Try these:
- This model may not be compatible with your system. Try another model.
- See help in our discord or slack.`))
		case geniex_sdk.ErrLlmTokenizationContextLength:
			fmt.Println(render.GetTheme().Info.Sprintf("Context length exceeded, please start a new conversation"))
		default:
			fmt.Println(render.GetTheme().Error.Sprintf("Error: %s", err))
		}
		os.Exit(1)
	}
	return inferCmd
}

func ensureModelAvailable(s *store.Store, name string, quant string) (*types.ModelManifest, error) {
	manifest, err := s.GetManifest(name)
	if errors.Is(err, os.ErrNotExist) {
		fmt.Println(render.GetTheme().Info.Sprintf("model not found, start download"))
		// Try AI Hub path first for allowlisted orgs (e.g. "qualcomm/<repo>").
		if org, repo, ok := splitOrgRepo(name); ok && slices.Contains(aiHubOrgs, org) {
			aiErr := tryPullAIHubModel(context.Background(), name, repo, false)
			if aiErr == nil {
				manifest, err = s.GetManifest(name)
				return manifest, err
			}
			if !errors.Is(aiErr, aihub.ErrModelNotFound) {
				return nil, fmt.Errorf("download model failed")
			}
		}
		err = pullModel(name, quant)
		if err != nil {
			return nil, fmt.Errorf("download model failed")
		}
		manifest, err = s.GetManifest(name)
	}
	return manifest, err
}

func selectQuant(manifest *types.ModelManifest) (string, error) {
	var options []huh.Option[string]
	for k, v := range manifest.ModelFile {
		if v.Downloaded {
			options = append(options, huh.NewOption(fmt.Sprintf("%-10s [%7s]", k, humanize.IBytes(uint64(v.Size))), k))
		}
	}
	if len(options) == 0 {
		return "", fmt.Errorf("no quant found")
	}
	if len(options) == 1 {
		return options[0].Value, nil
	}
	var quant string
	if err := huh.NewSelect[string]().Title("Select a quant from local folder").Options(options...).Value(&quant).Run(); err != nil {
		return "", err
	}
	return quant, nil
}

func getPromptOrInput() (string, error) {
	if input != "" {
		content, err := os.ReadFile(input)
		// print prompt
		prompt := strings.TrimSpace(string(content))
		firstLine := true
		for line := range strings.SplitSeq(prompt, "\n") {
			if firstLine {
				fmt.Print(render.GetTheme().Prompt.Sprintf("> "))
				fmt.Println(render.GetTheme().Normal.Sprint(line))
				firstLine = false
			} else {
				fmt.Println(render.GetTheme().Normal.Sprintf(". %s", line))
			}

		}
		input = ""
		return prompt, err
	}
	if len(prompt) > 0 {
		p := prompt[0]
		fmt.Print(render.GetTheme().Prompt.Sprintf("> "))
		fmt.Println(render.GetTheme().Normal.Sprint(p))
		prompt = prompt[1:]
		return p, nil
	}
	return "", io.EOF
}

func loadStopSequences() ([]string, error) {
	var stopSequences []string
	if stopFile != "" {
		content, err := os.ReadFile(stopFile)
		if err != nil {
			return nil, err
		}
		for line := range strings.SplitSeq(string(content), "\n") {
			if line != "" {
				stopSequences = append(stopSequences, line)
			}
		}
	}
	stopSequences = append(stopSequences, stop...)
	return stopSequences, nil
}

func inferLLM(manifest *types.ModelManifest, quant string) error {
	samplerConfig := &geniex_sdk.SamplerConfig{
		Temperature:       temperature,
		TopP:              topP,
		TopK:              topK,
		MinP:              minP,
		RepetitionPenalty: repetitionPenalty,
		PresencePenalty:   presencePenalty,
		FrequencyPenalty:  frequencyPenalty,
		Seed:              seed,
		GrammarPath:       grammarPath,
		GrammarString:     grammarString,
		EnableJson:        enableJson,
	}
	stopSequences, err := loadStopSequences()
	if err != nil {
		return err
	}

	s := store.Get()
	modelfile := s.ModelfilePath(manifest.Name, manifest.ModelFile[quant].Name)
	spin := render.NewSpinner("loading model...")
	spin.Start()

	p, err := geniex_sdk.NewLLM(geniex_sdk.LlmCreateInput{
		ModelName: manifest.ModelName,
		ModelPath: modelfile,
		PluginID:  manifest.PluginId,
		DeviceID:  manifest.DeviceId,
		Config: geniex_sdk.ModelConfig{
			NCtx:       nctx,
			NGpuLayers: ngl,
		},
	})
	spin.Stop()

	if err != nil {
		return err
	}
	defer p.Destroy()

	var history []geniex_sdk.LlmChatMessage
	if systemPrompt != "" {
		history = append(history, geniex_sdk.LlmChatMessage{Role: geniex_sdk.LLMRoleSystem, Content: systemPrompt})
	}

	// Check if using token ID input mode
	var tokenIDs []int32
	if tokenFile != "" {
		content, err := os.ReadFile(tokenFile)
		if err != nil {
			return fmt.Errorf("failed to read token file: %w", err)
		}
		for field := range strings.FieldsSeq(string(content)) {
			tokenID, err := strconv.ParseInt(field, 10, 32)
			if err != nil {
				return fmt.Errorf("invalid token ID: %s", field)
			}
			tokenIDs = append(tokenIDs, int32(tokenID))
		}
		fmt.Println(render.GetTheme().Info.Sprintf("Using token IDs from file: %s (%d tokens)", tokenFile, len(tokenIDs)))
	}

	processor := &common.Processor{
		HideThink: hideThink,
		Verbose:   verbose,
		TestMode:  testMode,
		Run: func(prompt string, _, _ []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
			var res geniex_sdk.LlmGenerateOutput
			var err error

			if len(tokenIDs) > 0 {
				// When using token IDs, skip chat template and use IDs directly
				res, err = p.Generate(geniex_sdk.LlmGenerateInput{
					InputIDs: tokenIDs,
					OnToken:  onToken,
					Config: &geniex_sdk.GenerationConfig{
						MaxTokens:     maxTokens,
						SamplerConfig: samplerConfig,
					},
				})
				if err != nil {
					return "", geniex_sdk.ProfileData{}, err
				}
				// Clear tokenIDs after use so subsequent calls use normal mode
				tokenIDs = nil
			} else {
				// Normal text prompt mode with chat template
				history = append(history, geniex_sdk.LlmChatMessage{Role: geniex_sdk.LLMRoleUser, Content: prompt})

				templateOutput, err := p.ApplyChatTemplate(geniex_sdk.LlmApplyChatTemplateInput{
					Messages:            history,
					EnableThink:         enableThink,
					AddGenerationPrompt: true,
				})
				if err != nil {
					return "", geniex_sdk.ProfileData{}, err
				}

				res, err = p.Generate(geniex_sdk.LlmGenerateInput{
					PromptUTF8: templateOutput.FormattedText,
					OnToken:    onToken,
					Config: &geniex_sdk.GenerationConfig{
						MaxTokens:     maxTokens,
						Stop:          stopSequences,
						SamplerConfig: samplerConfig,
					},
				})

				if err != nil {
					return "", geniex_sdk.ProfileData{}, err
				}

				history = append(history, geniex_sdk.LlmChatMessage{Role: geniex_sdk.LLMRoleAssistant, Content: res.FullText})
			}

			return res.FullText, res.ProfileData, nil
		},
	}

	if len(tokenIDs) > 0 {
		// Token ID mode: return empty prompt once, then EOF to exit after first round
		firstCall := true
		processor.GetPrompt = func() (string, error) {
			if firstCall {
				firstCall = false
				return "", nil // Trigger first round with empty prompt (token IDs will be used)
			}
			return "", io.EOF // Exit after first round
		}
	} else if len(prompt) > 0 || input != "" {
		processor.GetPrompt = getPromptOrInput
	} else {
		repl := common.Repl{
			Reset: func() error {
				err := p.Reset()
				if err == nil {
					history = nil
				}
				return err
			},

			SaveKVCache: func(path string) error {
				_, err := p.SaveKVCache(geniex_sdk.LlmSaveKVCacheInput{Path: path})
				return err
			},

			LoadKVCache: func(path string) error {
				_, err := p.LoadKVCache(geniex_sdk.LlmLoadKVCacheInput{Path: path})
				if err == nil {
					history = nil
				}
				return err
			},
		}
		defer repl.Close()
		processor.GetPrompt = repl.GetPrompt
	}

	return processor.Process()
}

func inferVLM(manifest *types.ModelManifest, quant string) error {
	samplerConfig := &geniex_sdk.SamplerConfig{
		Temperature:       temperature,
		TopP:              topP,
		TopK:              topK,
		MinP:              minP,
		RepetitionPenalty: repetitionPenalty,
		PresencePenalty:   presencePenalty,
		FrequencyPenalty:  frequencyPenalty,
		Seed:              seed,
		GrammarPath:       grammarPath,
		GrammarString:     grammarString,
		EnableJson:        enableJson,
	}
	stopSequences, err := loadStopSequences()
	if err != nil {
		return err
	}

	s := store.Get()
	modelfile := s.ModelfilePath(manifest.Name, manifest.ModelFile[quant].Name)
	var mmprojfile string
	if manifest.MMProjFile.Name != "" {
		mmprojfile = s.ModelfilePath(manifest.Name, manifest.MMProjFile.Name)
	}
	var tokenizerfile string
	if manifest.TokenizerFile.Name != "" {
		tokenizerfile = s.ModelfilePath(manifest.Name, manifest.TokenizerFile.Name)
	}
	spin := render.NewSpinner("loading model...")
	spin.Start()
	p, err := geniex_sdk.NewVLM(geniex_sdk.VlmCreateInput{
		ModelName:     manifest.ModelName,
		ModelPath:     modelfile,
		MmprojPath:    mmprojfile,
		TokenizerPath: tokenizerfile,
		PluginID:      manifest.PluginId,
		DeviceID:      manifest.DeviceId,
		Config: geniex_sdk.ModelConfig{
			NCtx:       nctx,
			NGpuLayers: ngl,
		},
	})
	spin.Stop()

	if err != nil {
		slog.Error("failed to create VLM", "error", err)
		return err
	}
	defer p.Destroy()

	var history []geniex_sdk.VlmChatMessage
	if systemPrompt != "" {
		history = append(history, geniex_sdk.VlmChatMessage{Role: geniex_sdk.VlmRoleSystem, Contents: []geniex_sdk.VlmContent{{Type: geniex_sdk.VlmContentTypeText, Text: systemPrompt}}})
	}

	processor := &common.Processor{
		ParseFile: true,
		HideThink: hideThink,
		Verbose:   verbose,
		TestMode:  testMode,
		Run: func(prompt string, images, audios []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
			msg := geniex_sdk.VlmChatMessage{Role: geniex_sdk.VlmRoleUser}
			msg.Contents = append(msg.Contents, geniex_sdk.VlmContent{Type: geniex_sdk.VlmContentTypeText, Text: prompt})
			for _, image := range images {
				msg.Contents = append(msg.Contents, geniex_sdk.VlmContent{Type: geniex_sdk.VlmContentTypeImage, Text: image})
			}
			for _, audio := range audios {
				msg.Contents = append(msg.Contents, geniex_sdk.VlmContent{Type: geniex_sdk.VlmContentTypeAudio, Text: audio})
			}

			history = append(history, msg)

			tmplOut, err := p.ApplyChatTemplate(geniex_sdk.VlmApplyChatTemplateInput{
				Messages:    history,
				EnableThink: enableThink,
			})
			if err != nil {
				return "", geniex_sdk.ProfileData{}, err
			}

			res, err := p.Generate(geniex_sdk.VlmGenerateInput{
				PromptUTF8: tmplOut.FormattedText,
				OnToken:    onToken,
				Config: &geniex_sdk.GenerationConfig{
					MaxTokens:      maxTokens,
					Stop:           stopSequences,
					SamplerConfig:  samplerConfig,
					ImagePaths:     images,
					ImageMaxLength: imageMaxLength,
					AudioPaths:     audios,
				},
			})
			if err != nil {
				return "", geniex_sdk.ProfileData{}, err
			}

			history = append(history, geniex_sdk.VlmChatMessage{
				Role: geniex_sdk.VlmRoleAssistant,
				Contents: []geniex_sdk.VlmContent{
					{Type: geniex_sdk.VlmContentTypeText, Text: res.FullText},
				},
			})

			return res.FullText, res.ProfileData, nil
		},
	}

	if len(prompt) > 0 || input != "" {
		processor.GetPrompt = getPromptOrInput
	} else {
		repl := common.Repl{
			Reset: func() error {
				err := p.Reset()
				if err == nil {
					history = nil
				}
				return err
			},
			Record: func() (*string, error) {
				t := strconv.Itoa(int(time.Now().Unix()))
				outputFile := filepath.Join(os.TempDir(), "geniex-cli", t+".wav")
				rec, err := record.NewRecorder(outputFile)
				if err != nil {
					return nil, err
				}

				fmt.Println(render.GetTheme().Info.Sprint("Recording is going on, press Ctrl-C to stop"))

				err = rec.Run()
				if err != nil {
					return nil, err
				}
				outfile := rec.GetOutputFile()
				return &outfile, nil
			},
		}
		defer repl.Close()
		processor.GetPrompt = repl.GetPrompt
	}

	return processor.Process()
}

