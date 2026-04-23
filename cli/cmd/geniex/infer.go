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
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	// "reflect"
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
	// taskType       string
	// query          string
	// document       []string
	input          string
	// output         string
	// voice          string
	// listVoice      bool
	// speechSpeed    float64
	systemPrompt   string
	// language       string
	// listLanguage   bool

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
	// embedderFlags = func() *pflag.FlagSet {
	// 	embedderFlags := pflag.NewFlagSet("Embedder", pflag.ExitOnError)
	// 	embedderFlags.SortFlags = false
	// 	embedderFlags.StringVarP(&input, "input", "i", "", "input text file or image file")
	// 	embedderFlags.StringArrayVarP(&prompt, "prompt", "p", nil, "pass prompt or image path (e.g., -p 'text' or -p '/path/to/image.jpg')")
	// 	embedderFlags.StringVarP(&taskType, "task-type", "", "default", "default|search_query|search_document")
	// 	return embedderFlags
	// }()
	// rerankerFlags = func() *pflag.FlagSet {
	// 	rerankerFlags := pflag.NewFlagSet("Reranker", pflag.ExitOnError)
	// 	rerankerFlags.SortFlags = false
	// 	rerankerFlags.StringVarP(&query, "query", "q", "", "query")
	// 	rerankerFlags.StringArrayVarP(&document, "document", "d", nil, "documents")
	// 	return rerankerFlags
	// }()
	// ttsFlags = func() *pflag.FlagSet {
	// 	ttsFlags := pflag.NewFlagSet("TTS", pflag.ExitOnError)
	// 	ttsFlags.SortFlags = false
	// 	ttsFlags.StringVarP(&input, "input", "i", "", "prompt txt file")
	// 	ttsFlags.StringArrayVarP(&prompt, "prompt", "p", nil, "pass prompt")
	// 	ttsFlags.StringVarP(&voice, "voice", "", "", "voice identifier")
	// 	ttsFlags.BoolVarP(&listVoice, "list-voice", "", false, "list available voices")
	// 	ttsFlags.Float64VarP(&speechSpeed, "speech-speed", "", 1.0, "speech speed (1.0 = normal)")
	// 	ttsFlags.StringVarP(&output, "output", "o", "", "output audio file")
	// 	return ttsFlags
	// }()
	// asrFlags = func() *pflag.FlagSet {
	// 	asrFlags := pflag.NewFlagSet("ASR", pflag.ExitOnError)
	// 	asrFlags.SortFlags = false
	// 	asrFlags.StringVarP(&input, "input", "i", "", "input audio file")
	// 	// asrFlags.StringVarP(&language, "language", "", "", "language code (e.g., en, zh, ja)")   // TODO: Language support not implemented yet
	// 	// asrFlags.BoolVarP(&listLanguage, "list-language", "", false, "list available languages") // TODO: Language support not implemented yet
	// 	return asrFlags
	// }()
	// diarizeFlags = func() *pflag.FlagSet {
	// 	diarizeFlags := pflag.NewFlagSet("Diarize", pflag.ExitOnError)
	// 	diarizeFlags.SortFlags = false
	// 	diarizeFlags.StringVarP(&input, "input", "i", "", "input audio file")
	// 	return diarizeFlags
	// }()
	// cvFlags = func() *pflag.FlagSet {
	// 	cvFlags := pflag.NewFlagSet("CV", pflag.ExitOnError)
	// 	cvFlags.SortFlags = false
	// 	cvFlags.StringVarP(&input, "input", "i", "", "input image file")
	// 	return cvFlags
	// }()
	// imageGenFlags = func() *pflag.FlagSet {
	// 	imageGenFlags := pflag.NewFlagSet("ImageGen", pflag.ExitOnError)
	// 	imageGenFlags.SortFlags = false
	// 	imageGenFlags.StringVarP(&input, "input", "i", "", "prompt txt file")
	// 	imageGenFlags.StringArrayVarP(&prompt, "prompt", "p", nil, "pass prompt")
	// 	imageGenFlags.StringVarP(&output, "output", "o", "", "output image file")
	// 	return imageGenFlags
	// }()
	flagGroups = []*pflag.FlagSet{
		samplerFlags, llmFlags, vlmFlags,
		// embedderFlags, rerankerFlags, ttsFlags, asrFlags, diarizeFlags, cvFlags, imageGenFlags,
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
		// case types.ModelTypeEmbedder:
		// 	err = inferEmbedder(manifest, quant)
		// case types.ModelTypeReranker:
		// 	err = inferReranker(manifest, quant)
		// case types.ModelTypeTTS:
		// 	err = inferTTS(manifest, quant)
		// case types.ModelTypeASR:
		// 	checkDependency()
		// 	err = inferASR(manifest, quant)
		// case types.ModelTypeDiarize:
		// 	err = inferDiarize(manifest, quant)
		// case types.ModelTypeCV:
		// 	err = inferCV(manifest, quant)
		// case types.ModelTypeImageGen:
		// 	// ImageGen model is a directory, not a file
		// 	err = inferImageGen(manifest, quant)
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

// func inferEmbedder(manifest *types.ModelManifest, quant string) error {
// 	s := store.Get()
// 	modelfile := s.ModelfilePath(manifest.Name, manifest.ModelFile[quant].Name)
// 	spin := render.NewSpinner("loading embedding model...")
// 	spin.Start()
//
// 	embedderInput := geniex_sdk.EmbedderCreateInput{
// 		ModelName: manifest.ModelName,
// 		ModelPath: modelfile,
// 		PluginID:  manifest.PluginId,
// 		DeviceID:  manifest.DeviceId,
// 	}
//
// 	p, err := geniex_sdk.NewEmbedder(embedderInput)
// 	spin.Stop()
//
// 	if err != nil {
// 		slog.Error("failed to create embedder", "error", err)
// 		return err
// 	}
// 	defer p.Destroy()
//
// 	processor := &common.Processor{
// 		ParseFile: true,
// 		Verbose:   verbose,
// 		TestMode:  testMode,
// 		Run: func(prompt string, images, _ []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
// 			embedInput := geniex_sdk.EmbedderEmbedInput{
// 				TaskType: taskType,
// 				Config:   &geniex_sdk.EmbeddingConfig{},
// 			}
//
// 			// Validate: image paths and text cannot be passed at the same time
// 			trimmedPrompt := strings.TrimSpace(prompt)
// 			if len(images) > 0 && trimmedPrompt != "" {
// 				return "", geniex_sdk.ProfileData{}, fmt.Errorf("cannot pass both image paths and text at the same time")
// 			}
//
// 			// Handle text or image inputs
// 			if len(images) > 0 {
// 				embedInput.ImagePaths = images
// 			} else if trimmedPrompt != "" {
// 				embedInput.Texts = []string{trimmedPrompt}
// 			} else {
// 				return "", geniex_sdk.ProfileData{}, fmt.Errorf("must provide either text or image path")
// 			}
//
// 			result, err := p.Embed(embedInput)
// 			if err != nil || len(result.Embeddings) == 0 {
// 				return "", result.ProfileData, err
// 			}
//
// 			emb := result.Embeddings[0]
// 			n := len(emb)
// 			info := render.GetTheme().Info.Sprintf("Embedding")
// 			var out string
// 			if n > 6 {
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
// 			return data, result.ProfileData, nil
// 		},
// 	}
//
// 	if len(prompt) > 0 || input != "" {
// 		processor.GetPrompt = getPromptOrInput
// 	} else {
// 		repl := common.Repl{}
// 		defer repl.Close()
// 		processor.GetPrompt = repl.GetPrompt
// 	}
//
// 	return processor.Process()
// }

// func inferReranker(manifest *types.ModelManifest, quant string) error {
// 	s := store.Get()
// 	modelfile := s.ModelfilePath(manifest.Name, manifest.ModelFile[quant].Name)
// 	spin := render.NewSpinner("loading reranker model...")
// 	spin.Start()
//
// 	rerankerInput := geniex_sdk.RerankerCreateInput{
// 		ModelName: manifest.ModelName,
// 		ModelPath: modelfile,
// 		PluginID:  manifest.PluginId,
// 		DeviceID:  manifest.DeviceId,
// 	}
//
// 	p, err := geniex_sdk.NewReranker(rerankerInput)
// 	spin.Stop()
//
// 	if err != nil {
// 		slog.Error("failed to create reranker", "error", err)
// 		return err
// 	}
// 	defer p.Destroy()
//
// 	const SEP = "\\n"
// 	processor := &common.Processor{
// 		Verbose:  verbose,
// 		TestMode: testMode,
// 		Run: func(prompt string, _, _ []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
// 			parsedPrompt := strings.Split(prompt, SEP)
// 			if len(parsedPrompt) < 2 {
// 				return "", geniex_sdk.ProfileData{}, fmt.Errorf("parsed prompt failed, query and document are required for reranking")
// 			}
// 			query := parsedPrompt[0]
// 			document := parsedPrompt[1:]
// 			fmt.Println(render.GetTheme().Info.Sprintf("Query: %s", query))
// 			fmt.Println(render.GetTheme().Info.Sprintf("Processing %d documents", len(document)))
//
// 			rerankInput := geniex_sdk.RerankerRerankInput{
// 				Query:     query,
// 				Documents: document,
// 				Config: &geniex_sdk.RerankConfig{
// 					BatchSize:       int32(len(document)),
// 					Normalize:       true,
// 					NormalizeMethod: "softmax",
// 				},
// 			}
//
// 			result, err := p.Rerank(rerankInput)
// 			if err != nil {
// 				return "", result.ProfileData, err
// 			}
//
// 			fmt.Println(render.GetTheme().Success.Sprintf("✓ Reranking completed successfully. Generated %d scores", len(result.Scores)))
//
// 			// Display results
// 			var data strings.Builder
// 			for i, doc := range document {
// 				if i < len(result.Scores) {
// 					line := fmt.Sprintf("\n%s [%d]: %s\n", render.GetTheme().Info.Sprintf("Document"), i+1, doc)
// 					onToken(line)
// 					data.WriteString(line)
// 					line = fmt.Sprintf("%s: %.6f\n", render.GetTheme().Info.Sprintf("Score"), result.Scores[i])
// 					onToken(line)
// 					data.WriteString(line)
// 				}
// 			}
// 			return data.String(), result.ProfileData, nil
// 		},
// 	}
//
// 	if query != "" || len(document) > 0 {
// 		if query == "" || len(document) == 0 {
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
//
// 	return processor.Process()
// }

// func inferTTS(manifest *types.ModelManifest, quant string) error {
// 	s := store.Get()
// 	modelfile := s.ModelfilePath(manifest.Name, manifest.ModelFile[quant].Name)
// 	spin := render.NewSpinner("loading TTS model...")
// 	spin.Start()
//
// 	ttsInput := geniex_sdk.TtsCreateInput{
// 		ModelName: manifest.ModelName,
// 		ModelPath: modelfile,
// 		PluginID:  manifest.PluginId,
// 		DeviceID:  manifest.DeviceId,
// 	}
//
// 	p, err := geniex_sdk.NewTTS(ttsInput)
// 	spin.Stop()
//
// 	if err != nil {
// 		slog.Error("failed to create TTS", "error", err)
// 		return err
// 	}
// 	defer p.Destroy()
//
// 	if listVoice {
// 		voices, err := p.ListAvailableVoices()
// 		if err != nil {
// 			return fmt.Errorf("Failed to list voices: %s", err)
// 		}
// 		fmt.Println(render.GetTheme().Success.Sprintf("Available voices: %v", voices.VoiceIDs))
// 		return nil
// 	}
//
// 	processor := &common.Processor{
// 		Verbose:  verbose,
// 		TestMode: testMode,
// 		Run: func(prompt string, _, _ []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
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
// 			// Create TTS config
// 			ttsConfig := &geniex_sdk.TTSConfig{
// 				Voice:      "af_heart",
// 				Speed:      float32(speechSpeed),
// 				SampleRate: 24000,
// 				Seed:       42,
// 			}
//
// 			if voice != "" {
// 				ttsConfig.Voice = voice
// 			}
//
// 			// Synthesize speech
// 			synthesizeInput := geniex_sdk.TtsSynthesizeInput{
// 				TextUTF8:   textToSynthesize,
// 				Config:     ttsConfig,
// 				OutputPath: outputFile,
// 			}
//
// 			result, err := p.Synthesize(synthesizeInput)
// 			if err != nil {
// 				return "", geniex_sdk.ProfileData{}, err
// 			}
//
// 			data := render.GetTheme().Success.Sprintf("✓ Audio saved: %s", result.Result.AudioPath)
// 			onToken(data)
// 			return data, result.ProfileData, nil
// 		},
// 	}
//
// 	if len(prompt) > 0 || input != "" {
// 		processor.GetPrompt = getPromptOrInput
// 	} else {
// 		repl := common.Repl{}
// 		defer repl.Close()
// 		processor.GetPrompt = repl.GetPrompt
// 	}
//
// 	return processor.Process()
// }

// func inferASR(manifest *types.ModelManifest, quant string) error {
// 	s := store.Get()
// 	modelfile := s.ModelfilePath(manifest.Name, manifest.ModelFile[quant].Name)
// 	spin := render.NewSpinner("loading ASR model...")
// 	spin.Start()
//
// 	asrInput := geniex_sdk.AsrCreateInput{
// 		ModelName: manifest.ModelName,
// 		ModelPath: modelfile,
// 		PluginID:  manifest.PluginId,
// 		DeviceID:  manifest.DeviceId,
// 		Language:  language,
// 	}
// 	p, err := geniex_sdk.NewASR(asrInput)
// 	spin.Stop()
//
// 	if err != nil {
// 		slog.Error("failed to create ASR", "error", err)
// 		return err
// 	}
// 	defer p.Destroy()
//
// 	if listLanguage {
// 		lans, err := p.ListSupportedLanguages()
// 		if err != nil {
// 			return fmt.Errorf("Failed to list available languages: %s", err)
// 		}
// 		fmt.Println(render.GetTheme().Success.Sprintf("Available languages: %v", lans.LanguageCodes))
// 		return nil
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
// 			asrConfig := &geniex_sdk.ASRConfig{
// 				Timestamps: "segment",
// 				BeamSize:   5,
// 				Stream:     false,
// 			}
//
// 			transcribeInput := geniex_sdk.AsrTranscribeInput{
// 				AudioPath: audios[0],
// 				Language:  language,
// 				Config:    asrConfig,
// 			}
//
// 			fmt.Println(render.GetTheme().Info.Sprintf("Transcribing audio file: %s", audios[0]))
//
// 			result, err := p.Transcribe(transcribeInput)
// 			if err != nil {
// 				return "", geniex_sdk.ProfileData{}, err
// 			}
// 			onToken(result.Result.Transcript)
// 			return result.Result.Transcript, result.ProfileData, nil
// 		},
// 	}
//
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
// 				streamConfig := geniex_sdk.ASRStreamConfig{
// 					ChunkDuration:   4.0,
// 					OverlapDuration: 3.5,
// 					SampleRate:      16000,
// 					MaxQueueSize:    10,
// 					BufferSize:      1024,
// 					Timestamps:      "segment",
// 					BeamSize:        4,
// 				}
// 				_, err := p.StreamBegin(geniex_sdk.AsrStreamBeginInput{
// 					StreamConfig: &streamConfig,
// 					Language:     "en",
// 					OnTranscription: func(text string, _ any) {
// 						tWidth := common.GetTerminalWidth()
// 						if len(text) > tWidth {
// 							text = "..." + text[len(text)-tWidth+3:]
// 						}
// 						text += strings.Repeat(" ", tWidth-len(text))
//
// 						fmt.Print("\r")
// 						fmt.Print(render.GetTheme().ModelOutput.Sprint(text))
// 					},
// 					UserData: nil,
// 				})
// 				slog.Debug("ASR StreamBegin", "error", err)
// 				if err != nil && !errors.Is(err, geniex_sdk.ErrCommonNotSupport) {
// 					return nil, err
// 				}
// 				defer p.StreamStop(geniex_sdk.AsrStreamStopInput{})
//
// 				// streaming not supported, fallback to file input
// 				if err != nil {
// 					t := strconv.Itoa(int(time.Now().Unix()))
// 					outputFile := filepath.Join(os.TempDir(), "geniex-cli", t+".wav")
// 					rec, err := record.NewRecorder(outputFile)
// 					if err != nil {
// 						return nil, err
// 					}
//
// 					fmt.Println(render.GetTheme().Info.Sprint("Recording is going on, press Ctrl-C to stop"))
//
// 					err = rec.Run()
// 					if err != nil {
// 						return nil, err
// 					}
// 					return &outputFile, nil
//
// 				} else {
// 					rec, err := record.NewStreamRecorder()
// 					if err != nil {
// 						return nil, err
// 					}
// 					fmt.Println(render.GetTheme().Info.Sprint("Streaming ASR recording started, press Ctrl-C to stop"))
// 					fmt.Println()
//
// 					if err := rec.Start(); err != nil {
// 						return nil, err
// 					}
// 					defer rec.Stop()
//
// 					buffer := make([]float32, 512)
// 					for {
// 						n, err := rec.ReadFloat32(buffer)
// 						if err == io.EOF {
// 							break
// 						}
//
// 						if err := p.StreamPushAudio(geniex_sdk.AsrStreamPushAudioInput{
// 							AudioData: buffer[:n],
// 						}); err != nil {
// 							fmt.Println(render.GetTheme().Error.Sprintf("error pushing audio data: %s", err))
// 							fmt.Println()
// 							return nil, err
// 						}
// 					}
//
// 					fmt.Println()
// 					render.GetTheme().Reset()
// 					fmt.Println()
// 				}
//
// 				return nil, nil
// 			},
// 		}
// 		defer repl.Close()
// 		processor.GetPrompt = repl.GetPrompt
// 	}
//
// 	return processor.Process()
// }

// func inferDiarize(manifest *types.ModelManifest, quant string) error {
// 	s := store.Get()
// 	modelfile := s.ModelfilePath(manifest.Name, manifest.ModelFile[quant].Name)
// 	spin := render.NewSpinner("loading diarization model...")
// 	spin.Start()
//
// 	diarizeInput := geniex_sdk.DiarizeCreateInput{
// 		ModelName: manifest.ModelName,
// 		ModelPath: modelfile,
// 		PluginID:  manifest.PluginId,
// 		DeviceID:  manifest.DeviceId,
// 	}
// 	p, err := geniex_sdk.NewDiarize(diarizeInput)
// 	spin.Stop()
//
// 	if err != nil {
// 		slog.Error("failed to create diarization model", "error", err)
// 		return err
// 	}
// 	defer p.Destroy()
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
// 				return "", geniex_sdk.ProfileData{}, fmt.Errorf("diarization only supports a single audio file, got %d files", len(audios))
// 			}
//
// 			diarizeConfig := &geniex_sdk.DiarizeConfig{
// 				MinSpeakers: 0, // auto-detect
// 				MaxSpeakers: 0, // no limit
// 			}
//
// 			inferInput := geniex_sdk.DiarizeInferInput{
// 				AudioPath: audios[0],
// 				Config:    diarizeConfig,
// 			}
//
// 			fmt.Println(render.GetTheme().Info.Sprintf("Analyzing audio file: %s", audios[0]))
//
// 			result, err := p.Infer(inferInput)
// 			if err != nil {
// 				return "", geniex_sdk.ProfileData{}, err
// 			}
//
// 			// Format the diarization output
// 			output := fmt.Sprint(render.GetTheme().Success.Sprintf("Detected %d speaker(s) in %.2f seconds of audio:\n\n", result.NumSpeakers, result.Duration))
// 			for i, segment := range result.Segments {
// 				output += fmt.Sprintf("%s %s\n",
// 					render.GetTheme().Info.Sprintf("[%d]", i+1),
// 					render.GetTheme().Success.Sprintf("%.2fs - %.2fs: %s", segment.StartTime, segment.EndTime, segment.SpeakerLabel))
// 			}
// 			onToken(output)
// 			return output, result.ProfileData, nil
// 		},
// 	}
//
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
// 				// Diarization doesn't support streaming, use file-based recording
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
//
// 	return processor.Process()
// }

// func inferCV(manifest *types.ModelManifest, quant string) error {
// 	s := store.Get()
// 	modelfile := s.ModelfilePath(manifest.Name, manifest.ModelFile[quant].Name)
// 	spin := render.NewSpinner("loading CV model...")
// 	spin.Start()
//
// 	cvInput := geniex_sdk.CVCreateInput{
// 		ModelName: manifest.ModelName,
// 		Config: geniex_sdk.CVModelConfig{
// 			Capabilities: geniex_sdk.CVCapabilityOCR,
// 			DetModelPath: modelfile,
// 			RecModelPath: modelfile,
// 		},
// 		PluginID: manifest.PluginId,
// 		DeviceID: manifest.DeviceId,
// 	}
//
// 	p, err := geniex_sdk.NewCV(cvInput)
// 	spin.Stop()
//
// 	if err != nil {
// 		slog.Error("failed to create CV", "error", err)
// 		return err
// 	}
// 	defer p.Destroy()
//
// 	processor := &common.Processor{
// 		ParseFile: true,
// 		Verbose:   verbose,
// 		TestMode:  testMode,
// 		Run: func(_ string, images, _ []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
// 			if len(images) == 0 {
// 				return "", geniex_sdk.ProfileData{}, common.ErrNoImage
// 			}
// 			if len(images) > 1 {
// 				return "", geniex_sdk.ProfileData{}, fmt.Errorf("only one image file is supported")
// 			}
//
// 			inferInput := geniex_sdk.CVInferInput{
// 				InputImagePath: images[0],
// 			}
//
// 			result, err := p.Infer(inferInput)
// 			slog.Debug("CV Infer result", "result", result, "error", err)
// 			if err != nil {
// 				return "", geniex_sdk.ProfileData{}, err
// 			}
//
// 			onToken(render.GetTheme().Success.Sprintf("✓ CV inference completed successfully"))
// 			onToken("\n")
// 			onToken(render.GetTheme().Info.Sprintf("  Found %d results, ", len(result.Results)))
//
// 			data := ""
//
// 			if len(result.Results) == 0 {
// 				onToken(render.GetTheme().Info.Sprintf("no output, skip generate output image\n"))
// 				return data, geniex_sdk.ProfileData{}, nil
// 			}
//
// 			if len(result.Results) == 1 && reflect.ValueOf(result.Results[0].BBox).IsZero() {
// 				// rmbg
// 				onToken(render.GetTheme().Info.Sprintf("Mask output detected\n"))
//
// 			} else {
// 				// bbox
// 				onToken(render.GetTheme().Info.Sprintf("BBox output detected\n"))
// 				for _, cvResult := range result.Results {
// 					result := fmt.Sprintf("[%s] %s\n",
// 						render.GetTheme().Info.Sprintf("%.3f", cvResult.Confidence),
// 						render.GetTheme().Success.Sprintf("\"%s\"", cvResult.Text))
// 					onToken(result)
// 					data += result
// 				}
// 			}
//
// 			// Only create output image if there are meaningful bboxes or masks to draw
// 			for _, cvResult := range result.Results {
// 				if (cvResult.BBox.Width > 0 && cvResult.BBox.Height > 0) || len(cvResult.Mask) > 0 {
// 					outputPath, err := logic.CVPostProcess(images[0], result.Results)
// 					if err != nil {
// 						return data, geniex_sdk.ProfileData{}, err
// 					}
//
// 					onToken(render.GetTheme().Success.Sprintf("  Result drawn and saved to: %s\n", outputPath))
// 					break
// 				}
// 			}
//
// 			return data, geniex_sdk.ProfileData{}, nil
// 		},
// 	}
//
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
//
// 	return processor.Process()
// }

// func inferImageGen(manifest *types.ModelManifest, _ string) error {
// 	s := store.Get()
// 	modeldir := s.ModelfilePath(manifest.Name, "")
//
// 	spin := render.NewSpinner("loading ImageGen model...")
// 	spin.Start()
// 	p, err := geniex_sdk.NewImageGen(geniex_sdk.ImageGenCreateInput{
// 		ModelName: manifest.ModelName,
// 		ModelPath: modeldir,
// 		PluginID:  manifest.PluginId,
// 		DeviceID:  manifest.DeviceId,
// 	})
// 	spin.Stop()
//
// 	if err != nil {
// 		slog.Error("failed to create ImageGen", "error", err)
// 		return err
// 	}
// 	defer p.Destroy()
//
// 	processor := &common.Processor{
// 		Verbose:  verbose,
// 		TestMode: testMode,
// 		Run: func(prompt string, _, _ []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
// 			textPrompt := strings.TrimSpace(prompt)
// 			if textPrompt == "" {
// 				return "", geniex_sdk.ProfileData{}, fmt.Errorf("prompt cannot be empty")
// 			}
//
// 			// Generate output filename if not specified
// 			outputFile := output
// 			if outputFile == "" {
// 				outputFile = fmt.Sprintf("imagegen_output_%d.png", time.Now().Unix())
// 			}
//
// 			result, err := p.Txt2Img(geniex_sdk.ImageGenTxt2ImgInput{
// 				PromptUTF8: textPrompt,
// 				Config: &geniex_sdk.ImageGenerationConfig{
// 					Prompts:         []string{textPrompt},
// 					NegativePrompts: []string{"blurry, low quality, distorted"},
// 					Height:          512,
// 					Width:           512,
// 					SamplerConfig: geniex_sdk.ImageSamplerConfig{
// 						Method:        "ddim",
// 						Steps:         20,
// 						GuidanceScale: 7.5,
// 						Eta:           0.0,
// 						Seed:          42,
// 					},
// 					SchedulerConfig: geniex_sdk.SchedulerConfig{
// 						Type:              "ddim",
// 						NumTrainTimesteps: 1000,
// 						StepsOffset:       1,
// 						BetaStart:         0.00085,
// 						BetaEnd:           0.012,
// 						BetaSchedule:      "scaled_linear",
// 						PredictionType:    "epsilon",
// 						TimestepType:      "discrete",
// 						TimestepSpacing:   "leading",
// 						InterpolationType: "linear",
// 						ConfigPath:        "",
// 					},
// 					Strength: 1.0,
// 				},
// 				OutputPath: outputFile,
// 			})
// 			if err != nil {
// 				return "", geniex_sdk.ProfileData{}, err
// 			}
//
// 			data := render.GetTheme().Success.Sprintf("✓ Image saved to: %s", result.OutputImagePath)
// 			onToken(data)
// 			return data, geniex_sdk.ProfileData{}, nil
// 		},
// 	}
//
// 	if len(prompt) > 0 || input != "" {
// 		processor.GetPrompt = getPromptOrInput
// 	} else {
// 		repl := common.Repl{}
// 		defer repl.Close()
// 		processor.GetPrompt = repl.GetPrompt
// 	}
//
// 	return processor.Process()
// }
