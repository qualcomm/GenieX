// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/cmd/geniex/common"
	"github.com/qualcomm/GenieX/cli/internal/config"
	"github.com/qualcomm/GenieX/cli/internal/record"
	"github.com/qualcomm/GenieX/cli/internal/render"
	"github.com/qualcomm/GenieX/cli/internal/store"
)

var (
	// disableStream *bool // reuse in run.go
	ngl           int32
	nctx          int32
	maxTokens     int32
	stop          []string
	stopFile      string
	enableThink   bool
	prompt        []string
	tokenFile     string
	input         string
	systemPrompt  string
	computeUnit   string
	slidingWindow bool
	specType      string
	draftModel    string
	draftTokens   int32
	draftMin      int32
	draftPMin     float32

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
		return samplerFlags
	}()
	llmFlags = func() *pflag.FlagSet {
		llmFlags := pflag.NewFlagSet("LLM/VLM Model", pflag.ExitOnError)
		llmFlags.SortFlags = false
		llmFlags.StringVarP(&computeUnit, "compute", "c", "", "compute unit to run on: cpu, gpu, npu, hybrid, or an explicit device list like HTP0,HTP1,HTP2,HTP3 (llama_cpp only) (default: npu)")
		llmFlags.Int32VarP(&ngl, "ngl", "n", -1, "number of layers to offload to gpu/npu, -1 = all (llama_cpp only)")
		llmFlags.Int32VarP(&nctx, "nctx", "", 4096, "context window size; raise to extend context (llama_cpp only)")
		llmFlags.Int32VarP(&maxTokens, "max-tokens", "", 2048, "max tokens")
		llmFlags.StringArrayVarP(&stop, "stop", "", nil, "stop sequences (llama_cpp only)")
		llmFlags.StringVarP(&stopFile, "stop-file", "", "", "file containing stop sequences (llama_cpp only)")
		llmFlags.BoolVarP(&enableThink, "think", "", true, "enable thinking mode (use --think=false to disable)")
		llmFlags.StringVarP(&systemPrompt, "system-prompt", "s", "", "system prompt to set model behavior")
		llmFlags.StringVarP(&input, "input", "i", "", "prompt txt file")
		llmFlags.StringArrayVarP(&prompt, "prompt", "p", nil, "pass prompt")
		llmFlags.StringVarP(&tokenFile, "token-file", "t", "", "path to token file (space-separated token IDs) (llama_cpp only)")
		llmFlags.BoolVarP(&slidingWindow, "sliding-window", "", false, "evict oldest context on overflow instead of erroring (qairt only)")
		llmFlags.StringVarP(&specType, "spec-type", "", "", "speculative decoding type(s), comma-separated: draft-mtp,draft-eagle3,draft-simple,ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache (llama_cpp only)")
		llmFlags.StringVarP(&draftModel, "draft-model", "", "", "draft/MTP model for draft-* spec types: catalogue name or local GGUF path (llama_cpp only)")
		llmFlags.Int32VarP(&draftTokens, "draft-tokens", "", 3, "max draft tokens per step for speculative decoding (llama_cpp only)")
		llmFlags.Int32VarP(&draftMin, "draft-min", "", 0, "min draft tokens per step (0 = llama.cpp default) (llama_cpp only)")
		llmFlags.Float32VarP(&draftPMin, "draft-p-min", "", 0.0, "min greedy draft probability (0 = llama.cpp default) (llama_cpp only)")
		return llmFlags
	}()
	vlmFlags = func() *pflag.FlagSet {
		vlmFlags := pflag.NewFlagSet("VLM Specific", pflag.ExitOnError)
		vlmFlags.SortFlags = false
		vlmFlags.StringArrayVarP(&prompt, "prompt", "p", nil, "pass prompt")
		return vlmFlags
	}()
	flagGroups = []*pflag.FlagSet{
		samplerFlags, llmFlags, vlmFlags,
	}
)

func infer() *cobra.Command {
	inferCmd := &cobra.Command{
		GroupID: "inference",
		Use:     "infer <model-name>[:<precision>]",
		Short:   "Infer with a model",
		Long:    "Run inference with a specified model. The model must be downloaded and cached locally. Append ':<precision>' to pick a specific precision; otherwise you'll be prompted to choose one when several are cached.",
	}

	inferCmd.Args = cobra.MatchAll(cobra.ExactArgs(1), cobra.OnlyValidArgs)
	for _, flags := range flagGroups {
		inferCmd.Flags().AddFlagSet(flags)
	}

	inferCmd.SetUsageFunc(flagGroupedUsage)

	inferCmd.RunE = func(cmd *cobra.Command, args []string) error {
		name, precision := geniex_sdk.SplitNamePrecision(args[0])

		if precision == "" {
			chosen, err := pickCachedPrecision(name)
			if err != nil {
				return err
			}
			precision = chosen
		}

		paths, err := ensureModelAvailable(cmd.Context(), name, precision)
		if err != nil {
			return err
		}

		if err := common.InitSDK(); err != nil {
			return err
		}

		// Host-aware default (e.g. RB3 Gen 2 → cpu) before resolution, so the
		// --verbose line and any server request see the same alias.
		var overridden bool
		if computeUnit, overridden = config.ComputeDefault(computeUnit, store.Get().ResolveChipset(true)); overridden {
			fmt.Println(render.GetTheme().Info.Sprintf("Defaulting to --compute %s for this device; pass --compute to override.", computeUnit))
		}

		effectiveType := paths.ModelType
		if effectiveType == geniex_sdk.ModelTypeVLM && specType != "" {
			fmt.Println(render.GetTheme().Warning.Sprintf(
				"Warning: --spec-type set on a VLM-classified model; running the LLM path, image / audio inputs will be ignored"))
			effectiveType = geniex_sdk.ModelTypeLLM
		}

		switch effectiveType {
		case geniex_sdk.ModelTypeLLM:
			err = inferLLM(cmd.Context(), paths)
		case geniex_sdk.ModelTypeVLM:
			err = inferVLM(paths)
		default:
			geniex_sdk.DeInit()
			return fmt.Errorf("unsupported model type: %s", paths.ModelType)
		}

		geniex_sdk.DeInit()
		if errors.Is(err, geniex_sdk.ErrCommonParamNotSupported) {
			err = fmt.Errorf("runtime %s: %w", paths.RuntimeID, err)
		}
		return err
	}
	return inferCmd
}

// ensureModelAvailable resolves a model's on-disk paths, pulling it first if
// it isn't cached. The optional quant selects a specific precision; when empty
// the SDK picks the default downloaded one.
func ensureModelAvailable(ctx context.Context, name, quant string) (*geniex_sdk.ModelPaths, error) {
	key := name
	if quant != "" {
		key = name + ":" + quant
		if err := checkPrecisionDownloaded(name, quant); err != nil {
			return nil, err
		}
	}
	paths, err := geniex_sdk.ModelGetPaths(key)
	if geniex_sdk.IsModelNotFound(err) {
		fmt.Println(render.GetTheme().Info.Sprintf("Model is not currently cached, downloading..."))
		if err := pullModel(ctx, name, quant); err != nil {
			return nil, fmt.Errorf("download model failed: %w", err)
		}
		paths, err = geniex_sdk.ModelGetPaths(key)
	}
	return paths, err
}

// checkPrecisionDownloaded reports ErrPrecisionNotFound when a cached model has
// no such precision, so the sentinel's hint (list / pull / drop the suffix)
// reaches the user instead of the SDK's generic invalid-input error. A model
// that isn't cached at all passes: the caller pulls it.
func checkPrecisionDownloaded(name, quant string) error {
	m, err := geniex_sdk.ModelGetDetailed(name)
	if err != nil {
		return nil
	}
	precisions := downloadedPrecisions(*m, true)
	// Nothing to match against (a runtime like qairt reports only N/A), so let
	// the SDK judge the key.
	if len(precisions) == 0 {
		return nil
	}
	// GGUF quant labels are matched case-insensitively by the SDK.
	if slices.ContainsFunc(precisions, func(p string) bool { return strings.EqualFold(p, quant) }) {
		return nil
	}
	return fmt.Errorf("%w: %s has %s, not %q",
		common.ErrPrecisionNotFound, name, strings.Join(precisions, ", "), quant)
}

// pickCachedPrecision asks which of a cached model's downloaded precisions to
// run, returning "" when there is nothing to disambiguate.
func pickCachedPrecision(name string) (string, error) {
	m, err := geniex_sdk.ModelGetDetailed(name)
	if err != nil {
		return "", nil
	}
	precisions := downloadedPrecisions(*m, true)
	if len(precisions) < 2 {
		return "", nil
	}

	// ModelGetPaths on the bare name resolves the SDK's own pick; choosePrecision
	// pre-selects the head, so that pick has to land there.
	def, err := geniex_sdk.ModelGetPaths(name)
	if err != nil {
		return "", err
	}
	// Size is left unset: ModelDetail.TotalSize aggregates every downloaded
	// precision, and the SDK does not expose the per-precision figure its
	// manifest already holds.
	candidates := make([]geniex_sdk.PrecisionCandidate, len(precisions))
	head := 0
	for i, p := range precisions {
		candidates[i].Precision = p
		mp, err := geniex_sdk.ModelGetPaths(name + ":" + p)
		if err != nil {
			continue
		}
		if mp.ModelPath == def.ModelPath {
			head = i
		}
	}
	candidates[0], candidates[head] = candidates[head], candidates[0]

	return choosePrecision("Select a precision from local folder", candidates)
}

// resolveDraftModel maps a --draft-model value to a GGUF path the SDK can load.
// An existing local file is used as-is; anything else is treated as a catalogue
// name and resolved (pulling if needed) like the main model.
func resolveDraftModel(ctx context.Context, draft string) (string, error) {
	if _, err := os.Stat(draft); err == nil {
		return draft, nil
	}
	name, precision := geniex_sdk.SplitNamePrecision(draft)
	paths, err := ensureModelAvailable(ctx, name, precision)
	if err != nil {
		return "", fmt.Errorf("resolve draft model %q: %w", draft, err)
	}
	return paths.ModelPath, nil
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

// modelLoadedLine summarizes the loaded session for --verbose. compute echoes
// the user alias, not the SDK's device_id (cpu/hybrid resolve to an empty one).
// Mirrors geniex_resolve_device: empty/"auto" and qairt → npu.
func modelLoadedLine(runtimeID, computeUnit string, ngl, nctx int32) string {
	computeUnit = strings.ToLower(strings.TrimSpace(computeUnit))
	if runtimeID == geniex_sdk.RuntimeQairt || computeUnit == "" || computeUnit == "auto" {
		computeUnit = geniex_sdk.ComputeUnitNPU
	}
	parts := []string{
		fmt.Sprintf("runtime=%s", runtimeID),
		fmt.Sprintf("compute=%s", computeUnit),
	}
	if runtimeID == geniex_sdk.RuntimeLlamaCpp {
		parts = append(parts,
			fmt.Sprintf("ngl=%d", ngl),
			fmt.Sprintf("nctx=%d", nctx),
		)
	}
	return "Model loaded: " + strings.Join(parts, " ")
}

// resolveModelParams resolves --compute / --ngl / --nctx into the
// (device_id, ngl, nctx) triple the SDK expects. --ngl (-1 = all) and
// --nctx are llama_cpp-only; qairt rejects any non-zero value, so both
// are zeroed for it (the SDK also forces ngl to 0). Compute-unit alias
// mapping is delegated to geniex_resolve_device (sdk/src/device.cpp).
func resolveModelParams(runtimeID, modelName string) (deviceID string, resolvedNgl, resolvedNctx int32, err error) {
	resolvedNgl, resolvedNctx = ngl, nctx
	if runtimeID != geniex_sdk.RuntimeLlamaCpp {
		resolvedNctx = 0
	}

	resolved, err := geniex_sdk.ResolveDevice(geniex_sdk.ResolveDeviceInput{
		RuntimeID:   runtimeID,
		ModelName:   modelName,
		ComputeUnit: computeUnit,
		NglDefault:  resolvedNgl,
	})
	if err != nil {
		return
	}
	deviceID = resolved.DeviceID
	resolvedNgl = resolved.Ngl
	if resolved.Warning != "" {
		fmt.Println(render.GetTheme().Warning.Sprintf("Warning: %s", resolved.Warning))
	}
	return
}

func inferLLM(ctx context.Context, paths *geniex_sdk.ModelPaths) error {
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
	}
	stopSequences, err := loadStopSequences()
	if err != nil {
		return err
	}

	deviceID, nglResolved, nctxResolved, err := resolveModelParams(paths.RuntimeID, paths.ModelName)
	if err != nil {
		return err
	}

	// Speculative decoding is llama_cpp-only; ignore the spec flags for other runtimes.
	resolvedSpecType := specType
	specDraftModel := draftModel
	if paths.RuntimeID != geniex_sdk.RuntimeLlamaCpp {
		if resolvedSpecType != "" || specDraftModel != "" {
			fmt.Println(render.GetTheme().Warning.Sprintf(
				"Warning: speculative decoding is only supported by llama_cpp; ignoring for runtime %s", paths.RuntimeID))
		}
		resolvedSpecType = ""
		specDraftModel = ""
	} else if specDraftModel != "" {
		// A draft model may be a local GGUF path or a catalogue name (resolved
		// and pulled like the main model).
		specDraftModel, err = resolveDraftModel(ctx, specDraftModel)
		if err != nil {
			return err
		}
	}

	spin := render.NewSpinner("loading model...")
	spin.Start()

	p, err := geniex_sdk.NewLLM(geniex_sdk.LlmCreateInput{
		ModelPath: paths.ModelPath,
		RuntimeID: paths.RuntimeID,
		DeviceID:  deviceID,
		Config: geniex_sdk.ModelConfig{
			NCtx:           nctxResolved,
			NGpuLayers:     nglResolved,
			SpecType:       resolvedSpecType,
			SpecDraftModel: specDraftModel,
			SpecNMax:       draftTokens,
			SpecNMin:       draftMin,
			SpecPMin:       draftPMin,
		},
	})
	spin.Stop()

	if err != nil {
		return err
	}
	defer p.Destroy()

	if verbose {
		fmt.Println(render.GetTheme().Info.Sprint(modelLoadedLine(paths.RuntimeID, computeUnit, nglResolved, nctxResolved)))
	}

	var history []geniex_sdk.LlmChatMessage
	if systemPrompt != "" {
		history = append(history, geniex_sdk.LlmChatMessage{Role: geniex_sdk.LlmRoleSystem, Content: systemPrompt})
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
		Verbose:  verbose,
		TestMode: testMode,
		Reset: func() error {
			err := p.Reset()
			if err == nil {
				history = nil
			}
			return err
		},
		Run: func(prompt string, _, _ []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
			var res *geniex_sdk.LlmGenerateOutput
			var err error

			if len(tokenIDs) > 0 {
				// When using token IDs, skip chat template and use IDs directly
				res, err = p.Generate(geniex_sdk.LlmGenerateInput{
					InputIDs: tokenIDs,
					OnToken:  onToken,
					Config: &geniex_sdk.GenerationConfig{
						MaxTokens:     maxTokens,
						SamplerConfig: samplerConfig,
						SlidingWindow: slidingWindow,
					},
				})
				if err != nil {
					// The SDK keeps whatever was generated before the failure; surface it.
					return res.FullText, res.ProfileData, err
				}
				// Clear tokenIDs after use so subsequent calls use normal mode
				tokenIDs = nil
			} else {
				// Normal text prompt mode with chat template
				history = append(history, geniex_sdk.LlmChatMessage{Role: geniex_sdk.LlmRoleUser, Content: prompt})

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
						SlidingWindow: slidingWindow,
					},
				})

				if err != nil {
					// The SDK keeps whatever was generated before the failure; surface it.
					return res.FullText, res.ProfileData, err
				}

				history = append(history, geniex_sdk.LlmChatMessage{Role: geniex_sdk.LlmRoleAssistant, Content: res.FullText})
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
		repl := common.Repl{}
		repl.Reset = processor.Reset
		defer repl.Close()
		processor.GetPrompt = repl.GetPrompt
	}

	return processor.Process()
}

func inferVLM(paths *geniex_sdk.ModelPaths) error {
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
	}
	stopSequences, err := loadStopSequences()
	if err != nil {
		return err
	}

	deviceID, nglResolved, nctxResolved, err := resolveModelParams(paths.RuntimeID, paths.ModelName)
	if err != nil {
		return err
	}

	spin := render.NewSpinner("loading model...")
	spin.Start()
	p, err := geniex_sdk.NewVLM(geniex_sdk.VlmCreateInput{
		ModelPath:  paths.ModelPath,
		MmprojPath: paths.MmprojPath,
		RuntimeID:  paths.RuntimeID,
		DeviceID:   deviceID,
		Config: geniex_sdk.ModelConfig{
			NCtx:       nctxResolved,
			NGpuLayers: nglResolved,
		},
	})
	spin.Stop()

	if err != nil {
		slog.Error("failed to create VLM", "error", err)
		return err
	}
	defer p.Destroy()

	if verbose {
		fmt.Println(render.GetTheme().Info.Sprint(modelLoadedLine(paths.RuntimeID, computeUnit, nglResolved, nctxResolved)))
	}

	caps, _ := p.Capabilities()
	slog.Debug("VLM capabilities", "vision", caps.SupportsVision, "audio", caps.SupportsAudio)
	if caps.SupportsAudio {
		checkAudioDependency()
	}

	// Warn once per unsupported modality; feeding an audio path to a
	// vision-only model corrupts the run (the pipeline has no audio encoder).
	warnedAudio, warnedImage := false, false

	var history []geniex_sdk.VlmChatMessage
	if systemPrompt != "" {
		history = append(history, geniex_sdk.VlmChatMessage{Role: geniex_sdk.VlmRoleSystem, Contents: []geniex_sdk.VlmContent{{Type: geniex_sdk.VlmContentTypeText, Text: systemPrompt}}})
	}

	processor := &common.Processor{
		ParseFile: true,
		Verbose:   verbose,
		TestMode:  testMode,
		Reset: func() error {
			err := p.Reset()
			if err == nil {
				history = nil
			}
			return err
		},
		Run: func(prompt string, images, audios []string, onToken func(string) bool) (string, geniex_sdk.ProfileData, error) {
			if len(audios) > 0 && !caps.SupportsAudio {
				if !warnedAudio {
					fmt.Println(render.GetTheme().Warning.Sprint("Warning: this model does not support audio; ignoring audio input."))
					warnedAudio = true
				}
				audios = nil
			}
			if len(images) > 0 && !caps.SupportsVision {
				if !warnedImage {
					fmt.Println(render.GetTheme().Warning.Sprint("Warning: this model does not support images; ignoring image input."))
					warnedImage = true
				}
				images = nil
			}

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
					MaxTokens:     maxTokens,
					Stop:          stopSequences,
					SamplerConfig: samplerConfig,
					ImagePaths:    images,
					AudioPaths:    audios,
				},
			})
			if err != nil {
				// The SDK keeps whatever was generated before the failure; surface it.
				return res.FullText, res.ProfileData, err
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
		repl := common.Repl{}
		repl.Reset = processor.Reset
		if caps.SupportsAudio {
			repl.Record = func() (*string, error) {
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
			}
		}
		defer repl.Close()
		processor.GetPrompt = repl.GetPrompt
	}

	return processor.Process()
}
