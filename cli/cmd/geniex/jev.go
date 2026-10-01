// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/charmbracelet/huh"
	"github.com/spf13/cobra"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	bindingjev "github.com/qualcomm/GenieX/bindings/go/jev"
	"github.com/qualcomm/GenieX/cli/cmd/geniex/common"
	"github.com/qualcomm/GenieX/cli/internal/config"
	"github.com/qualcomm/GenieX/cli/internal/jev"
	"github.com/qualcomm/GenieX/cli/internal/jev/browser"
	"github.com/qualcomm/GenieX/cli/internal/render"
	"github.com/qualcomm/GenieX/cli/internal/store"
)

type jevOptions struct {
	labels        []string
	options       []string
	instruction   string
	context       string
	contextFile   string
	images        []string
	confidence    bool
	minSelections int
	maxSelections int
	benchmark    int
	message      string
	threads      int32
	threadsBatch int32
	batch        int32
	ubatch       int32

	task        string
	url         string
	browser     string
	browserPath string
	attach      string
	profile     string
	traceDir    string
	headless    bool
	dryRun      bool
	maxSteps    int
	auto        string
}

func jevCmd() *cobra.Command {
	options := jevOptions{}
	command := &cobra.Command{
		GroupID: "inference",
		Use:     "jev <model-name>[:<precision>] <classify|choose|triage|browser>",
		Short:   "Run a structured decision or browser task",
		Long: `Run a text, visual, or browser JEV task with one local model.

classify, choose, and triage return validated JSON candidates and never execute side effects. browser runs the JEV browser agent, whose approval policy remains in force for consequential actions.`,
		Args: func(_ *cobra.Command, args []string) error {
			if len(args) != 2 {
				return fmt.Errorf("requires a model name and a JEV mode: classify, choose, triage, or browser")
			}
			switch args[1] {
			case "classify", "choose", "triage", "browser":
				return nil
			default:
				return fmt.Errorf("unknown JEV mode %q (expected classify, choose, triage, or browser)", args[1])
			}
		},
		RunE: func(cmd *cobra.Command, args []string) error {
			if options.benchmark < 0 {
				return fmt.Errorf("--benchmark must be at least 0")
			}
			if err := validateJEVRuntimeOptions(options); err != nil {
				return err
			}
			if options.benchmark > 0 && args[1] == "browser" {
				return fmt.Errorf("--benchmark supports classify, choose, and triage; browser timing includes separate observation and execution costs")
			}
			switch args[1] {
			case "classify":
				return runJEVClassify(cmd.Context(), args[0], options)
			case "choose":
				return runJEVChoose(cmd.Context(), args[0], options)
			case "triage":
				return runJEVTriage(cmd.Context(), args[0], options)
			case "browser":
				return runJEV(cmd.Context(), args[0], options)
			}
			return nil
		},
	}
	command.Flags().AddFlagSet(jevModelFlags())
	command.Flags().StringArrayVar(&options.labels, "label", nil, "allowed classification label ID (repeatable)")
	command.Flags().StringArrayVar(&options.options, "option", nil, "allowed choice as id=text (repeatable)")
	command.Flags().StringVar(&options.instruction, "instruction", "", "task instruction")
	command.Flags().StringVar(&options.context, "context", "", "untrusted task context")
	command.Flags().StringVar(&options.contextFile, "context-file", "", "file containing untrusted task context")
	command.Flags().StringArrayVar(&options.images, "image", nil, "image path for a visual decision (repeatable)")
	command.Flags().BoolVar(&options.confidence, "confidence", false, "request advisory model-reported confidence")
	command.Flags().IntVar(&options.minSelections, "min-selections", 0, "minimum selected options (choose mode)")
	command.Flags().IntVar(&options.maxSelections, "max-selections", 0, "maximum selected options (choose mode)")
	command.Flags().IntVar(&options.benchmark, "benchmark", 0, "measure JEV inference with this many repetitions after one warmup (classify, choose, or triage)")
	command.Flags().StringVar(&options.message, "message", "", "untrusted customer message (triage mode)")
	command.Flags().Int32Var(&options.threads, "threads", 0, "model decode thread count; 0 uses the SDK default")
	command.Flags().Int32Var(&options.threadsBatch, "threads-batch", 0, "model prefill thread count; 0 uses the SDK default")
	command.Flags().Int32Var(&options.batch, "batch", 0, "model batch size; 0 uses the SDK default")
	command.Flags().Int32Var(&options.ubatch, "ubatch", 0, "model micro-batch size; 0 uses the SDK default")
	command.Flags().StringVarP(&options.task, "task", "T", "", "browser task to complete (browser mode)")
	command.Flags().StringVar(&options.url, "url", "", "initial http(s) URL (browser mode)")
	command.Flags().StringVar(&options.browser, "browser", "auto", "browser to launch: auto, chrome, edge (browser mode)")
	command.Flags().StringVar(&options.browserPath, "browser-path", "", "path to Chrome or Edge executable (browser mode)")
	command.Flags().StringVar(&options.attach, "attach", "", "existing Chrome DevTools Protocol endpoint (browser mode)")
	command.Flags().StringVar(&options.profile, "profile", "", "browser profile directory (browser mode)")
	command.Flags().StringVar(&options.traceDir, "trace-dir", "", "write browser traces to this directory (browser mode)")
	command.Flags().BoolVar(&options.headless, "headless", false, "launch the browser headlessly (browser mode)")
	command.Flags().BoolVar(&options.dryRun, "dry-run", false, "do not execute browser actions (browser mode)")
	command.Flags().IntVar(&options.maxSteps, "max-steps", 25, "maximum browser actions (browser mode)")
	command.Flags().StringVar(&options.auto, "auto", string(jev.ApprovalRead), "approval policy: read, none, all (browser mode)")
	command.Flags().SortFlags = false
	return command
}

func jevContext(options jevOptions) (string, error) {
	if options.context != "" && options.contextFile != "" {
		return "", fmt.Errorf("--context and --context-file cannot be used together")
	}
	if options.contextFile != "" {
		content, err := os.ReadFile(options.contextFile)
		if err != nil {
			return "", fmt.Errorf("read --context-file: %w", err)
		}
		return string(content), nil
	}
	if strings.TrimSpace(options.context) == "" {
		return "", fmt.Errorf("one of --context or --context-file is required")
	}
	return options.context, nil
}

func runJEVClassify(ctx context.Context, model string, options jevOptions) error {
	contextText, err := jevContext(options)
	if err != nil {
		return err
	}
	if err := validateJEVLabels(options.labels, options.instruction); err != nil {
		return err
	}
	return withJEVModel(ctx, model, options, func(paths *geniex_sdk.ModelPaths, llm *geniex_sdk.LLM, vlm *geniex_sdk.VLM) error {
		spec := bindingjev.ClassificationSpec{Labels: options.labels, Instructions: options.instruction, Context: contextText, ImagePaths: options.images, MaxTokens: maxTokens, RequestConfidence: options.confidence}
		if llm != nil {
			decider := bindingjev.LLMDecider{LLM: llm, RuntimeID: paths.RuntimeID}
			result, timing, err := runJEVBenchmark(options.benchmark, llm.Reset, func() (any, bindingjev.DecisionTiming, error) {
				return decider.ClassifyWithTiming(ctx, spec)
			})
			if err != nil {
				return err
			}
			printJEVBenchmark(options.benchmark, options, timing)
			return printJEVJSON(result)
		}
		decider := bindingjev.VLMDecider{VLM: vlm, RuntimeID: paths.RuntimeID}
		result, timing, err := runJEVBenchmark(options.benchmark, vlm.Reset, func() (any, bindingjev.DecisionTiming, error) {
			return decider.ClassifyWithTiming(ctx, spec)
		})
		if err != nil {
			return err
		}
		printJEVBenchmark(options.benchmark, options, timing)
		return printJEVJSON(result)
	})
}

func runJEVChoose(ctx context.Context, model string, options jevOptions) error {
	contextText, err := jevContext(options)
	if err != nil {
		return err
	}
	parsed, err := parseJEVOptions(options.options)
	if err != nil {
		return err
	}
	if err := validateJEVOptions(parsed, options.instruction, options.minSelections, options.maxSelections); err != nil {
		return err
	}
	return withJEVModel(ctx, model, options, func(paths *geniex_sdk.ModelPaths, llm *geniex_sdk.LLM, vlm *geniex_sdk.VLM) error {
		spec := bindingjev.MultipleChoiceSpec{Options: parsed, Instructions: options.instruction, Context: contextText, ImagePaths: options.images, MaxTokens: maxTokens, MinSelections: options.minSelections, MaxSelections: options.maxSelections, RequestConfidence: options.confidence}
		if llm != nil {
			decider := bindingjev.LLMDecider{LLM: llm, RuntimeID: paths.RuntimeID}
			result, timing, err := runJEVBenchmark(options.benchmark, llm.Reset, func() (any, bindingjev.DecisionTiming, error) {
				return decider.ChooseWithTiming(ctx, spec)
			})
			if err != nil {
				return err
			}
			printJEVBenchmark(options.benchmark, options, timing)
			return printJEVJSON(result)
		}
		decider := bindingjev.VLMDecider{VLM: vlm, RuntimeID: paths.RuntimeID}
		result, timing, err := runJEVBenchmark(options.benchmark, vlm.Reset, func() (any, bindingjev.DecisionTiming, error) {
			return decider.ChooseWithTiming(ctx, spec)
		})
		if err != nil {
			return err
		}
		printJEVBenchmark(options.benchmark, options, timing)
		return printJEVJSON(result)
	})
}

func runJEVTriage(ctx context.Context, model string, options jevOptions) error {
	if strings.TrimSpace(options.message) == "" {
		return fmt.Errorf("--message is required")
	}
	return withJEVModel(ctx, model, options, func(paths *geniex_sdk.ModelPaths, llm *geniex_sdk.LLM, vlm *geniex_sdk.VLM) error {
		spec := bindingjev.TriageSpec{Message: options.message, ImagePaths: options.images, MaxTokens: maxTokens}
		if llm != nil {
			decider := bindingjev.LLMDecider{LLM: llm, RuntimeID: paths.RuntimeID}
			result, timing, err := runJEVBenchmark(options.benchmark, llm.Reset, func() (any, bindingjev.DecisionTiming, error) {
				return decider.TriageWithTiming(ctx, spec)
			})
			if err != nil {
				return err
			}
			printJEVBenchmark(options.benchmark, options, timing)
			return printJEVJSON(result)
		}
		decider := bindingjev.VLMDecider{VLM: vlm, RuntimeID: paths.RuntimeID}
		result, timing, err := runJEVBenchmark(options.benchmark, vlm.Reset, func() (any, bindingjev.DecisionTiming, error) {
			return decider.TriageWithTiming(ctx, spec)
		})
		if err != nil {
			return err
		}
		printJEVBenchmark(options.benchmark, options, timing)
		return printJEVJSON(result)
	})
}

func parseJEVOptions(values []string) ([]bindingjev.Option, error) {
	options := make([]bindingjev.Option, 0, len(values))
	for _, value := range values {
		id, text, ok := strings.Cut(value, "=")
		if !ok || strings.TrimSpace(id) == "" || strings.TrimSpace(text) == "" {
			return nil, fmt.Errorf("--option must use id=text")
		}
		options = append(options, bindingjev.Option{ID: id, Text: text})
	}
	return options, nil
}

func validateJEVRuntimeOptions(options jevOptions) error {
	for _, option := range []struct {
		name  string
		value int32
	}{
		{"--threads", options.threads},
		{"--threads-batch", options.threadsBatch},
		{"--batch", options.batch},
		{"--ubatch", options.ubatch},
	} {
		if option.value < 0 {
			return fmt.Errorf("%s must be at least 0", option.name)
		}
	}
	return nil
}

func jevModelConfig(nctx, ngl int32, options jevOptions) geniex_sdk.ModelConfig {
	return geniex_sdk.ModelConfig{
		NCtx:          nctx,
		NThreads:      options.threads,
		NThreadsBatch: options.threadsBatch,
		NBatch:        options.batch,
		NUbatch:       options.ubatch,
		NGpuLayers:    ngl,
	}
}

func validateJEVLabels(labels []string, instruction string) error {
	if strings.TrimSpace(instruction) == "" {
		return fmt.Errorf("--instruction is required")
	}
	if len(labels) == 0 {
		return fmt.Errorf("at least one --label is required")
	}
	seen := make(map[string]struct{}, len(labels))
	for _, label := range labels {
		if strings.TrimSpace(label) == "" {
			return fmt.Errorf("--label cannot be empty")
		}
		if _, duplicate := seen[label]; duplicate {
			return fmt.Errorf("duplicate --label %q", label)
		}
		seen[label] = struct{}{}
	}
	return nil
}

func validateJEVOptions(options []bindingjev.Option, instruction string, min, max int) error {
	if strings.TrimSpace(instruction) == "" {
		return fmt.Errorf("--instruction is required")
	}
	if len(options) == 0 {
		return fmt.Errorf("at least one --option is required")
	}
	seen := make(map[string]struct{}, len(options))
	for _, option := range options {
		if _, duplicate := seen[option.ID]; duplicate {
			return fmt.Errorf("duplicate --option ID %q", option.ID)
		}
		seen[option.ID] = struct{}{}
	}
	if min == 0 && max == 0 {
		return nil
	}
	if min < 0 || max < min || max > len(options) {
		return fmt.Errorf("invalid --min-selections/--max-selections bounds")
	}
	return nil
}

func printJEVJSON(value any) error {
	encoded, err := json.Marshal(value)
	if err != nil {
		return err
	}
	fmt.Println(string(encoded))
	return nil
}

type jevTimingSummary struct {
	Samples          int     `json:"samples"`
	PromptHash       string  `json:"prompt_sha256"`
	PromptVersion    string  `json:"prompt_version"`
	SDKTotalMS       float64 `json:"sdk_total_ms"`
	EndToEndMS       float64 `json:"end_to_end_ms"`
	HostOverheadMS   float64 `json:"host_overhead_ms"`
	TTFTMS           float64 `json:"ttft_ms"`
	PromptTokens     int64   `json:"prompt_tokens"`
	GeneratedTokens  int64   `json:"generated_tokens"`
	PrefillTokensSec float64 `json:"prefill_tokens_per_second"`
	DecodeTokensSec  float64 `json:"decode_tokens_per_second"`
	EndToEndStdDevMS float64 `json:"end_to_end_stddev_ms"`
	Threads          int32   `json:"threads"`
	ThreadsBatch     int32   `json:"threads_batch"`
	Batch            int32   `json:"batch"`
	UBatch           int32   `json:"ubatch"`
}

func runJEVBenchmark(repetitions int, reset func() error, run func() (any, bindingjev.DecisionTiming, error)) (any, []bindingjev.DecisionTiming, error) {
	if repetitions == 0 {
		result, timing, err := run()
		return result, []bindingjev.DecisionTiming{timing}, err
	}
	if _, _, err := run(); err != nil {
		return nil, nil, fmt.Errorf("JEV benchmark warmup: %w", err)
	}
	if err := reset(); err != nil {
		return nil, nil, fmt.Errorf("reset model after JEV benchmark warmup: %w", err)
	}

	timings := make([]bindingjev.DecisionTiming, 0, repetitions)
	var result any
	for attempt := 0; attempt < repetitions; attempt++ {
		var timing bindingjev.DecisionTiming
		var err error
		result, timing, err = run()
		if err != nil {
			return nil, nil, fmt.Errorf("JEV benchmark repetition %d: %w", attempt+1, err)
		}
		timings = append(timings, timing)
		if attempt+1 < repetitions {
			if err := reset(); err != nil {
				return nil, nil, fmt.Errorf("reset model after JEV benchmark repetition %d: %w", attempt+1, err)
			}
		}
	}
	return result, timings, nil
}

func printJEVBenchmark(repetitions int, options jevOptions, timings []bindingjev.DecisionTiming) {
	if repetitions == 0 || len(timings) == 0 {
		return
	}
	endToEnd := make([]float64, len(timings))
	sdkTotal := make([]float64, len(timings))
	promptHash := timings[0].PromptHash
	promptVersion := timings[0].PromptVersion
	for _, timing := range timings[1:] {
		if timing.PromptHash != promptHash || timing.PromptVersion != promptVersion {
			fmt.Fprintln(os.Stderr, "JEV timing provenance differs between repetitions")
			promptHash, promptVersion = "", ""
			break
		}
	}
	ttft := make([]float64, len(timings))
	prefill := make([]float64, len(timings))
	decode := make([]float64, len(timings))
	var promptTokens, generatedTokens float64
	for index, timing := range timings {
		profile := timing.ProfileData
		endToEnd[index] = float64(timing.TotalTime) / float64(time.Millisecond)
		sdkTotal[index] = float64(profile.TotalTimeUs()) / 1000
		ttft[index] = float64(profile.TTFT) / 1000
		prefill[index] = profile.PrefillSpeed
		decode[index] = profile.DecodingSpeed
		promptTokens += float64(profile.PromptTokens)
		generatedTokens += float64(profile.GeneratedTokens)
	}
	medianEndToEnd := medianJEVTiming(endToEnd)
	medianSDKTotal := medianJEVTiming(sdkTotal)
	mean := 0.0
	for _, value := range endToEnd {
		mean += value
	}
	mean /= float64(len(endToEnd))
	variance := 0.0
	for _, value := range endToEnd {
		variance += (value - mean) * (value - mean)
	}
	summary := jevTimingSummary{
		Samples:          len(timings),
		PromptHash:       promptHash,
		PromptVersion:    promptVersion,
		SDKTotalMS:       medianSDKTotal,
		EndToEndMS:       medianEndToEnd,
		HostOverheadMS:   medianEndToEnd - medianSDKTotal,
		TTFTMS:           medianJEVTiming(ttft),
		PromptTokens:     int64(promptTokens / float64(len(timings))),
		GeneratedTokens:  int64(generatedTokens / float64(len(timings))),
		PrefillTokensSec: medianJEVTiming(prefill),
		DecodeTokensSec:  medianJEVTiming(decode),
		EndToEndStdDevMS: math.Sqrt(variance / float64(len(endToEnd))),
		Threads:          options.threads,
		ThreadsBatch:     options.threadsBatch,
		Batch:            options.batch,
		UBatch:           options.ubatch,
	}
	encoded, err := json.Marshal(summary)
	if err != nil {
		fmt.Fprintf(os.Stderr, "JEV timing encoding failed: %v\n", err)
		return
	}
	fmt.Fprintf(os.Stderr, "JEV timing (one warmup, %d measured): %s\n", repetitions, encoded)
}

func medianJEVTiming(values []float64) float64 {
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	middle := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return (sorted[middle-1] + sorted[middle]) / 2
	}
	return sorted[middle]
}

func withJEVModel(ctx context.Context, model string, options jevOptions, run func(*geniex_sdk.ModelPaths, *geniex_sdk.LLM, *geniex_sdk.VLM) error) error {
	name, precision := geniex_sdk.SplitNamePrecision(model)
	paths, err := ensureModelAvailable(ctx, name, precision)
	if err != nil {
		return err
	}
	if len(options.images) != 0 && paths.ModelType != geniex_sdk.ModelTypeVLM {
		return fmt.Errorf("--image requires a VLM; %s is %s", paths.ModelName, paths.ModelType)
	}
	if err := common.InitSDK(); err != nil {
		return err
	}
	defer geniex_sdk.DeInit()
	if overridden := applyJEVComputeDefault(); overridden {
		fmt.Println(render.GetTheme().Info.Sprintf("Defaulting to --compute %s for this device; pass --compute to override.", computeUnit))
	}
	deviceID, nglResolved, nctxResolved, err := resolveModelParams(paths.RuntimeID, paths.ModelName)
	if err != nil {
		return err
	}
	spin := render.NewSpinner("loading JEV model...")
	spin.Start()
	switch paths.ModelType {
	case geniex_sdk.ModelTypeLLM:
		llm, err := geniex_sdk.NewLLM(geniex_sdk.LlmCreateInput{ModelName: paths.ModelName, ModelPath: paths.ModelPath, RuntimeID: paths.RuntimeID, DeviceID: deviceID, Config: jevModelConfig(nctxResolved, nglResolved, options)})
		spin.Stop()
		if err != nil {
			return err
		}
		defer llm.Destroy()
		return run(paths, llm, nil)
	case geniex_sdk.ModelTypeVLM:
		vlm, err := geniex_sdk.NewVLM(geniex_sdk.VlmCreateInput{ModelName: paths.ModelName, ModelPath: paths.ModelPath, MmprojPath: paths.MmprojPath, RuntimeID: paths.RuntimeID, DeviceID: deviceID, Config: jevModelConfig(nctxResolved, nglResolved, options)})
		spin.Stop()
		if err != nil {
			return err
		}
		defer vlm.Destroy()
		if len(options.images) != 0 {
			capabilities, err := vlm.Capabilities()
			if err != nil {
				return err
			}
			if !capabilities.SupportsVision && paths.RuntimeID == geniex_sdk.RuntimeLlamaCpp {
				return fmt.Errorf("model %s does not expose vision support", paths.ModelName)
			}
		}
		return run(paths, nil, vlm)
	default:
		return fmt.Errorf("unsupported model type: %s", paths.ModelType)
	}
}

func applyJEVComputeDefault() bool {
	var overridden bool
	computeUnit, overridden = config.ComputeDefault(computeUnit, store.Get().ResolveChipset(true))
	return overridden
}

func runJEV(ctx context.Context, model string, options jevOptions) error {
	if strings.TrimSpace(options.task) == "" {
		return fmt.Errorf("--task is required")
	}
	if options.maxSteps < 1 {
		return fmt.Errorf("--max-steps must be at least 1")
	}
	mode := jev.ApprovalMode(strings.ToLower(options.auto))
	if mode != jev.ApprovalRead && mode != jev.ApprovalNone && mode != jev.ApprovalAll {
		return fmt.Errorf("--auto must be read, none, or all")
	}
	if mode == jev.ApprovalAll {
		fmt.Println(render.GetTheme().Warning.Sprint("Warning: --auto=all bypasses confirmation prompts; model and action validation still apply."))
	}

	name, precision := geniex_sdk.SplitNamePrecision(model)
	paths, err := ensureModelAvailable(ctx, name, precision)
	if err != nil {
		return err
	}
	if paths.ModelType != geniex_sdk.ModelTypeVLM {
		return fmt.Errorf("geniex jev browser requires a VLM; %s is %s", paths.ModelName, paths.ModelType)
	}
	if err := common.InitSDK(); err != nil {
		return err
	}
	defer geniex_sdk.DeInit()

	if overridden := applyJEVComputeDefault(); overridden {
		fmt.Println(render.GetTheme().Info.Sprintf("Defaulting to --compute %s for this device; pass --compute to override.", computeUnit))
	}
	deviceID, nglResolved, nctxResolved, err := resolveModelParams(paths.RuntimeID, paths.ModelName)
	if err != nil {
		return err
	}
	spin := render.NewSpinner("loading visual browser agent model...")
	spin.Start()
	vlm, err := geniex_sdk.NewVLM(geniex_sdk.VlmCreateInput{
		ModelName:  paths.ModelName,
		ModelPath:  paths.ModelPath,
		MmprojPath: paths.MmprojPath,
		RuntimeID:  paths.RuntimeID,
		DeviceID:   deviceID,
		Config:     jevModelConfig(nctxResolved, nglResolved, options),
	})
	spin.Stop()
	if err != nil {
		return err
	}
	defer vlm.Destroy()
	if verbose {
		fmt.Println(render.GetTheme().Info.Sprint(modelLoadedLine(paths.RuntimeID, computeUnit, nglResolved, nctxResolved)))
	}
	capabilities, err := vlm.Capabilities()
	if err != nil {
		return err
	}
	if !capabilities.SupportsVision && paths.RuntimeID == geniex_sdk.RuntimeLlamaCpp {
		return fmt.Errorf("model %s does not expose vision support", paths.ModelName)
	}
	if !capabilities.SupportsVision {
		fmt.Println(render.GetTheme().Warning.Sprint("Vision capability probing is unavailable for this runtime; continuing with the selected VLM."))
	}

	b, err := browser.Open(ctx, browser.Config{
		Browser:     options.browser,
		BrowserPath: options.browserPath,
		AttachURL:   options.attach,
		ProfileDir:  options.profile,
		Headless:    options.headless,
		InitialURL:  options.url,
		TraceDir:    options.traceDir,
	})
	if err != nil {
		return err
	}
	defer b.Close()

	decider := bindingjev.VLMDecider{VLM: vlm, RuntimeID: paths.RuntimeID}
	agent := jev.Agent{
		Browser:  b,
		Decider:  decider,
		Approver: terminalApprover{},
		Config: jev.AgentConfig{
			Task:         options.task,
			MaxSteps:     options.maxSteps,
			MaxRetries:   2,
			ApprovalMode: mode,
			DryRun:       options.dryRun,
		},
	}
	result, err := agent.Run(ctx)
	if err != nil {
		return err
	}
	fmt.Println(render.GetTheme().Success.Sprintf("✔  %s", result.FinalAnswer))
	return nil
}

type terminalApprover struct{}

func (terminalApprover) Approve(request jev.ApprovalRequest) (jev.ApprovalDecision, error) {
	var answer string
	options := []huh.Option[string]{
		huh.NewOption("Approve this action", string(jev.ApprovalApprove)),
		huh.NewOption("Skip this action", string(jev.ApprovalSkip)),
		huh.NewOption("Abort browser task", string(jev.ApprovalAbort)),
	}
	if err := huh.NewSelect[string]().
		Title("Approval required: " + jev.ApprovalSummary(request)).
		Options(options...).
		Value(&answer).
		Run(); err != nil {
		return jev.ApprovalAbort, err
	}
	return jev.ApprovalDecision(answer), nil
}
