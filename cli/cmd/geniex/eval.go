// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/cmd/geniex/common"
	"github.com/qualcomm/GenieX/cli/internal/eval"
	"github.com/qualcomm/GenieX/cli/internal/render"
)

var (
	evalPackRef   string
	evalJSONOut   string
	evalCompute   string
	evalNgl       int32
	evalNctx      int32
	evalMaxTokens int32
)

func evalCmd() *cobra.Command {
	cmd := &cobra.Command{
		GroupID: "inference",
		Use:     "eval <model-name>[:<precision>] [<model-name>...]",
		Short:   "Evaluate and compare models on an eval pack",
		Long: "Run one or more models over an eval pack and compare their accuracy.\n" +
			"--eval selects a built-in pack (" + strings.Join(eval.BuiltinNames(), ", ") + ") or a custom pack from a JSON file.\n" +
			"Runs are greedy (top-k 1, fixed seed) with thinking disabled, so results are repeatable.",
	}
	cmd.Args = cobra.MinimumNArgs(1)

	cmd.Flags().SortFlags = false
	cmd.Flags().StringVarP(&evalPackRef, "eval", "e", "basic", "built-in eval pack name or path to a custom pack (.json)")
	cmd.Flags().StringVarP(&evalJSONOut, "json", "j", "", "also write full per-task results to this JSON file")
	cmd.Flags().StringVarP(&evalCompute, "compute", "c", "", "compute unit to run on: cpu, gpu, npu, or hybrid (default: npu)")
	cmd.Flags().Int32VarP(&evalNgl, "ngl", "n", -1, "number of layers to offload to gpu/npu, -1 = all (llama_cpp only)")
	cmd.Flags().Int32VarP(&evalNctx, "nctx", "", 4096, "context window size (llama_cpp only)")
	cmd.Flags().Int32VarP(&evalMaxTokens, "max-tokens", "", 256, "max tokens generated per task")

	cmd.RunE = func(cmd *cobra.Command, args []string) error {
		pack, err := eval.Resolve(evalPackRef)
		if err != nil {
			return err
		}

		if err := common.InitSDK(); err != nil {
			return err
		}
		defer geniex_sdk.DeInit()

		var reports []*eval.ModelReport
		for _, arg := range args {
			report, err := evalOneModel(cmd, arg, pack)
			if err != nil {
				return err
			}
			reports = append(reports, report)
		}

		fmt.Println()
		fmt.Print(eval.RenderTable(pack, reports))
		if evalJSONOut != "" {
			if err := eval.WriteJSON(evalJSONOut, pack, reports); err != nil {
				return fmt.Errorf("write %s: %w", evalJSONOut, err)
			}
			fmt.Println(render.GetTheme().Info.Sprintf("\nFull results written to %s", evalJSONOut))
		}
		return nil
	}
	return cmd
}

func evalOneModel(cmd *cobra.Command, arg string, pack *eval.Pack) (*eval.ModelReport, error) {
	name, precision := geniex_sdk.SplitNamePrecision(arg)
	paths, err := ensureModelAvailable(cmd.Context(), name, precision)
	if err != nil {
		return nil, err
	}
	if paths.ModelType != geniex_sdk.ModelTypeLLM {
		return nil, fmt.Errorf("model %s is a %s; eval currently supports LLM models only", name, paths.ModelType)
	}

	resolved, err := geniex_sdk.ResolveDevice(geniex_sdk.ResolveDeviceInput{
		RuntimeID:   paths.RuntimeID,
		ModelName:   paths.ModelName,
		ComputeUnit: evalCompute,
		NglDefault:  evalNgl,
	})
	if err != nil {
		return nil, err
	}
	if resolved.Warning != "" {
		fmt.Println(render.GetTheme().Warning.Sprintf("Warning: %s", resolved.Warning))
	}
	nctx := evalNctx
	if paths.RuntimeID != geniex_sdk.RuntimeLlamaCpp {
		nctx = 0
	}

	spin := render.NewSpinner(fmt.Sprintf("loading %s...", name))
	spin.Start()
	p, err := geniex_sdk.NewLLM(geniex_sdk.LlmCreateInput{
		ModelPath: paths.ModelPath,
		RuntimeID: paths.RuntimeID,
		DeviceID:  resolved.DeviceID,
		Config: geniex_sdk.ModelConfig{
			NCtx:       nctx,
			NGpuLayers: resolved.Ngl,
		},
	})
	spin.Stop()
	if err != nil {
		return nil, fmt.Errorf("load model %s: %w", name, err)
	}
	defer p.Destroy()

	// Greedy decoding with a fixed seed keeps runs repeatable.
	sampler := &geniex_sdk.SamplerConfig{TopK: 1, Seed: 42}

	theme := render.GetTheme()
	fmt.Println(theme.Info.Sprintf("Evaluating %s on %s (%d tasks)", name, pack.Name, len(pack.Tasks)))

	report := &eval.ModelReport{Model: name}
	for i, task := range pack.Tasks {
		result := eval.Result{TaskID: task.ID, Category: task.Category}

		// Fresh context per task so tasks cannot leak into each other.
		if err := p.Reset(); err != nil {
			return nil, fmt.Errorf("reset before task %s: %w", task.ID, err)
		}

		tmpl, err := p.ApplyChatTemplate(geniex_sdk.LlmApplyChatTemplateInput{
			Messages: []geniex_sdk.LlmChatMessage{
				{Role: geniex_sdk.LlmRoleUser, Content: eval.RenderPrompt(task)},
			},
			EnableThink:         false,
			AddGenerationPrompt: true,
		})
		if err != nil {
			return nil, fmt.Errorf("apply chat template for task %s: %w", task.ID, err)
		}

		res, err := p.Generate(geniex_sdk.LlmGenerateInput{
			PromptUTF8: tmpl.FormattedText,
			Config: &geniex_sdk.GenerationConfig{
				MaxTokens:     evalMaxTokens,
				SamplerConfig: sampler,
			},
		})
		if err != nil {
			result.Error = err.Error()
		} else {
			result.Output = res.FullText
			result.Correct = eval.Score(task, res.FullText)
			report.DecodeTokens += res.ProfileData.GeneratedTokens
			report.DecodeTimeUs += res.ProfileData.DecodeTime
		}
		report.Results = append(report.Results, result)

		mark := theme.Error.Sprint("✗")
		if result.Correct {
			mark = theme.Success.Sprint("✓")
		}
		fmt.Printf("  [%d/%d] %s %s\n", i+1, len(pack.Tasks), mark, task.ID)
	}
	return report, nil
}
