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
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	geniex_sdk "github.com/qcom-it-nexa-ai/geniex/bindings/go"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/render"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/store"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/types"
)

var tool []string

func functionCall() *cobra.Command {
	fcCmd := &cobra.Command{
		GroupID: "inference",
		Use:     "functioncall <model-name>",
		Aliases: []string{"fc"},
		Short:   "Function call with a model",
		Long:    "Run function call with a specified model. The model must be downloaded and cached locally.",
	}

	fcCmd.Args = cobra.MatchAll(cobra.ExactArgs(1), cobra.OnlyValidArgs)

	fcCmd.Flags().SortFlags = false
	fcCmd.Flags().StringArrayVarP(&tool, "tool", "t", nil, "[llm|vlm] add function name for function call")
	fcCmd.Flags().StringArrayVarP(&prompt, "prompt", "p", nil, "[llm|vlm] pass prompt")

	fcCmd.Run = func(cmd *cobra.Command, args []string) {
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

		modelfile := s.ModelfilePath(manifest.Name, manifest.ModelFile[quant].Name)

		switch manifest.ModelType {
		case types.ModelTypeLLM:
			err = fcLLM(manifest.PluginId, modelfile)
		case types.ModelTypeVLM:
			var mmprojfile string
			if manifest.MMProjFile.Name != "" {
				mmprojfile = s.ModelfilePath(manifest.Name, manifest.MMProjFile.Name)
			}
			var tokenizerfile string
			if manifest.TokenizerFile.Name != "" {
				tokenizerfile = s.ModelfilePath(manifest.Name, manifest.TokenizerFile.Name)
			}
			err = fcVLM(manifest.PluginId, modelfile, mmprojfile, tokenizerfile)
		default:
			panic("not support model type")
		}

		if err != nil {
			os.Exit(1)
		}
	}
	return fcCmd
}

func checkParseTools(tools []string) (string, error) {
	if len(tools) == 0 {
		return "", fmt.Errorf("no tools provided")
	}
	return "[" + strings.Join(tools, ",") + "]", nil
}

func fcLLM(plugin, modelfile string) error {
	tools, err := checkParseTools(tool)
	if err != nil {
		fmt.Println(render.GetTheme().Error.Sprint(err))
		return err
	}

	spin := render.NewSpinner("loading model...")
	spin.Start()
	p, err := geniex_sdk.NewLLM(geniex_sdk.LlmCreateInput{
		ModelPath: modelfile,
		PluginID:  plugin,
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

	messages := make([]geniex_sdk.LlmChatMessage, len(prompt))
	for i, p := range prompt {
		messages[i] = geniex_sdk.LlmChatMessage{Role: geniex_sdk.LLMRoleUser, Content: p}
	}
	templateOutput, err := p.ApplyChatTemplate(geniex_sdk.LlmApplyChatTemplateInput{
		Messages:            messages,
		EnableThink:         false, // disable thinking mode for function call mode
		Tools:               tools,
		AddGenerationPrompt: true,
	})
	if err != nil {
		fmt.Println(render.GetTheme().Error.Sprintf("apply chat template error: %s", err))
		return err
	}
	res, err := p.Generate(geniex_sdk.LlmGenerateInput{
		PromptUTF8: templateOutput.FormattedText,
		Config: &geniex_sdk.GenerationConfig{
			MaxTokens: 2048,
		},
	},
	)
	if err != nil {
		fmt.Println(render.GetTheme().Error.Sprintf("generate error: %s", err))
		return err
	}

	fmt.Println()
	fmt.Println(render.GetTheme().Success.Sprintf("%s", res.FullText))
	fmt.Println()

	return nil
}

func fcVLM(plugin, modelfile, mmprojfile, tokenizerfile string) error {
	tools, err := checkParseTools(tool)
	if err != nil {
		fmt.Println(render.GetTheme().Error.Sprint(err))
		return err
	}
	spin := render.NewSpinner("loading model...")
	spin.Start()
	p, err := geniex_sdk.NewVLM(geniex_sdk.VlmCreateInput{
		ModelPath:     modelfile,
		MmprojPath:    mmprojfile,
		TokenizerPath: tokenizerfile,
		PluginID:      plugin,
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

	messages := make([]geniex_sdk.VlmChatMessage, len(prompt))
	for i, p := range prompt {
		messages[i] = geniex_sdk.VlmChatMessage{Role: geniex_sdk.VlmRoleUser, Contents: []geniex_sdk.VlmContent{{Type: geniex_sdk.VlmContentTypeText, Text: p}}}
	}
	templateOutput, err := p.ApplyChatTemplate(geniex_sdk.VlmApplyChatTemplateInput{
		Messages:    messages,
		EnableThink: false, // disable thinking mode for function call mode
		Tools:       tools,
	})
	if err != nil {
		fmt.Println(render.GetTheme().Error.Sprintf("apply chat template error: %s", err))
		return err
	}
	res, err := p.Generate(geniex_sdk.VlmGenerateInput{
		PromptUTF8: templateOutput.FormattedText,
		Config: &geniex_sdk.GenerationConfig{
			MaxTokens: 2048,
		},
	})
	if err != nil {
		fmt.Println(render.GetTheme().Error.Sprintf("generate error: %s", err))
		return err
	}

	fmt.Println()
	fmt.Println(render.GetTheme().Success.Sprintf("%s", res.FullText))
	fmt.Println()

	return nil
}
