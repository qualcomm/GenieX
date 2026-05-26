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
	"fmt"
	"log/slog"
	"os"
	"regexp"
	"slices"
	"sort"
	"strings"

	"github.com/charmbracelet/huh"
	"github.com/dustin/go-humanize"
	"github.com/jedib0t/go-pretty/v6/table"
	"github.com/spf13/cobra"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub/aihub"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/render"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/store"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/types"
)

var (
	modelHub  string
	localPath string
	modelType string
)

// pull creates a command to download and cache a model by name.
// Usage: geniex pull <model-name>
func pull() *cobra.Command {
	pullCmd := &cobra.Command{
		GroupID: "model",
		Use:     "pull <model-name>[:<precision>]",

		Short: "Pull model from HuggingFace or Qualcomm AI Hub Models",
		Long:  "Download and cache a model by name. Append ':<precision>' to pull a specific precision; otherwise you'll be prompted to choose one.",
	}

	pullCmd.Args = cobra.MatchAll(cobra.ExactArgs(1), cobra.OnlyValidArgs)

	pullCmd.Flags().SortFlags = false
	pullCmd.Flags().StringVarP(&modelHub, "model-hub", "", "", "specify model hub to use: aihub|hf|localfs")
	pullCmd.Flags().StringVarP(&localPath, "local-path", "", "", "[localfs] path to local directory or aihub zip file")
	pullCmd.Flags().StringVarP(&modelType, "model-type", "", "", "specify model type to use: [llm|vlm]")

	pullCmd.RunE = func(cmd *cobra.Command, args []string) error {
		name, quant := model_hub.NormalizeModelName(args[0])
		if err := pullModel(name, quant); err != nil {
			fmt.Println(render.GetTheme().Error.Sprintf("✘  Failed to pull model: %s", err))
			return err
		}
		return nil
	}

	return pullCmd
}

// Usage: geniex remove <model-name>[:<quant>]
func remove() *cobra.Command {
	var assumeYes bool

	removeCmd := &cobra.Command{
		GroupID: "model",
		Use:     "remove <model-name>[:<precision>] [<model-name>[:<precision>] ...]",
		Aliases: []string{"rm"},
		Short:   "Remove cached model",
		Long:    "Delete a cached model by name. Append ':<precision>' to remove a single precision; otherwise the whole model is removed.",
	}

	removeCmd.Args = cobra.MatchAll(cobra.MinimumNArgs(1), cobra.OnlyValidArgs)
	removeCmd.Flags().BoolVarP(&assumeYes, "yes", "y", false, "Skip the confirmation prompt")

	removeCmd.RunE = func(cmd *cobra.Command, args []string) error {
		if !assumeYes {
			title := fmt.Sprintf("Are you sure you want to delete %s?", args[0])
			if len(args) > 1 {
				title = fmt.Sprintf("Are you sure you want to delete %d models?\n  %s",
					len(args), strings.Join(args, "\n  "))
			}
			var ok bool
			if err := huh.NewConfirm().Title(title).Value(&ok).Run(); err != nil {
				return err
			}
			if !ok {
				fmt.Println(render.GetTheme().Info.Sprint("Aborted"))
				return nil
			}
		}

		s := store.Get()
		var lastErr error
		for _, arg := range args {
			name, quant := model_hub.NormalizeModelName(arg)
			label := name
			if quant != "" {
				label = name + ":" + quant
			}
			if err := s.Remove(name, quant); err != nil {
				fmt.Println(render.GetTheme().Error.Sprintf("✘  Failed to remove %s: %s", label, err))
				lastErr = err
				continue
			}
			fmt.Println(render.GetTheme().Success.Sprintf("✔  Removed %s", label))
		}
		return lastErr
	}

	return removeCmd
}

// clean creates a command to remove all cached models and free up storage.
// Usage: geniex clean
func clean() *cobra.Command {
	cleanCmd := &cobra.Command{
		GroupID: "model",
		Use:     "clean",
		Short:   "remove all cached models",
		Long:    "Remove all cached models and free up storage. This will delete all model files from the local cache.",
	}

	cleanCmd.Run = func(cmd *cobra.Command, args []string) {
		s := store.Get()
		c := s.Clean()
		fmt.Println(render.GetTheme().Success.Sprintf("✔  Removed %d models", c))
	}

	return cleanCmd
}

// list creates a command to display all cached models in a formatted table.
// Shows model names and their storage sizes.
// Usage: geniex list
func list() *cobra.Command {
	listCmd := &cobra.Command{
		GroupID: "model",
		Use:     "list",
		Aliases: []string{"ls"},
		Short:   "List all cached models",
		Long:    "Display all cached models in a formatted table, showing model names, types, and sizes.",
	}

	listCmd.RunE = func(cmd *cobra.Command, args []string) error {
		s := store.Get()
		models, e := s.List()
		if e != nil {
			fmt.Println(render.GetTheme().Error.Sprint(e))
			return e
		}

		// Create formatted table output
		tw := table.NewWriter()
		tw.SetOutputMirror(os.Stdout)
		tw.SetStyle(table.StyleLight)
		if verbose {
			tw.AppendHeader(table.Row{"NAME", "SIZE", "PLUGIN", "TYPE", "PRECISIONS"})
			for _, model := range models {
				tw.AppendRow(table.Row{
					model.Name,
					humanize.IBytes(uint64(model.GetSize())),
					model.PluginId,
					model.ModelType,
					strings.Join(func() []string {
						quants := make([]string, 0)
						for q := range model.ModelFile {
							if model.ModelFile[q].Downloaded {
								quants = append(quants, q)
							}
						}
						slices.Sort(quants)
						return quants
					}(), ","),
				})
			}
		} else {
			tw.AppendHeader(table.Row{"NAME", "SIZE", "PRECISIONS"})
			for _, model := range models {
				tw.AppendRow(table.Row{model.Name, humanize.IBytes(uint64(model.GetSize())), strings.Join(func() []string {
					quants := make([]string, 0)
					for q := range model.ModelFile {
						if model.ModelFile[q].Downloaded && q != "N/A" {
							quants = append(quants, q)
						}
					}
					slices.Sort(quants)
					return quants
				}(), ",")})
			}
		}
		tw.Render()
		return nil
	}

	return listCmd
}

// modelCmd builds the `geniex model` command tree:
//
// It is the home for all per-model management operations
func modelCmd() *cobra.Command {
	cmd := &cobra.Command{
		GroupID: "model",
		Use:     "model",
		Short:   "Manage cached models",
		Long:    "Commands to manage cached models, including reconfiguring model-specific settings.",
	}
	cmd.AddCommand(setTypeCmd())
	return cmd
}

// setTypeCmd builds the `geniex model set-type` subcommand.
// It overwrites the ModelType field in an already-downloaded model's manifest.
func setTypeCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "set-type <model-name> [llm|vlm]",
		Short: "Override the model type for a cached model",
		Long:  "Update the model type stored in a cached model's manifest.\n\nOmit the type argument to choose interactively.",
		Args:  cobra.RangeArgs(1, 2),
		ValidArgs: func() []string {
			s := make([]string, len(types.AllModelTypes))
			for i, t := range types.AllModelTypes {
				s[i] = string(t)
			}
			return s
		}(),
		Run: func(cmd *cobra.Command, args []string) {
			name, _ := model_hub.NormalizeModelName(args[0])

			// Verify the model is present before prompting for a type.
			if _, err := store.Get().GetManifest(name); err != nil {
				fmt.Fprintln(os.Stderr, render.GetTheme().Error.Sprintf("Model %q not found: %s", name, err))
				return
			}

			var mt types.ModelType
			if len(args) == 2 {
				mt = types.ModelType(strings.ToLower(args[1]))
				valid := false
				for _, t := range types.AllModelTypes {
					if mt == t {
						valid = true
						break
					}
				}
				if !valid {
					validStrs := make([]string, len(types.AllModelTypes))
					for i, t := range types.AllModelTypes {
						validStrs[i] = string(t)
					}
					fmt.Fprintln(os.Stderr, render.GetTheme().Error.Sprintf(
						"Unknown model type %q (valid: %s)", args[1], strings.Join(validStrs, ", ")))
					return
				}
			} else {
				if err := huh.NewSelect[types.ModelType]().
					Title("Choose Model Type").
					Options(huh.NewOptions(types.AllModelTypes...)...).
					Value(&mt).
					Run(); err != nil {
					return
				}
			}

			if err := store.Get().SetModelType(name, mt); err != nil {
				fmt.Fprintln(os.Stderr, render.GetTheme().Error.Sprintf("Failed to update model type: %s", err))
				return
			}
			fmt.Println(render.GetTheme().Success.Sprintf("✔  %s → %s", name, mt))
		},
	}
}

func pullModel(name string, quant string) error {
	slog.Debug("pullModel", "name", name, "quant", quant)

	s := store.Get()

	mf, err := s.GetManifest(name)
	if err != nil && !os.IsNotExist(err) { // model not exist is not an error, we will create it
		return err
	}

	// specify model hub
	if localPath != "" && modelHub == "" {
		modelHub = "localfs"
	}
	if modelHub != "" {
		switch strings.ToLower(modelHub) {
		case "aihub":
			model_hub.SetHub(model_hub.NewAIHub(chipsetGet(s)))
		case "hf", "huggingface":
			model_hub.SetHub(model_hub.NewHuggingFace())
		case "local", "localfs":
			if localPath == "" {
				return fmt.Errorf("local path is required for localfs model hub")
			}
			model_hub.SetHub(model_hub.NewLocalFS(localPath))
		default:
			return fmt.Errorf("unknown model hub: %s", modelHub)
		}
	}

	// Resolve chipset before the spinner — chipsetEnsure may pop an
	// interactive device picker, which can't share the terminal with a
	// spinner. Skip when the explicit hub doesn't need it.
	if _, ok := aihub.IsAIHubName(name); ok {
		if err := chipsetEnsure(context.TODO(), s); err != nil {
			return err
		}
	}

	spin := render.NewSpinner("download manifest from: " + name)
	spin.Start()
	files, hmf, err := model_hub.ModelInfo(context.TODO(), name)
	spin.Stop()
	if err != nil {
		return err
	}

	if mf != nil {
		// check all downloaded
		downloaded := true
		for _, f := range mf.ModelFile {
			if !f.Downloaded {
				downloaded = false
				break
			}
		}
		if downloaded {
			fmt.Println(render.GetTheme().Info.Sprint("Already downloaded all precisions"))
			return nil
		}

		oldSize := mf.GetSize()

		// choose quant to download
		err := chooseQuantFiles(quant, mf)
		if err != nil {
			return err
		}

		// start download
		pgCh, errCh := s.PullExtraQuant(context.TODO(), *mf)
		bar := render.NewProgressBar(mf.GetSize()-oldSize, "downloading")
		for pg := range pgCh {
			bar.Set(pg.TotalDownloaded)
		}
		bar.Exit()

		for err := range errCh {
			bar.Clear()
			return err
		}
	} else {
		var manifest types.ModelManifest

		if hmf != nil {
			manifest.ModelName = hmf.ModelName
			manifest.PluginId = hmf.PluginId
			manifest.ModelType = hmf.ModelType
		}

		if manifest.PluginId == "" {
			pid, err := model_hub.ChoosePluginId(files)
			if err != nil {
				return err
			}
			manifest.PluginId = pid
		}

		err := chooseFiles(name, quant, files, &manifest)
		if err != nil {
			return err
		}

		// --model-type flag overrides all auto-detection. Validate early so
		// we fail before downloading anything.
		if modelType != "" {
			mt := types.ModelType(strings.ToLower(modelType))
			if !slices.Contains(types.AllModelTypes, mt) {
				return fmt.Errorf("unknown model type %q (valid: %s)", modelType,
					strings.Join(func() []string {
						s := make([]string, len(types.AllModelTypes))
						for i, t := range types.AllModelTypes {
							s[i] = string(t)
						}
						return s
					}(), ", "))
			}
			manifest.ModelType = mt
		}

		// start download
		pgCh, errCh := s.Pull(context.TODO(), manifest)
		bar := render.NewProgressBar(manifest.GetSize(), "downloading")
		for pg := range pgCh {
			bar.Set(pg.TotalDownloaded)
		}
		bar.Exit()

		for err := range errCh {
			bar.Clear()
			return err
		}

		detectedType := manifest.ModelType
		if updated, err := s.GetManifest(name); err == nil {
			detectedType = updated.ModelType
		}

		if detectedType == "" {
			fmt.Println(render.GetTheme().Warning.Sprintf(
				"⚠  Could not detect model type; run:\n"+
					"     geniex model set-type %s <llm|vlm>", name))
		} else {
			fmt.Println(render.GetTheme().Info.Sprintf("   Detected model type: %s", detectedType))
		}
	}

	fmt.Println(render.GetTheme().Success.Sprintf("✔  Download success"))

	return nil
}

// =============== quant name parse ===============
var quantRegix = regexp.MustCompile(`(` + strings.Join([]string{
	"[fF][pP][0-9]+",                 // FP32, FP16, FP64
	"[fF][0-9]+",                     // F64, F32, F16
	"[iI][0-9]+",                     // I64, I32, I16, I8
	"[qQ][0-9]+(_[A-Za-z0-9]+)*",     // Q8_0, Q8_1, Q8_K, Q6_K, Q5_0, Q5_1, Q5_K, Q4_0, Q4_1, Q4_K, Q3_K, Q2_K
	"[iI][qQ][0-9]+(_[A-Za-z0-9]+)*", // IQ4_NL, IQ4_XS, IQ3_S, IQ3_XXS, IQ2_XXS, IQ2_S, IQ2_XS, IQ1_S, IQ1_M
	"[bB][fF][0-9]+",                 // BF16
	"[0-9]+[bB][iI][tT]",             // 1bit, 2bit, 3bit, 4bit, 16bit, 1BIT, 16Bit, etc.
}, "|") + `)`)

func getQuant(name string) string {
	quant := strings.ToUpper(quantRegix.FindString(name))
	if quant == "" {
		quant = "N/A"
	}
	return quant
}

var partRegex = regexp.MustCompile(`-\d+-of-\d+\.gguf$`)

func chooseFiles(name, specifiedQuant string, files []model_hub.ModelFileInfo, res *types.ModelManifest) (err error) {
	if len(files) == 0 {
		err = fmt.Errorf("repo is empty")
		return
	}

	res.Name = name
	res.ModelFile = make(map[string]types.ModelFileInfo)

	// check gguf
	var mmprojs []model_hub.ModelFileInfo
	ggufs := make(map[string][]model_hub.ModelFileInfo) // key is gguf name without part
	// qwen2.5-7b-instruct-q8_0-00003-of-00003.gguf original name is qwen2.5-7b-instruct-q8_0 *d-of-*d like this

	for _, file := range files {
		name := strings.ToLower(file.Name)
		if strings.HasSuffix(name, ".gguf") {
			if strings.HasPrefix(name, "mmproj") {
				mmprojs = append(mmprojs, file)
			} else {
				name := partRegex.ReplaceAllString(file.Name, "")
				ggufs[name] = append(ggufs[name], file)
			}
		}
	}

	// choose model file
	if len(ggufs) > 0 {
		// gguf models

		if len(ggufs) == 1 {
			// single quant
			fileInfo := types.ModelFileInfo{}
			for name, gguf := range ggufs {
				fileInfo.Name = gguf[0].Name
				fileInfo.Size = gguf[0].Size
				fileInfo.Downloaded = true
				res.ModelFile[getQuant(name)] = fileInfo
				// other fragments
				for _, file := range gguf[1:] {
					res.ExtraFiles = append(res.ExtraFiles, types.ModelFileInfo{
						Name:       file.Name,
						Downloaded: true,
						Size:       file.Size,
					})
				}
			}
			if specifiedQuant != "" && res.ModelFile[specifiedQuant].Name == "" {
				return fmt.Errorf("Specified precision %s not found", specifiedQuant)
			}

		} else {
			// choose quant
			var file string

			// sort key by quant
			ggufNames := make([]string, 0, len(ggufs))
			for k := range ggufs {
				ggufNames = append(ggufNames, k)

				if file == "" {
					file = k
					continue
				}

				// prefer Q4_0, Q4_K_M, Q8_0
				kq := getQuant(k)
				fq := getQuant(file)
				sortKey := []string{"Q8_0", "Q4_K_M", "Q4_0"}
				if slices.Index(sortKey, kq) > slices.Index(sortKey, fq) {
					file = k
				}
			}
			sort.Slice(ggufNames, func(i, j int) bool {
				return sumSize(ggufs[ggufNames[i]]) > sumSize(ggufs[ggufNames[j]])
			})

			if specifiedQuant != "" {
				for _, ggufName := range ggufNames {
					if getQuant(ggufName) == specifiedQuant {
						file = ggufName
						break
					}
				}
				if getQuant(file) != specifiedQuant {
					return fmt.Errorf("specified precision %s not found", specifiedQuant)
				}
			} else {
				var options []huh.Option[string]
				for _, ggufName := range ggufNames {
					fmtStr := "%-10s [%7s]"
					if ggufName == file {
						fmtStr += " (default)"
					}
					options = append(options, huh.NewOption(
						fmt.Sprintf(fmtStr, getQuant(ggufName), humanize.IBytes(uint64(sumSize(ggufs[ggufName])))),
						ggufName,
					))
				}

				if err = huh.NewSelect[string]().
					Title("Choose a precision version to download").
					Options(options...).
					Value(&file).
					Run(); err != nil {
					return err
				}
			}

			for k, gguf := range ggufs {
				downloaded := k == file
				// sort files by name
				sort.Slice(gguf, func(i, j int) bool {
					return gguf[i].Name < gguf[j].Name
				})
				res.ModelFile[getQuant(k)] = types.ModelFileInfo{
					Name:       gguf[0].Name,
					Downloaded: downloaded,
					Size:       sumSize(ggufs[k]),
				}
				for _, file := range ggufs[k][1:] {
					res.ExtraFiles = append(res.ExtraFiles, types.ModelFileInfo{
						Name:       file.Name,
						Downloaded: downloaded,
						Size:       file.Size,
					})
				}
			}
		}

		// detect mmproj: pick the biggest when multiple are present
		var biggest model_hub.ModelFileInfo
		for _, mmproj := range mmprojs {
			if mmproj.Size > biggest.Size {
				biggest = mmproj
			}
		}
		if biggest.Name != "" {
			res.MMProjFile = types.ModelFileInfo{
				Name:       biggest.Name,
				Size:       biggest.Size,
				Downloaded: true,
			}
		}

	} else {
		// qairt models

		if specifiedQuant != "" {
			return fmt.Errorf("specified precision %s only supported in gguf model", specifiedQuant)
		}

		if len(files) == 1 {
			// single file (typically the AI Hub .zip)
			mainFile := files[0]
			res.ModelFile["N/A"] = types.ModelFileInfo{
				Name:       mainFile.Name,
				Downloaded: true,
				Size:       mainFile.Size,
			}
		} else {
			// folder structure
			for _, f := range files {
				res.ExtraFiles = append(res.ExtraFiles, types.ModelFileInfo{
					Name:       f.Name,
					Downloaded: true,
					Size:       f.Size,
				})
			}
		}
	}

	return
}

func chooseQuantFiles(specifiedQuant string, res *types.ModelManifest) error {
	// sort key by quant
	ggufQuants := make([]string, 0, len(res.ModelFile))
	for k := range res.ModelFile {
		ggufQuants = append(ggufQuants, k)
	}
	sort.Slice(ggufQuants, func(i, j int) bool {
		return res.ModelFile[ggufQuants[i]].Size > res.ModelFile[ggufQuants[j]].Size
	})

	// choose quant
	var quant string
	if specifiedQuant != "" {
		if fileinfo, ok := res.ModelFile[specifiedQuant]; !ok {
			return fmt.Errorf("specified precision %s not found", specifiedQuant)
		} else if fileinfo.Downloaded {
			return fmt.Errorf("specified precision %s already downloaded", specifiedQuant)
		}
		quant = specifiedQuant
	} else {
		options := make([]huh.Option[string], 0, len(res.ModelFile))
		for _, q := range ggufQuants {
			m := res.ModelFile[q]
			if m.Downloaded {
				continue
			}
			options = append(options, huh.NewOption(
				fmt.Sprintf("%-10s [%7s]", q, humanize.IBytes(uint64(m.Size))), q,
			))
		}

		if err := huh.NewSelect[string]().
			Title("Choose a precision version to download").
			Options(options...).
			Value(&quant).
			Run(); err != nil {
			return err
		}
	}

	res.ModelFile[quant] = types.ModelFileInfo{
		Name:       res.ModelFile[quant].Name,
		Downloaded: true,
		Size:       res.ModelFile[quant].Size,
	}

	// other fragments
	file := res.ModelFile[quant].Name
	ggufName := partRegex.ReplaceAllString(file, "")
	for i, f := range res.ExtraFiles {
		if ggufName == partRegex.ReplaceAllString(f.Name, "") {
			res.ExtraFiles[i] = types.ModelFileInfo{
				Name:       f.Name,
				Downloaded: true,
			}
		}

	}

	return nil
}

func sumSize(files []model_hub.ModelFileInfo) int64 {
	var size int64
	for _, f := range files {
		size += f.Size
	}
	return size
}
