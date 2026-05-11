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

	"github.com/bytedance/sonic"

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
		Use:     "pull <model-name>",

		Short: "Pull model from HuggingFace",
		Long:  "Download and cache a model by name.",
	}

	pullCmd.Args = cobra.MatchAll(cobra.ExactArgs(1), cobra.OnlyValidArgs)

	pullCmd.Flags().SortFlags = false
	pullCmd.Flags().StringVarP(&modelHub, "model-hub", "", "", "specify model hub to use: aihub|hf|localfs")
	pullCmd.Flags().StringVarP(&localPath, "local-path", "", "", "[localfs] path to local directory")
	pullCmd.Flags().StringVarP(&modelType, "model-type", "", "", "specify model type to use: [llm|vlm]")

	pullCmd.Run = func(cmd *cobra.Command, args []string) {
		name, quant := normalizeModelName(args[0])
		if err := pullModel(name, quant); err != nil {
			fmt.Println(render.GetTheme().Error.Sprintf("✘  Failed to pull model: %s", err))
			os.Exit(1)
		}
	}

	return pullCmd
}

// remove creates a command to delete a cached model by name.
// Usage: geniex remove <model-name>
func remove() *cobra.Command {
	removeCmd := &cobra.Command{
		GroupID: "model",
		Use:     "remove <model-name> [<model-name> ...]",
		Aliases: []string{"rm"},
		Short:   "Remove cached model",
		Long:    "Delete a cached model by name. This will remove the model files from the local cache.",
	}

	removeCmd.Args = cobra.MatchAll(cobra.MinimumNArgs(1), cobra.OnlyValidArgs)

	removeCmd.Run = func(cmd *cobra.Command, args []string) {
		for _, arg := range args {
			name, quant := normalizeModelName(arg)
			if quant != "" {
				fmt.Println(render.GetTheme().Error.Sprintf("Currently not support remove a single quant, please remove the whole model: %s", name))
				os.Exit(1)
			}

			s := store.Get()
			e := s.Remove(name)
			if e != nil {
				fmt.Println(render.GetTheme().Error.Sprintf("✘  Failed to remove model: %s", name))
				os.Exit(1)
			} else {
				fmt.Println(render.GetTheme().Success.Sprintf("✔  Removed %s", name))
			}
		}
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

	listCmd.Run = func(cmd *cobra.Command, args []string) {
		s := store.Get()
		models, e := s.List()
		if e != nil {
			fmt.Println(e)
			os.Exit(1)
		}

		// Create formatted table output
		tw := table.NewWriter()
		tw.SetOutputMirror(os.Stdout)
		tw.SetStyle(table.StyleLight)
		if verbose {
			tw.AppendHeader(table.Row{"NAME", "SIZE", "PLUGIN", "TYPE", "QUANTS"})
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
			tw.AppendHeader(table.Row{"NAME", "SIZE", "QUANTS"})
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
	}

	return listCmd
}

func pullModel(name string, quant string) error {
	slog.Debug("pullModel", "name", name, "quant", quant)

	s := store.Get()

	mf, err := s.GetManifest(name)
	if err == nil {
		downloaded := true
		for _, f := range mf.ModelFile {
			if !f.Downloaded {
				downloaded = false
				break
			}
		}

		if downloaded {
			fmt.Println(render.GetTheme().Info.Sprint("Already downloaded all quant"))
			return nil
		}
	}

	// specify model hub
	if localPath != "" && modelHub == "" {
		modelHub = "localfs"
	}
	if modelHub != "" {
		switch strings.ToLower(modelHub) {
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

	if _, ok := aihub.IsAIHubName(name); ok {
		if err := ensureChipset(context.TODO()); err != nil {
			return err
		}
	}

	spin := render.NewSpinner("download manifest from: " + name)
	spin.Start()
	files, hmf, err := model_hub.ModelInfo(context.TODO(), name)
	spin.Stop()
	if err != nil {
		fmt.Println(render.GetTheme().Error.Sprintf("Get ModelInfo error: %s", err))
		return err
	}

	if mf != nil {
		// deepcopy manifest
		var omf types.ModelManifest
		mfs, _ := sonic.Marshal(mf)
		sonic.Unmarshal(mfs, &omf)

		err := chooseQuantFiles(quant, mf)
		if err != nil {
			fmt.Println(render.GetTheme().Error.Sprintf("Error: %s", err))
			return err
		}
		pgCh, errCh := s.PullExtraQuant(context.TODO(), omf, *mf)
		bar := render.NewProgressBar(mf.GetSize()-omf.GetSize(), "downloading")

		for pg := range pgCh {
			bar.Set(pg.TotalDownloaded)
		}
		bar.Exit()

		for err := range errCh {
			bar.Clear()
			fmt.Println(render.GetTheme().Error.Sprintf("Error: %s", err))
			return err
		}
	} else {
		var manifest types.ModelManifest

		if hmf != nil {
			manifest.ModelName = hmf.ModelName
			manifest.PluginId = hmf.PluginId
			manifest.DeviceId = hmf.DeviceId
			manifest.ModelType = hmf.ModelType
		}

		if manifest.ModelName == "" {
			manifest.ModelName = name
		}
		if manifest.PluginId == "" {
			manifest.PluginId = choosePluginId(name)
		}
		if manifest.ModelType == "" {
			if ctype, err := chooseModelType(); err != nil {
				fmt.Println(render.GetTheme().Error.Sprintf("Error: %s", err))
				return err
			} else {
				manifest.ModelType = ctype
			}
		}

		err := chooseFiles(name, quant, files, &manifest)
		if err != nil {
			fmt.Println(render.GetTheme().Error.Sprintf("Error: %s", err))
			return err
		}

		pgCh, errCh := s.Pull(context.TODO(), manifest)

		bar := render.NewProgressBar(manifest.GetSize(), "downloading")

		for pg := range pgCh {
			bar.Set(pg.TotalDownloaded)
		}
		bar.Exit()

		for err := range errCh {
			bar.Clear()
			fmt.Println(render.GetTheme().Error.Sprintf("Error: %s", err))
			return err
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

func choosePluginId(name string) string {
	name = strings.ToLower(name)
	switch {
	// prefer plugin by model name keyword
	default:
		return "llama_cpp"
	}

}

func chooseModelType() (types.ModelType, error) {
	if modelType != "" {
		mt := types.ModelType(modelType)
		if !slices.Contains(types.AllModelTypes, mt) {
			return "", fmt.Errorf("unknown model type: %s", modelType)
		}
		return mt, nil
	}

	var modelType types.ModelType
	if err := huh.NewSelect[types.ModelType]().
		Title("Choose Model Type").
		Options(huh.NewOptions(types.AllModelTypes...)...).
		Value(&modelType).
		Run(); err != nil {
		return "", err
	}
	return modelType, nil
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
	var tokenizers []model_hub.ModelFileInfo
	var onnxFiles []model_hub.ModelFileInfo
	var geniexFiles []model_hub.ModelFileInfo
	var npyFiles []model_hub.ModelFileInfo
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
		} else if strings.HasSuffix(name, "tokenizer.json") {
			tokenizers = append(tokenizers, file)
		} else if strings.HasSuffix(name, ".onnx") {
			onnxFiles = append(onnxFiles, file)
		} else if strings.HasSuffix(name, ".geniex") {
			geniexFiles = append(geniexFiles, file)
		} else if strings.HasSuffix(name, ".npy") {
			npyFiles = append(npyFiles, file)
		}
	}

	// choose model file
	if len(ggufs) > 0 {
		// detect gguf
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
				return fmt.Errorf("Specified quant %s not found", specifiedQuant)
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
					return fmt.Errorf("specified quant %s not found", specifiedQuant)
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
					Title("Choose a quant version to download").
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

		// detect mmproj
		switch len(mmprojs) {
		case 0:
			// fallback to onnx file as mmproj if no regular mmproj found and exactly one onnx file exists
			if len(onnxFiles) == 1 {
				res.MMProjFile.Name = onnxFiles[0].Name
				res.MMProjFile.Size = onnxFiles[0].Size
				res.MMProjFile.Downloaded = true
			} else if len(geniexFiles) == 1 {
				// fallback to geniex file as mmproj if no onnx file and exactly one geniex file exists
				res.MMProjFile.Name = geniexFiles[0].Name
				res.MMProjFile.Size = geniexFiles[0].Size
				res.MMProjFile.Downloaded = true
			}
		case 1:
			res.MMProjFile.Name = mmprojs[0].Name
			res.MMProjFile.Size = mmprojs[0].Size
			res.MMProjFile.Downloaded = true

		default:
			// match biggest
			var file model_hub.ModelFileInfo
			for _, mmproj := range mmprojs {
				if mmproj.Size > file.Size {
					file = mmproj
				}
			}

			res.MMProjFile.Name = file.Name
			res.MMProjFile.Size = file.Size
			res.MMProjFile.Downloaded = true
		}

		// detect tokenizer for gguf models
		switch len(tokenizers) {
		case 0:
			// No tokenizer file found - skip
		case 1:
			res.TokenizerFile.Name = tokenizers[0].Name
			res.TokenizerFile.Size = tokenizers[0].Size
			res.TokenizerFile.Downloaded = true

		default:
			return fmt.Errorf("multiple tokenizer files found: %v. Expected exactly one tokenizer file", tokenizers)
		}

		// Always include .geniex files as extra files when gguf is the main model, except if used as mmproj
		for _, geniexFile := range geniexFiles {
			// Skip if this geniex file is being used as mmproj
			if res.MMProjFile.Name != geniexFile.Name {
				res.ExtraFiles = append(res.ExtraFiles, types.ModelFileInfo{
					Name:       geniexFile.Name,
					Downloaded: true,
					Size:       geniexFile.Size,
				})
			}
		}

		// Always include .npy files as extra files when gguf is the main model
		for _, npyFile := range npyFiles {
			res.ExtraFiles = append(res.ExtraFiles, types.ModelFileInfo{
				Name:       npyFile.Name,
				Downloaded: true,
				Size:       npyFile.Size,
			})
		}

	} else {
		// other
		if specifiedQuant != "" {
			return fmt.Errorf("specified quant %s only support in gguf model", specifiedQuant)
		}

		// quant
		quant := getQuant(name)
		if quant == "N/A" {
			if q, err := model_hub.GetFileContent(context.TODO(), name, "config.json"); err != nil {
			} else if b, err := sonic.Get(q, "quantization_config", "bits"); err != nil {
			} else if q, err := b.Float64(); err != nil {
			} else {
				quant = fmt.Sprintf("%dBIT", uint32(q))
			}
		}

		// Detect macOS model bundles (.mlmodelc and .mlpackage)
		// These appear as folders on HuggingFace but are actually model bundles
		bundlePaths := detectMacOSBundles(files)

		if len(bundlePaths) > 0 {
			// Use the first bundle as the model path
			bundlePath := bundlePaths[0]

			// Calculate total size of the primary bundle
			var primaryBundleSize int64
			for _, file := range files {
				if strings.HasPrefix(file.Name, bundlePath+"/") {
					primaryBundleSize += file.Size
				}
			}

			// Set the first bundle path as the model file (this is a directory reference, not a downloadable file)
			res.ModelFile[quant] = types.ModelFileInfo{
				Name:       bundlePath,
				Downloaded: true, // Mark as available for inference
				Size:       primaryBundleSize,
			}

			// Add ALL files to ExtraFiles - this includes:
			// 1. All files from the primary bundle
			// 2. All files from other bundles
			// 3. All other files in the repo
			// The bundle paths themselves are not downloadable, only the files within them are
			for _, file := range files {
				res.ExtraFiles = append(res.ExtraFiles, types.ModelFileInfo{
					Name:       file.Name,
					Downloaded: true,
					Size:       file.Size,
				})
			}
		} else {
			// Original logic for non-bundle files
			// detect main model file
			isSupportedModelFile := func(filename string) bool {
				lower := strings.ToLower(filename)
				return strings.HasSuffix(lower, "safetensors") ||
					strings.HasSuffix(lower, "npz") ||
					strings.HasSuffix(lower, "geniex") ||
					strings.HasSuffix(lower, "bin")
			}

			// First pass: prefer non-nested supported files (not in subdirectories)
			for _, file := range files {
				if isSupportedModelFile(file.Name) && !strings.Contains(file.Name, "/") {
					res.ModelFile[quant] = types.ModelFileInfo{Name: file.Name, Size: file.Size}
					break
				}
			}

			// Second pass: if no non-nested file found, fall back to any supported file
			if res.ModelFile[quant].Name == "" {
				for _, file := range files {
					if isSupportedModelFile(file.Name) {
						res.ModelFile[quant] = types.ModelFileInfo{Name: file.Name, Size: file.Size}
						break
					}
				}
			}

			// add other files to ExtraFiles
			for _, file := range files {
				if file.Name != res.ModelFile[quant].Name {
					res.ExtraFiles = append(res.ExtraFiles, types.ModelFileInfo{Name: file.Name, Size: file.Size})
				}
			}

			// fallback to first file
			if res.ModelFile[quant].Name == "" {
				res.ModelFile[quant] = types.ModelFileInfo{Name: files[0].Name, Size: files[0].Size}
				res.ExtraFiles = res.ExtraFiles[1:]
			}

			res.ModelFile[quant] = types.ModelFileInfo{
				Name:       res.ModelFile[quant].Name,
				Downloaded: true,
				Size:       res.ModelFile[quant].Size,
			}
			for i, v := range res.ExtraFiles {
				res.ExtraFiles[i] = types.ModelFileInfo{
					Name:       v.Name,
					Downloaded: true,
					Size:       v.Size,
				}
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
			return fmt.Errorf("specified quant %s not found", specifiedQuant)
		} else if fileinfo.Downloaded {
			return fmt.Errorf("specified quant %s already downloaded", specifiedQuant)
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
			Title("Choose a quant version to download").
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

// detectMacOSBundles detects .mlmodelc and .mlpackage bundles from file list
// Returns a list of bundle paths (e.g., "EmbedNeuralVision.mlmodelc")
func detectMacOSBundles(files []model_hub.ModelFileInfo) []string {
	bundleMap := make(map[string]bool)

	for _, file := range files {
		// Case-insensitive check for .mlmodelc/ or .mlpackage/
		lowerName := strings.ToLower(file.Name)
		if idx := strings.Index(lowerName, ".mlmodelc/"); idx != -1 {
			bundlePath := file.Name[:idx+len(".mlmodelc")]
			bundleMap[bundlePath] = true
		} else if idx := strings.Index(lowerName, ".mlpackage/"); idx != -1 {
			bundlePath := file.Name[:idx+len(".mlpackage")]
			bundleMap[bundlePath] = true
		}
	}

	// Convert map to sorted slice for consistent ordering
	bundles := make([]string, 0, len(bundleMap))
	for bundle := range bundleMap {
		bundles = append(bundles, bundle)
	}
	sort.Strings(bundles)

	return bundles
}

// AI Hub pull flow moved to cli/internal/model_hub/model_hub_aihub.go;
// pullModel handles qualcomm/* and qai-hub-models/* uniformly via the
// ModelHub interface and the PostDownload hook (unzip into flat layout).
