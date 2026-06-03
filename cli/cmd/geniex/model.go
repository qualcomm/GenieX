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
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"strconv"
	"strings"

	"github.com/charmbracelet/huh"
	"github.com/dustin/go-humanize"
	"github.com/jedib0t/go-pretty/v6/table"
	"github.com/spf13/cobra"

	geniex_sdk "github.com/qcom-it-nexa-ai/geniex/bindings/go"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub/aihub"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/render"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/store"
)

var (
	modelHub  string
	localPath string
	modelType string
)

// resolveHub maps the --model-hub flag to a HubSource, defaulting to Auto.
func resolveHub() (geniex_sdk.HubSource, error) {
	if localPath != "" && modelHub == "" {
		modelHub = "localfs"
	}
	switch strings.ToLower(modelHub) {
	case "":
		return geniex_sdk.HubAuto, nil
	case "aihub":
		return geniex_sdk.HubAIHub, nil
	case "hf", "huggingface":
		return geniex_sdk.HubHuggingFace, nil
	case "local", "localfs":
		if localPath == "" {
			return 0, fmt.Errorf("local path is required for localfs model hub")
		}
		return geniex_sdk.HubLocalFS, nil
	default:
		return 0, fmt.Errorf("unknown model hub: %s", modelHub)
	}
}

// pull creates a command to download and cache a model by name.
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
		name, quant := geniex_sdk.SplitNameQuant(args[0])
		return pullModel(cmd.Context(), name, quant)
	}

	return pullCmd
}

// remove creates a command to delete cached models.
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

		var errs []error
		for _, arg := range args {
			name, quant := geniex_sdk.SplitNameQuant(arg)
			key := name
			if quant != "" {
				key = name + ":" + quant
			}
			if err := geniex_sdk.ModelRemove(key); err != nil {
				errs = append(errs, fmt.Errorf("remove %s: %w", key, err))
				continue
			}
			fmt.Println(render.GetTheme().Success.Sprintf("✔  Removed %s", key))
		}
		if len(errs) > 0 {
			return errs[0]
		}
		return nil
	}

	return removeCmd
}

// clean creates a command to remove all cached models.
func clean() *cobra.Command {
	cleanCmd := &cobra.Command{
		GroupID: "model",
		Use:     "clean",
		Short:   "remove all cached models",
		Long:    "Remove all cached models and free up storage. This will delete all model files from the local cache.",
	}

	cleanCmd.RunE = func(cmd *cobra.Command, args []string) error {
		c, err := geniex_sdk.ModelClean()
		if err != nil {
			return err
		}
		fmt.Println(render.GetTheme().Success.Sprintf("✔  Removed %d models", c))
		return nil
	}

	return cleanCmd
}

// list creates a command to display all cached models.
func list() *cobra.Command {
	var format string

	listCmd := &cobra.Command{
		GroupID: "model",
		Use:     "list",
		Aliases: []string{"ls"},
		Short:   "List all cached models",
		Long: "Display all cached models.\n" +
			"Use --format json or --format csv for machine-readable output; both have a " +
			"stable schema and --verbose only affects the table view.",
	}

	listCmd.Flags().StringVar(&format, "format", "table", "output format: table|json|csv")

	listCmd.RunE = func(cmd *cobra.Command, args []string) error {
		switch format {
		case "table", "json", "csv":
		default:
			return fmt.Errorf("invalid --format %q (valid: table, json, csv)", format)
		}
		models, err := geniex_sdk.ModelListDetailed()
		if err != nil {
			return err
		}
		switch format {
		case "json":
			return printListJSON(models)
		case "csv":
			return printListCSV(models)
		}
		printListTable(models, verbose)
		return nil
	}

	return listCmd
}

// listedModel is the stable schema for `geniex list --format json|csv`.
type listedModel struct {
	Name       string   `json:"name"`
	Size       int64    `json:"size"`
	Plugin     string   `json:"plugin"`
	Type       string   `json:"type"`
	Precisions []string `json:"precisions"`
}

// downloadedPrecisions returns the model's precisions, optionally hiding the
// QuantNA placeholder used by non-quantized models (table view only).
func downloadedPrecisions(m geniex_sdk.ModelDetail, hideQuantNA bool) []string {
	quants := make([]string, 0, len(m.Precisions))
	for _, q := range m.Precisions {
		if hideQuantNA && q == geniex_sdk.QuantNA {
			continue
		}
		quants = append(quants, q)
	}
	slices.Sort(quants)
	return quants
}

func printListTable(models []geniex_sdk.ModelDetail, verbose bool) {
	tw := table.NewWriter()
	tw.SetOutputMirror(os.Stdout)
	tw.SetStyle(table.StyleLight)
	if verbose {
		tw.AppendHeader(table.Row{"NAME", "SIZE", "PLUGIN", "TYPE", "PRECISIONS"})
	} else {
		tw.AppendHeader(table.Row{"NAME", "SIZE", "PRECISIONS"})
	}
	for _, model := range models {
		size := humanize.IBytes(uint64(model.TotalSize))
		quants := strings.Join(downloadedPrecisions(model, !verbose), ",")
		if verbose {
			tw.AppendRow(table.Row{model.Name, size, model.PluginID, model.ModelType, quants})
		} else {
			tw.AppendRow(table.Row{model.Name, size, quants})
		}
	}
	tw.Render()
}

func toListedModels(models []geniex_sdk.ModelDetail) []listedModel {
	out := make([]listedModel, 0, len(models))
	for _, m := range models {
		out = append(out, listedModel{
			Name:       m.Name,
			Size:       m.TotalSize,
			Plugin:     m.PluginID,
			Type:       m.ModelType.String(),
			Precisions: downloadedPrecisions(m, false),
		})
	}
	return out
}

func printListJSON(models []geniex_sdk.ModelDetail) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(toListedModels(models))
}

func printListCSV(models []geniex_sdk.ModelDetail) error {
	w := csv.NewWriter(os.Stdout)
	if err := w.Write([]string{"name", "size", "plugin", "type", "precisions"}); err != nil {
		return err
	}
	for _, m := range toListedModels(models) {
		row := []string{
			m.Name,
			strconv.FormatInt(m.Size, 10),
			m.Plugin,
			m.Type,
			strings.Join(m.Precisions, ","),
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

// modelCmd builds the `geniex model` command tree.
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

// modelTypeNames are the accepted --model-type / set-type values.
var modelTypeNames = []string{
	geniex_sdk.ModelTypeLLM.String(),
	geniex_sdk.ModelTypeVLM.String(),
}

// setTypeCmd builds the `geniex model set-type` subcommand.
func setTypeCmd() *cobra.Command {
	return &cobra.Command{
		Use:       "set-type <model-name> [llm|vlm]",
		Short:     "Override the model type for a cached model",
		Long:      "Update the model type stored in a cached model's manifest.\n\nOmit the type argument to choose interactively.",
		Args:      cobra.RangeArgs(1, 2),
		ValidArgs: modelTypeNames,
		RunE: func(cmd *cobra.Command, args []string) error {
			name, _ := geniex_sdk.SplitNameQuant(args[0])

			// Verify the model is present before prompting for a type.
			if _, err := geniex_sdk.ModelGetType(name); err != nil {
				return fmt.Errorf("model %q not found: %w", name, err)
			}

			var mt geniex_sdk.ModelType
			if len(args) == 2 {
				parsed, ok := geniex_sdk.ParseModelType(args[1])
				if !ok {
					return fmt.Errorf("unknown model type %q (valid: %s)", args[1], strings.Join(modelTypeNames, ", "))
				}
				mt = parsed
			} else {
				var choice string
				if err := huh.NewSelect[string]().
					Title("Choose Model Type").
					Options(huh.NewOptions(modelTypeNames...)...).
					Value(&choice).
					Run(); err != nil {
					return err
				}
				mt, _ = geniex_sdk.ParseModelType(choice)
			}

			if err := geniex_sdk.ModelSetType(name, mt); err != nil {
				return fmt.Errorf("failed to update model type: %w", err)
			}
			fmt.Println(render.GetTheme().Success.Sprintf("✔  %s → %s", name, mt))
			return nil
		},
	}
}

func pullModel(ctx context.Context, name string, quant string) error {
	slog.Debug("pullModel", "name", name, "quant", quant)

	hub, err := resolveHub()
	if err != nil {
		return err
	}

	// AI Hub pulls need a chipset; resolve it before the spinner since the
	// picker can't share the terminal with one.
	var chipset string
	if _, ok := aihub.IsAIHubName(name); ok || hub == geniex_sdk.HubAIHub {
		s := store.Get()
		if err := chipsetEnsure(ctx, s); err != nil {
			return err
		}
		chipset = chipsetGet(s)
	}

	in := geniex_sdk.ModelPullInput{
		ModelName:   name,
		Quant:       quant,
		Hub:         hub,
		LocalPath:   localPath,
		Chipset:     chipset,
		DisplayName: "",
	}

	// Validate --model-type early so we fail before downloading anything, and
	// let the pull write it into the manifest in one shot (no set-type round-trip).
	if modelType != "" {
		mt, ok := geniex_sdk.ParseModelType(modelType)
		if !ok {
			return fmt.Errorf("unknown model type %q (valid: %s)", modelType, strings.Join(modelTypeNames, ", "))
		}
		in.ModelType = &mt
	}

	// No precision requested: query the remote candidates and let the user
	// pick (skipped for localfs, which has no remote listing).
	if quant == "" && hub != geniex_sdk.HubLocalFS {
		spin := render.NewSpinner("fetching available precisions from: " + name)
		spin.Start()
		q, err := geniex_sdk.ModelQuery(in)
		spin.Stop()
		if err != nil {
			return err
		}
		if chosen, err := choosePrecision(q.Candidates); err != nil {
			return err
		} else {
			in.Quant = chosen
			quant = chosen
		}
	}

	// Download with a progress bar driven by the SDK callback.
	var bar *render.ProgressBar
	in.OnProgress = func(files []geniex_sdk.FileProgress) bool {
		var downloaded, total int64
		for _, f := range files {
			downloaded += f.DownloadedBytes
			if f.TotalBytes > 0 {
				total += f.TotalBytes
			}
		}
		if bar == nil {
			bar = render.NewProgressBar(total, "downloading")
		}
		bar.Set(downloaded)
		return true
	}

	if err := geniex_sdk.ModelPull(in); err != nil {
		if bar != nil {
			bar.Clear()
		}
		return err
	}
	if bar != nil {
		bar.Exit()
	}

	if t, err := geniex_sdk.ModelGetType(name); err == nil {
		fmt.Println(render.GetTheme().Info.Sprintf("   Detected model type: %s", t))
	} else {
		fmt.Println(render.GetTheme().Warning.Sprintf(
			"⚠  Could not detect model type; run:\n"+
				"     geniex model set-type %s <llm|vlm>", name))
	}

	fmt.Println(render.GetTheme().Success.Sprint("✔  Download success"))
	return nil
}

// choosePrecision picks a precision from the remote candidates: the only one
// when there's a single option, otherwise an interactive picker that defaults
// to the SDK-recommended quant.
func choosePrecision(candidates []geniex_sdk.QuantCandidate) (string, error) {
	if len(candidates) == 0 {
		return "", fmt.Errorf("no precision available for this model")
	}
	if len(candidates) == 1 {
		return candidates[0].Quant, nil
	}

	var defaultQuant string
	var options []huh.Option[string]
	for _, c := range candidates {
		label := fmt.Sprintf("%-10s [%7s]", c.Quant, humanize.IBytes(uint64(c.Size)))
		if c.IsDefault {
			label += " (default)"
			defaultQuant = c.Quant
		}
		options = append(options, huh.NewOption(label, c.Quant))
	}

	chosen := defaultQuant
	if err := huh.NewSelect[string]().
		Title("Choose a precision version to download").
		Options(options...).
		Value(&chosen).
		Run(); err != nil {
		return "", err
	}
	return chosen, nil
}
