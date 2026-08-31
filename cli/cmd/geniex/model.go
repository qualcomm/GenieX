// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/charmbracelet/huh"
	"github.com/dustin/go-humanize"
	"github.com/jedib0t/go-pretty/v6/table"
	"github.com/spf13/cobra"
	"golang.org/x/term"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/internal/render"
	"github.com/qualcomm/GenieX/cli/internal/store"
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
	case "docker", "dockerhub":
		return geniex_sdk.HubDocker, nil
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

		Short: "Pull model from HuggingFace, Qualcomm AI Hub Models, or Docker Hub",
		Long: "Download and cache a model by name. Append ':<precision>' to pull a specific precision; otherwise you'll be prompted to choose one.\n\n" +
			"Docker Hub models (e.g. docker.io/ai/gemma3, or ai/gemma3 with --model-hub docker) " +
			"use ':<tag>' instead of a precision — omit it to pull the 'latest' tag.",
	}

	pullCmd.Args = cobra.MatchAll(cobra.ExactArgs(1), cobra.OnlyValidArgs)

	pullCmd.Flags().SortFlags = false
	pullCmd.Flags().StringVarP(&modelHub, "model-hub", "", "", "specify model hub to use: aihub|hf|docker|localfs")
	pullCmd.Flags().StringVarP(&localPath, "local-path", "", "", "[localfs] path to local directory or aihub zip file")
	pullCmd.Flags().StringVarP(&modelType, "model-type", "", "", "specify model type to use: [llm|vlm]")

	pullCmd.RunE = func(cmd *cobra.Command, args []string) error {
		name, quant := geniex_sdk.SplitNamePrecision(args[0])
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
			name, quant := geniex_sdk.SplitNamePrecision(arg)
			key := geniex_sdk.JoinNamePrecision(name, quant)
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
	var assumeYes bool

	cleanCmd := &cobra.Command{
		GroupID: "model",
		Use:     "clean",
		Short:   "remove all cached models",
		Long:    "Remove all cached models and free up storage. This will delete all model files from the local cache.",
	}

	cleanCmd.Flags().BoolVarP(&assumeYes, "yes", "y", false, "Skip the confirmation prompt")

	cleanCmd.RunE = func(cmd *cobra.Command, args []string) error {
		if !assumeYes {
			var ok bool
			if err := huh.NewConfirm().Title("Are you sure you want to delete all cached models?").Value(&ok).Run(); err != nil {
				return err
			}
			if !ok {
				fmt.Println(render.GetTheme().Info.Sprint("Aborted"))
				return nil
			}
		}

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
		fmt.Println(render.GetTheme().Info.Sprintf("Models cached in %s", filepath.Join(store.Get().DataPath(), "models")))
		fmt.Println()
		printListTable(models, verbose)
		return nil
	}

	return listCmd
}

// listedModel is the stable schema for `geniex list --format json|csv`.
type listedModel struct {
	Name       string   `json:"name"`
	Size       int64    `json:"size"`
	Runtime    string   `json:"runtime"`
	Type       string   `json:"type"`
	Precisions []string `json:"precisions"`
}

// downloadedPrecisions returns the model's precisions in the SDK's order — its
// head is a bare name's pick — optionally hiding PrecisionNA (table view only).
func downloadedPrecisions(m geniex_sdk.ModelDetail, hidePrecisionNA bool) []string {
	quants := make([]string, 0, len(m.Precisions))
	for _, q := range m.Precisions {
		if hidePrecisionNA && q == geniex_sdk.PrecisionNA {
			continue
		}
		quants = append(quants, q)
	}
	return quants
}

func printListTable(models []geniex_sdk.ModelDetail, verbose bool) {
	tw := table.NewWriter()
	tw.SetOutputMirror(os.Stdout)
	tw.SetStyle(table.StyleLight)
	if verbose {
		tw.AppendHeader(table.Row{"NAME", "SIZE", "RUNTIME", "TYPE", "PRECISIONS"})
	} else {
		tw.AppendHeader(table.Row{"NAME", "SIZE", "PRECISIONS"})
	}
	for _, model := range models {
		var size string
		if model.TotalSize > 0 {
			size = humanize.IBytes(uint64(model.TotalSize))
		} else {
			size = "—"
		}
		quants := strings.Join(downloadedPrecisions(model, !verbose), ",")
		if verbose {
			tw.AppendRow(table.Row{model.Name, size, model.RuntimeID, model.ModelType, quants})
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
			Runtime:    m.RuntimeID,
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
	if err := w.Write([]string{"name", "size", "runtime", "type", "precisions"}); err != nil {
		return err
	}
	for _, m := range toListedModels(models) {
		row := []string{
			m.Name,
			strconv.FormatInt(m.Size, 10),
			m.Runtime,
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
	cmd.AddCommand(setTypeCmd(), listHubCmd())
	return cmd
}

func printHubTable(models []geniex_sdk.HubModel, showChipsets bool) {
	tw := table.NewWriter()
	tw.SetOutputMirror(os.Stdout)
	tw.SetStyle(table.StyleLight)
	if showChipsets {
		tw.AppendHeader(table.Row{"NAME", "TYPE", "CHIPSETS"})
	} else {
		tw.AppendHeader(table.Row{"NAME", "TYPE"})
	}
	for _, m := range models {
		if showChipsets {
			chips := make([]string, len(m.Chipsets))
			for i, c := range m.Chipsets {
				chips[i] = strings.TrimPrefix(strings.TrimPrefix(c, "qualcomm-"), "snapdragon-")
			}
			tw.AppendRow(table.Row{m.Name, m.ModelType, strings.Join(chips, ", ")})
		} else {
			tw.AppendRow(table.Row{m.Name, m.ModelType})
		}
	}
	tw.Render()
}

func listHubCmd() *cobra.Command {
	var all bool
	cmd := &cobra.Command{
		Use:   "list",
		Short: "List Qualcomm AI Hub models geniex can run",
		Long: "List Qualcomm AI Hub models with a qairt (NPU) build.\n\n" +
			"By default only models compatible with the current device are shown; " +
			"pass --all to list every model. Names are ready to pull, e.g. " +
			"'geniex pull qualcomm/Qwen3-4B'.",
		RunE: func(cmd *cobra.Command, args []string) error {
			// "" lists every model; --all skips filtering entirely.
			var chipset string
			if !all {
				c, err := ensureChipset()
				if err != nil {
					return err
				}
				chipset = c
			}
			models, err := geniex_sdk.ModelListHub(chipset)
			if err != nil {
				return err
			}
			if all {
				fmt.Println(render.GetTheme().Info.Sprint("Qualcomm AI Hub models geniex can run:"))
			} else {
				fmt.Println(render.GetTheme().Info.Sprintf("Qualcomm AI Hub models for %s (use --all to see every model):", chipset))
			}
			fmt.Println()
			// CHIPSETS column only with --all; filtered rows all share one chipset.
			printHubTable(models, all)
			return nil
		},
	}
	cmd.Flags().BoolVar(&all, "all", false, "list every model, not just ones compatible with this device")
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
			name, _ := geniex_sdk.SplitNamePrecision(args[0])

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

func pullModel(ctx context.Context, name, quant string) error {
	slog.Debug("pullModel", "name", name, "quant", quant)

	hub, err := resolveHub()
	if err != nil {
		return err
	}
	// resolveHub() only sees --model-hub, so a docker.io/... prefix still reads as
	// HubAuto; the guard below needs the hub the pull will actually use.
	effectiveHub, err := geniex_sdk.ResolveHub(name, hub)
	if err != nil {
		return err
	}

	in := geniex_sdk.ModelPullInput{
		ModelName:   name,
		Precision:   quant,
		Hub:         hub,
		LocalPath:   localPath,
		DisplayName: "",
	}

	// Resolve before the spinner — the picker can't share the terminal with one.
	if in.Chipset, err = ensureChipset(); err != nil {
		return err
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

	// No precision requested: query the remote candidates and let the user pick.
	// Skipped for localfs (no remote listing) and Docker, where an empty quant
	// already means the `latest` tag and a query would feed back a quant label.
	if quant == "" && effectiveHub != geniex_sdk.HubLocalFS && effectiveHub != geniex_sdk.HubDocker {
		spin := render.NewSpinner("fetching available precisions from: " + name)
		spin.Start()
		q, err := geniex_sdk.ModelQuery(in)
		spin.Stop()
		if err != nil {
			return err
		}
		// Only the picker hides cached precisions: filtering would make a repeated
		// `geniex pull <model>` walk the list instead of re-resolving to the head.
		candidates := q.Candidates
		if hasTerminal() {
			// N/A stays in: qairt stores a cached bundle under it.
			var cached []string
			if m, err := geniex_sdk.ModelGetDetailed(name); err == nil {
				cached = downloadedPrecisions(*m, false)
			}
			if pending := skipDownloaded(candidates, cached); len(pending) > 0 {
				candidates = pending
			} else if len(cached) > 0 {
				// A re-pull still repairs a truncated file or refetches a
				// requantized or per-chipset bundle (the store is keyed by name).
				fmt.Println(render.GetTheme().Info.Sprint("Every precision is already downloaded; pick one to re-download."))
			}
			slog.Debug("pull precisions", "remote", len(q.Candidates), "cached", cached, "offered", len(candidates))
		}

		if chosen, err := choosePrecision("Choose a precision version to download", candidates); err != nil {
			return err
		} else {
			in.Precision = chosen
			quant = chosen
		}
	}

	ctx, stop := signal.NotifyContext(ctx, os.Interrupt)
	defer stop()

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
			bar = render.NewProgressBar(total, downloaded, "downloading")
			fmt.Println(render.GetTheme().Info.Sprint("   Press Ctrl+C to cancel — progress is saved, not discarded."))
		}
		bar.Set(downloaded)
		return ctx.Err() == nil
	}

	if err := geniex_sdk.ModelPull(in); err != nil {
		if bar != nil {
			bar.Clear()
		}
		if ctx.Err() != nil {
			key := geniex_sdk.JoinNamePrecision(name, quant)
			fmt.Println(render.GetTheme().Warning.Sprint("✗  Download cancelled"))
			fmt.Println(render.GetTheme().Info.Sprintf("   Run 'geniex pull %s' to resume, or 'geniex remove %s' to free the disk space instead.", key, key))
			return nil
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

	key := geniex_sdk.JoinNamePrecision(name, quant)
	if m, err := geniex_sdk.ModelGetDetailed(name); err == nil && m.TotalSize > 0 {
		fmt.Println(render.GetTheme().Info.Sprintf("   Size:      %s", humanize.IBytes(uint64(m.TotalSize))))
	}
	if paths, err := geniex_sdk.ModelGetPaths(key); err == nil && paths.ModelPath != "" {
		fmt.Println(render.GetTheme().Info.Sprintf("   Location:  %s", filepath.Dir(paths.ModelPath)))
	}
	if quant != "" {
		fmt.Println(render.GetTheme().Info.Sprintf("   Precision: %s", quant))
	}
	return nil
}

func skipDownloaded(candidates []geniex_sdk.PrecisionCandidate, cached []string) []geniex_sdk.PrecisionCandidate {
	if len(cached) == 0 {
		return candidates
	}
	return slices.DeleteFunc(slices.Clone(candidates), func(c geniex_sdk.PrecisionCandidate) bool {
		// GGUF quant labels match case-insensitively, as the SDK does.
		return slices.ContainsFunc(cached, func(p string) bool { return strings.EqualFold(p, c.Precision) })
	})
}

// hasTerminal reports whether a picker can be drawn: keys come from stdin, and
// huh draws on stderr, so both have to be a terminal.
func hasTerminal() bool {
	return term.IsTerminal(int(os.Stdin.Fd())) && term.IsTerminal(int(os.Stderr.Fd()))
}

// choosePrecision picks from candidates, whose head must be the recommended one:
// it wins outright when alone or without a terminal, else the picker preselects it.
func choosePrecision(title string, candidates []geniex_sdk.PrecisionCandidate) (string, error) {
	if len(candidates) == 0 {
		return "", fmt.Errorf("no precision available for this model")
	}
	if len(candidates) == 1 || !hasTerminal() {
		return candidates[0].Precision, nil
	}

	// Sizes come from the remote query; a local pick has none, so drop the
	// column rather than render a row of placeholders.
	withSize := slices.ContainsFunc(candidates, func(c geniex_sdk.PrecisionCandidate) bool {
		return c.Size > 0
	})
	options := make([]huh.Option[string], 0, len(candidates))
	for _, c := range candidates {
		label := c.Precision
		if withSize {
			sz := "—"
			if c.Size > 0 {
				sz = humanize.IBytes(uint64(c.Size))
			}
			label = fmt.Sprintf("%-10s [%7s]", c.Precision, sz)
		}
		options = append(options, huh.NewOption(label, c.Precision))
	}

	chosen := candidates[0].Precision
	if err := huh.NewSelect[string]().
		Title(title).
		Options(options...).
		Value(&chosen).
		Run(); err != nil {
		return "", err
	}
	return chosen, nil
}
