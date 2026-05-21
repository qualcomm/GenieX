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
	"regexp"
	"strings"
	"unicode"

	"github.com/spf13/cobra"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/render"
)

var (
	flagPattern        = regexp.MustCompile(`(^\s+)(-[a-zA-Z],\s+--[a-zA-Z0-9-]+|--[a-zA-Z0-9-]+|-[a-zA-Z])(\s+(stringArray|stringSlice|bytesBase64|bytesHex|float64|float32|duration|ipMask|uint16|uint32|uint64|int16|int32|int64|string|float|count|ipNet|uint8|uint|int8|int|bool|ip))?`)
	parenHintPattern   = regexp.MustCompile(`\([^()]*\)`)
	placeholderPattern = regexp.MustCompile(`<[^<>]+>|\[[^\[\]]*\]`)
)

func colorPlaceholders(s string) string {
	t := render.GetTheme()
	return placeholderPattern.ReplaceAllStringFunc(s, func(m string) string {
		switch m {
		case "[flags]":
			return t.Flag.Sprint(m)
		case "[command]":
			return t.Command.Sprint(m)
		default:
			return t.Reference.Sprint(m)
		}
	})
}

func colorFlagUsages(usages string) string {
	t := render.GetTheme()
	lines := strings.Split(usages, "\n")
	for i, line := range lines {
		line = flagPattern.ReplaceAllStringFunc(line, func(m string) string {
			sub := flagPattern.FindStringSubmatch(m)
			out := sub[1] + t.Flag.Sprint(sub[2])
			if sub[4] != "" {
				out += " " + t.Reference.Sprint(sub[4])
			}
			return out
		})
		line = parenHintPattern.ReplaceAllStringFunc(line, func(m string) string {
			return t.Reference.Sprint(m)
		})
		lines[i] = line
	}
	return strings.Join(lines, "\n")
}

func colorUseLine(c *cobra.Command) string {
	line, path := c.UseLine(), c.CommandPath()
	if strings.HasPrefix(line, path) {
		return render.GetTheme().Command.Sprint(path) + colorPlaceholders(line[len(path):])
	}
	return colorPlaceholders(line)
}

func colorNameAndAliases(c *cobra.Command) string {
	t := render.GetTheme()
	names := append([]string{c.Name()}, c.Aliases...)
	for i, n := range names {
		names[i] = t.Command.Sprint(n)
	}
	return strings.Join(names, ", ")
}

func flagGroupedUsage(c *cobra.Command) error {
	w := c.OutOrStdout()
	h := render.GetTheme().Heading
	fmt.Fprint(w, h.Sprint("Usage:"))
	if c.Runnable() {
		fmt.Fprintf(w, "\n  %s", colorUseLine(c))
	}
	if len(c.Aliases) > 0 {
		fmt.Fprintf(w, "\n\n%s\n", h.Sprint("Aliases:"))
		fmt.Fprintf(w, "  %s", colorNameAndAliases(c))
	}

	for _, flags := range flagGroups {
		fmt.Fprintf(w, "\n\n%s\n", h.Sprintf("%s Flags:", flags.Name()))
		fmt.Fprint(w, colorFlagUsages(strings.TrimRightFunc(flags.FlagUsages(), unicode.IsSpace)))
	}

	if c.HasAvailableInheritedFlags() {
		fmt.Fprintf(w, "\n\n%s\n", h.Sprint("Global Flags:"))
		fmt.Fprint(w, colorFlagUsages(strings.TrimRightFunc(c.InheritedFlags().FlagUsages(), unicode.IsSpace)))
	}
	fmt.Fprintln(w)
	return nil
}

func applyHelpStyle(cmd *cobra.Command) {
	t := render.GetTheme()
	cobra.AddTemplateFunc("heading", func(s string) string { return t.Heading.Sprint(s) })
	cobra.AddTemplateFunc("colorFlags", colorFlagUsages)
	cobra.AddTemplateFunc("colorCmd", func(s string) string { return t.Command.Sprint(s) })
	cobra.AddTemplateFunc("colorPlaceholder", colorPlaceholders)
	cobra.AddTemplateFunc("colorUseLine", colorUseLine)
	cobra.AddTemplateFunc("colorAliases", colorNameAndAliases)

	cmd.SetUsageTemplate(usageTemplate)
	cmd.SetHelpTemplate(helpTemplate)
}

const usageTemplate = `{{"Usage:" | heading}}{{if .Runnable}}
  {{. | colorUseLine}}{{end}}{{if .HasAvailableSubCommands}}
  {{.CommandPath | colorCmd}} {{"[command]" | colorPlaceholder}}{{end}}{{if gt (len .Aliases) 0}}

{{"Aliases:" | heading}}
  {{. | colorAliases}}{{end}}{{if .HasExample}}

{{"Examples:" | heading}}
{{.Example}}{{end}}{{if .HasAvailableSubCommands}}{{$cmds := .Commands}}{{if eq (len .Groups) 0}}

{{"Available Commands:" | heading}}{{range $cmds}}{{if (or .IsAvailableCommand (eq .Name "help"))}}
  {{rpad .Name .NamePadding | colorCmd}} {{.Short}}{{end}}{{end}}{{else}}{{range $group := .Groups}}

{{.Title | heading}}{{range $cmds}}{{if (and (eq .GroupID $group.ID) (or .IsAvailableCommand (eq .Name "help")))}}
  {{rpad .Name .NamePadding | colorCmd}} {{.Short}}{{end}}{{end}}{{end}}{{if not .AllChildCommandsHaveGroup}}

{{"Additional Commands:" | heading}}{{range $cmds}}{{if (and (eq .GroupID "") (or .IsAvailableCommand (eq .Name "help")))}}
  {{rpad .Name .NamePadding | colorCmd}} {{.Short}}{{end}}{{end}}{{end}}{{end}}{{end}}{{if .HasAvailableLocalFlags}}

{{"Flags:" | heading}}
{{.LocalFlags.FlagUsages | colorFlags | trimTrailingWhitespaces}}{{end}}{{if .HasAvailableInheritedFlags}}

{{"Global Flags:" | heading}}
{{.InheritedFlags.FlagUsages | colorFlags | trimTrailingWhitespaces}}{{end}}{{if .HasHelpSubCommands}}

{{"Additional help topics:" | heading}}{{range .Commands}}{{if .IsAdditionalHelpTopicCommand}}
  {{rpad .CommandPath .CommandPathPadding | colorCmd}} {{.Short}}{{end}}{{end}}{{end}}{{if .HasAvailableSubCommands}}

Use "{{.CommandPath}} [command] --help" for more information about a command.{{end}}
`

const helpTemplate = `{{with (or .Long .Short)}}{{. | trimTrailingWhitespaces}}

{{end}}{{if or .Runnable .HasSubCommands}}{{.UsageString}}{{end}}`
