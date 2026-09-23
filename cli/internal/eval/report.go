// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package eval

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"text/tabwriter"
)

type Result struct {
	TaskID   string `json:"task_id"`
	Category string `json:"category,omitempty"`
	Correct  bool   `json:"correct"`
	Output   string `json:"output"`
	Error    string `json:"error,omitempty"`
}

type ModelReport struct {
	Model        string   `json:"model"`
	Results      []Result `json:"results"`
	DecodeTokens int64    `json:"decode_tokens"`
	DecodeTimeUs int64    `json:"decode_time_us"`
}

func (r *ModelReport) Correct() int {
	n := 0
	for _, res := range r.Results {
		if res.Correct {
			n++
		}
	}
	return n
}

func (r *ModelReport) Accuracy() float64 {
	if len(r.Results) == 0 {
		return 0
	}
	return float64(r.Correct()) / float64(len(r.Results))
}

func (r *ModelReport) TokPerSec() float64 {
	if r.DecodeTimeUs <= 0 {
		return 0
	}
	return float64(r.DecodeTokens) / (float64(r.DecodeTimeUs) / 1e6)
}

// RenderTable renders the cross-model comparison as an aligned text table.
func RenderTable(pack *Pack, reports []*ModelReport) string {
	seen := map[string]bool{}
	var cats []string
	for _, r := range reports {
		for _, res := range r.Results {
			if res.Category != "" && !seen[res.Category] {
				seen[res.Category] = true
				cats = append(cats, res.Category)
			}
		}
	}
	sort.Strings(cats)

	var b strings.Builder
	fmt.Fprintf(&b, "Eval: %s (%d tasks)\n\n", pack.Name, len(pack.Tasks))
	w := tabwriter.NewWriter(&b, 2, 0, 3, ' ', 0)
	header := "MODEL\tACCURACY\tCORRECT\tTOK/S"
	if len(cats) > 0 {
		header += "\t" + strings.ToUpper(strings.Join(cats, "\t"))
	}
	fmt.Fprintln(w, header)

	for _, r := range reports {
		row := fmt.Sprintf("%s\t%.1f%%\t%d/%d", r.Model, r.Accuracy()*100, r.Correct(), len(r.Results))
		if r.TokPerSec() > 0 {
			row += fmt.Sprintf("\t%.1f", r.TokPerSec())
		} else {
			row += "\t-"
		}
		for _, c := range cats {
			correct, total := 0, 0
			for _, res := range r.Results {
				if res.Category == c {
					total++
					if res.Correct {
						correct++
					}
				}
			}
			row += fmt.Sprintf("\t%d/%d", correct, total)
		}
		fmt.Fprintln(w, row)
	}
	w.Flush()
	return b.String()
}

// WriteJSON writes the full run to path for diffing or feeding other tools.
func WriteJSON(path string, pack *Pack, reports []*ModelReport) error {
	doc := struct {
		Eval        string         `json:"eval"`
		Description string         `json:"description,omitempty"`
		TaskCount   int            `json:"task_count"`
		Models      []*ModelReport `json:"models"`
	}{pack.Name, pack.Description, len(pack.Tasks), reports}
	data, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}
