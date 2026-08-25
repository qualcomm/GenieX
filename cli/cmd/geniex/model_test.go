// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import (
	"encoding/csv"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/internal/testutil"
)

var sampleListModels = []geniex_sdk.ModelDetail{
	{
		Name:       "acme/llama",
		ModelType:  geniex_sdk.ModelTypeLLM,
		RuntimeID:  "llama_cpp",
		TotalSize:  3072,
		Precisions: []string{"Q4_0", "Q8_0"},
	},
	{
		Name:       "acme/yolo",
		ModelType:  geniex_sdk.ModelTypeLLM,
		RuntimeID:  "qairt",
		TotalSize:  512,
		Precisions: []string{geniex_sdk.PrecisionNA},
	},
}

func TestPrintListTable(t *testing.T) {
	out, _, _ := testutil.CaptureOutput(t, func() error {
		printListTable(sampleListModels, false)
		return nil
	})
	for _, want := range []string{"NAME", "SIZE", "PRECISIONS", "acme/llama", "Q4_0,Q8_0", "acme/yolo"} {
		if !strings.Contains(out, want) {
			t.Errorf("table output missing %q:\n%s", want, out)
		}
	}
	// Non-verbose hides RUNTIME/TYPE columns and the PrecisionNA placeholder.
	if strings.Contains(out, "RUNTIME") || strings.Contains(out, geniex_sdk.PrecisionNA) {
		t.Errorf("non-verbose table leaked verbose-only fields:\n%s", out)
	}
}

func TestPrintListTableVerbose(t *testing.T) {
	out, _, _ := testutil.CaptureOutput(t, func() error {
		printListTable(sampleListModels, true)
		return nil
	})
	for _, want := range []string{"NAME", "SIZE", "RUNTIME", "TYPE", "PRECISIONS", "llama_cpp"} {
		if !strings.Contains(out, want) {
			t.Errorf("verbose table output missing %q:\n%s", want, out)
		}
	}
}

func TestPrintListJSON(t *testing.T) {
	raw, _, err := testutil.CaptureOutput(t, func() error {
		return printListJSON(sampleListModels)
	})
	if err != nil {
		t.Fatalf("printListJSON: %v", err)
	}
	var got []listedModel
	if err := json.Unmarshal([]byte(raw), &got); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, raw)
	}
	if len(got) != 2 {
		t.Fatalf("len(got) = %d, want 2", len(got))
	}
	if got[0].Name != "acme/llama" || got[0].Runtime != "llama_cpp" || got[0].Type != "llm" {
		t.Errorf("got[0] = %+v", got[0])
	}
	if got[0].Size != 3072 {
		t.Errorf("got[0].Size = %d, want 3072", got[0].Size)
	}
	if want := []string{"Q4_0", "Q8_0"}; !slices.Equal(got[0].Precisions, want) {
		t.Errorf("got[0].Precisions = %v, want %v", got[0].Precisions, want)
	}
	// JSON keeps the full inventory regardless of --verbose, so PrecisionNA is exposed.
	if want := []string{geniex_sdk.PrecisionNA}; !slices.Equal(got[1].Precisions, want) {
		t.Errorf("got[1].Precisions = %v, want %v", got[1].Precisions, want)
	}
}

func TestPrintListCSV(t *testing.T) {
	raw, _, err := testutil.CaptureOutput(t, func() error {
		return printListCSV(sampleListModels)
	})
	if err != nil {
		t.Fatalf("printListCSV: %v", err)
	}
	rows, err := csv.NewReader(strings.NewReader(raw)).ReadAll()
	if err != nil {
		t.Fatalf("csv parse: %v\n%s", err, raw)
	}
	want := [][]string{
		{"name", "size", "runtime", "type", "precisions"},
		{"acme/llama", "3072", "llama_cpp", "llm", "Q4_0,Q8_0"},
		{"acme/yolo", "512", "qairt", "llm", geniex_sdk.PrecisionNA},
	}
	if len(rows) != len(want) {
		t.Fatalf("rows = %d, want %d:\n%s", len(rows), len(want), raw)
	}
	for i := range want {
		if !slices.Equal(rows[i], want[i]) {
			t.Errorf("row %d = %v, want %v", i, rows[i], want[i])
		}
	}
}

func TestSkipDownloaded(t *testing.T) {
	candidates := []geniex_sdk.PrecisionCandidate{
		{Precision: "Q4_0", Size: 100},
		{Precision: "Q4_K_M", Size: 200},
		{Precision: "Q8_0", Size: 300},
	}
	tests := []struct {
		name   string
		cached []string
		want   []string
	}{
		{"nothing cached keeps every candidate", nil, []string{"Q4_0", "Q4_K_M", "Q8_0"}},
		{"cached precision is dropped", []string{"Q4_0"}, []string{"Q4_K_M", "Q8_0"}},
		{"matching is case-insensitive", []string{"q4_k_m"}, []string{"Q4_0", "Q8_0"}},
		{"all cached leaves nothing to pull", []string{"Q8_0", "Q4_0", "Q4_K_M"}, nil},
		{"a cached precision the remote lost is ignored", []string{"Q2_K"}, []string{"Q4_0", "Q4_K_M", "Q8_0"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := skipDownloaded(candidates, tt.cached)
			var names []string
			for _, c := range got {
				names = append(names, c.Precision)
			}
			if !slices.Equal(names, tt.want) {
				t.Errorf("skipDownloaded(%v) = %v, want %v", tt.cached, names, tt.want)
			}
			if len(candidates) != 3 {
				t.Errorf("input mutated: %v", candidates)
			}
		})
	}
}
