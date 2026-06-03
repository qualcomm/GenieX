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
	"encoding/csv"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	geniex_sdk "github.com/qcom-it-nexa-ai/geniex/bindings/go"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/testutil"
)

var sampleListModels = []geniex_sdk.ModelDetail{
	{
		Name:       "acme/llama",
		ModelType:  geniex_sdk.ModelTypeLLM,
		PluginID:   "llama_cpp",
		TotalSize:  3072,
		Precisions: []string{"Q4_0", "Q8_0"},
	},
	{
		Name:       "acme/yolo",
		ModelType:  geniex_sdk.ModelTypeLLM,
		PluginID:   "qairt",
		TotalSize:  512,
		Precisions: []string{geniex_sdk.QuantNA},
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
	// Non-verbose hides PLUGIN/TYPE columns and the QuantNA precision.
	if strings.Contains(out, "PLUGIN") || strings.Contains(out, geniex_sdk.QuantNA) {
		t.Errorf("non-verbose table leaked verbose-only fields:\n%s", out)
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
	if got[0].Name != "acme/llama" || got[0].Plugin != "llama_cpp" || got[0].Type != "llm" {
		t.Errorf("got[0] = %+v", got[0])
	}
	if got[0].Size != 3072 {
		t.Errorf("got[0].Size = %d, want 3072", got[0].Size)
	}
	if want := []string{"Q4_0", "Q8_0"}; !slices.Equal(got[0].Precisions, want) {
		t.Errorf("got[0].Precisions = %v, want %v", got[0].Precisions, want)
	}
	// JSON keeps the full inventory regardless of --verbose, so QuantNA is exposed.
	if want := []string{geniex_sdk.QuantNA}; !slices.Equal(got[1].Precisions, want) {
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
		{"name", "size", "plugin", "type", "precisions"},
		{"acme/llama", "3072", "llama_cpp", "llm", "Q4_0,Q8_0"},
		{"acme/yolo", "512", "qairt", "llm", geniex_sdk.QuantNA},
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
