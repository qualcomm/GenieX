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

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/testutil"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/types"
)

func TestQuantRegix_MatchAllQuantLevels(t *testing.T) {
	// Test all quantization levels based on the provided list
	quantLevels := []string{
		// Uppercase
		"FP32", "FP16", "FP64",
		"F64", "F32", "F16",
		"I64", "I32", "I16", "I8",
		"BF16",
		"Q8_0", "Q8_1", "Q8_K", "Q6_K", "Q5_0", "Q5_1", "Q5_K", "Q4_0", "Q4_1", "Q4_K", "Q3_K", "Q2_K",
		"IQ4_NL", "IQ4_XS", "IQ3_S", "IQ3_XXS", "IQ2_XXS", "IQ2_S", "IQ2_XS", "IQ1_S", "IQ1_M",
		"1bit", "2bit", "3bit", "4bit", "16bit",

		// Lowercase versions
		"fp32", "fp16", "fp64",
		"f64", "f32", "f16",
		"i64", "i32", "i16", "i8",
		"bf16",
		"q8_0", "q8_1", "q8_k", "q6_k", "q5_0", "q5_1", "q5_k", "q4_0", "q4_1", "q4_k", "q3_k", "q2_k",
		"iq4_nl", "iq4_xs", "iq3_s", "iq3_xxs", "iq2_xxs", "iq2_s", "iq2_xs", "iq1_s", "iq1_m",
		"1bit", "2bit", "3bit", "4bit", "16bit",

		// Mixed case versions
		"Fp32", "fP16", "Fp64",
		"F64", "f32", "F16",
		"I64", "i32", "I16", "I8",
		"Bf16", "bF16",
		"Q8_0", "q8_1", "Q8_k", "q6_K", "Q5_0", "q5_1", "Q5_k", "Q4_0", "q4_1", "Q4_k", "q3_K", "Q2_k",
		"Iq4_nl", "iQ4_xs", "Iq3_s", "iQ3_xxs", "Iq2_xxs", "iQ2_s", "Iq2_xs", "iQ1_s", "Iq1_m",
		"1BIT", "2BIT", "3BIT", "4BIT", "16BIT",
		"1Bit", "2Bit", "3Bit", "4Bit", "16Bit",
		"1bIt", "2bIt", "3bIt", "4bIt", "16bIt",
	}

	for _, level := range quantLevels {
		matched := quantRegix.FindString(level)
		if matched != level {
			t.Errorf("quantRegix did not match: %s, %s", level, matched)
		}
	}
}


var sampleListModels = []types.ModelManifest{
	{
		Name:      "acme/llama",
		ModelType: types.ModelTypeLLM,
		PluginId:  "llama_cpp",
		ModelFile: map[string]types.ModelFileInfo{
			"Q4_0": {Name: "llama-q4_0.gguf", Downloaded: true, Size: 1024},
			"Q8_0": {Name: "llama-q8_0.gguf", Downloaded: true, Size: 2048},
			"FP16": {Name: "llama-fp16.gguf", Downloaded: false, Size: 4096},
		},
	},
	{
		Name:      "acme/yolo",
		ModelType: "",
		PluginId:  "qairt",
		ModelFile: map[string]types.ModelFileInfo{
			types.QuantNA: {Name: "yolo.zip", Downloaded: true, Size: 512},
		},
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
	if strings.Contains(out, "PLUGIN") || strings.Contains(out, types.QuantNA) {
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
		t.Errorf("got[0].Size = %d, want 3072 (only downloaded files)", got[0].Size)
	}
	if want := []string{"Q4_0", "Q8_0"}; !slices.Equal(got[0].Precisions, want) {
		t.Errorf("got[0].Precisions = %v, want %v", got[0].Precisions, want)
	}
	// JSON keeps the full inventory regardless of --verbose, so QuantNA is exposed.
	if want := []string{types.QuantNA}; !slices.Equal(got[1].Precisions, want) {
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
		{"acme/yolo", "512", "qairt", "", types.QuantNA},
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


// Single-file qairt input (e.g. AI Hub's <repo>.zip) goes straight into
// ModelFile["N/A"]; multi-file qairt input is left for the hub's
// PostDownload to promote a placeholder entrypoint.
func TestChooseFiles_QairtSingleFile(t *testing.T) {
	files := []model_hub.ModelFileInfo{{Name: "foo.zip", Size: 1234}}
	res := &types.ModelManifest{PluginId: "qairt"}
	if err := chooseFiles("acme/foo", "", files, res); err != nil {
		t.Fatalf("chooseFiles: %v", err)
	}
	main, ok := res.ModelFile["N/A"]
	if !ok || main.Name != "foo.zip" || main.Size != 1234 {
		t.Errorf("main = %+v, want foo.zip/1234", main)
	}
	if len(res.ExtraFiles) != 0 {
		t.Errorf("ExtraFiles = %v, want none for single-file qairt", res.ExtraFiles)
	}
}
