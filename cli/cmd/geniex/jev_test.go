// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	bindingjev "github.com/qualcomm/GenieX/bindings/go/jev"
)

func TestJEVCommandModes(t *testing.T) {
	command := jevCmd()
	for _, test := range []struct {
		name string
		args []string
		ok   bool
	}{
		{"classify", []string{"model", "classify"}, true},
		{"choose", []string{"model", "choose"}, true},
		{"triage", []string{"model", "triage"}, true},
		{"browser", []string{"model", "browser"}, true},
		{"missing mode", []string{"model"}, false},
		{"unknown mode", []string{"model", "answer"}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := command.Args(command, test.args)
			if (err == nil) != test.ok {
				t.Fatalf("Args(%q) error = %v, want success=%v", test.args, err, test.ok)
			}
		})
	}
}

func TestJEVContext(t *testing.T) {
	if _, err := jevContext(jevOptions{}); err == nil {
		t.Fatal("jevContext accepted missing context")
	}
	if _, err := jevContext(jevOptions{context: "a", contextFile: "b"}); err == nil {
		t.Fatal("jevContext accepted both context sources")
	}
	value, err := jevContext(jevOptions{context: "question"})
	if err != nil || value != "question" {
		t.Fatalf("jevContext() = (%q, %v)", value, err)
	}
}

func TestParseJEVOptions(t *testing.T) {
	options, err := parseJEVOptions([]string{"a=Earth", "b=Mars"})
	if err != nil || len(options) != 2 || options[1].ID != "b" || options[1].Text != "Mars" {
		t.Fatalf("parseJEVOptions() = (%#v, %v)", options, err)
	}
	for _, value := range [][]string{{"invalid"}, {"=empty"}, {"id="}} {
		if _, err := parseJEVOptions(value); err == nil {
			t.Fatalf("parseJEVOptions(%q) succeeded", value)
		}
	}
}

func TestJEVInputValidation(t *testing.T) {
	if err := validateJEVLabels([]string{"yes", "yes"}, "Classify."); err == nil {
		t.Fatal("validateJEVLabels accepted duplicate labels")
	}
	if err := validateJEVLabels([]string{"yes"}, ""); err == nil {
		t.Fatal("validateJEVLabels accepted missing instruction")
	}
	if err := validateJEVOptions([]bindingjev.Option{{ID: "a", Text: "One"}, {ID: "a", Text: "Duplicate"}}, "Choose.", 0, 0); err == nil {
		t.Fatal("validateJEVOptions accepted duplicate IDs")
	}
	if err := validateJEVOptions([]bindingjev.Option{{ID: "a", Text: "One"}}, "Choose.", 0, 2); err == nil {
		t.Fatal("validateJEVOptions accepted invalid bounds")
	}
}

func TestJEVOnlyAcceptsConsumedModelFlags(t *testing.T) {
	command := jevCmd()
	for _, name := range []string{"compute", "ngl", "nctx", "max-tokens", "image-max-length", "threads", "threads-batch", "batch", "ubatch"} {
		if command.Flags().Lookup(name) == nil {
			t.Fatalf("JEV is missing consumed flag --%s", name)
		}
	}
	for _, name := range []string{"temperature", "top-p", "top-k", "min-p", "repetition-penalty", "presence-penalty", "frequency-penalty", "seed", "grammar-path", "grammar-string", "enable-json", "stop", "stop-file", "think", "system-prompt", "input", "prompt", "token-file", "sliding-window", "spec-type", "draft-model", "draft-tokens", "draft-min", "draft-p-min"} {
		if command.Flags().Lookup(name) != nil {
			t.Fatalf("JEV accepts ignored flag --%s", name)
		}
	}

	inferCommand := infer()
	for _, name := range []string{"temperature", "grammar-string", "think", "spec-type", "draft-model"} {
		if inferCommand.Flags().Lookup(name) == nil {
			t.Fatalf("infer lost flag --%s", name)
		}
	}
}

func TestJEVRuntimeOptions(t *testing.T) {
	if err := validateJEVRuntimeOptions(jevOptions{}); err != nil {
		t.Fatalf("validateJEVRuntimeOptions() error = %v", err)
	}
	for _, options := range []jevOptions{
		{threads: -1},
		{threadsBatch: -1},
		{batch: -1},
		{ubatch: -1},
	} {
		if err := validateJEVRuntimeOptions(options); err == nil {
			t.Fatalf("validateJEVRuntimeOptions(%#v) succeeded", options)
		}
	}
	config := jevModelConfig(4096, -1, jevOptions{threads: 4, threadsBatch: 6, batch: 512, ubatch: 256})
	if config.NCtx != 4096 || config.NGpuLayers != -1 || config.NThreads != 4 || config.NThreadsBatch != 6 || config.NBatch != 512 || config.NUbatch != 256 {
		t.Fatalf("jevModelConfig() = %#v", config)
	}
}

func TestJEVJSONShape(t *testing.T) {
	encoded, err := json.Marshal(bindingjev.ClassificationResult{Label: "capital_of_france"})
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if string(encoded) != `{"label":"capital_of_france"}` {
		t.Fatalf("classification JSON = %s", encoded)
	}
	encoded, err = json.Marshal(bindingjev.MultipleChoiceResult{SelectedIDs: []string{"paris"}})
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if string(encoded) != `{"selected_ids":["paris"]}` {
		t.Fatalf("multiple-choice JSON = %s", encoded)
	}
	encoded, err = json.Marshal(bindingjev.TriageResult{Intent: "refund", IsUrgent: "no", Frustration: "2", RefundRequested: "yes", ChurnRisk: "no"})
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if string(encoded) != `{"intent":"refund","is_urgent":"no","frustration":"2","refund_requested":"yes","churn_risk":"no"}` {
		t.Fatalf("triage JSON = %s", encoded)
	}
}

func TestRunJEVBenchmark(t *testing.T) {
	calls := 0
	resets := 0
	result, timings, err := runJEVBenchmark(2, func() error {
		resets++
		return nil
	}, func() (any, bindingjev.DecisionTiming, error) {
		calls++
		return calls, bindingjev.DecisionTiming{TotalTime: time.Millisecond}, nil
	})
	if err != nil || result != 3 || calls != 3 || resets != 2 || len(timings) != 2 {
		t.Fatalf("runJEVBenchmark() = (%#v, %#v, %v), calls=%d resets=%d", result, timings, err, calls, resets)
	}
}

func TestJEVTriageRequiresMessage(t *testing.T) {
	if err := runJEVTriage(context.Background(), "model", jevOptions{}); err == nil {
		t.Fatal("runJEVTriage accepted a missing message")
	}
}

func TestJEVBenchmarkValidation(t *testing.T) {
	command := jevCmd()
	command.SetArgs([]string{"model", "browser", "--benchmark", "1"})
	if err := command.Execute(); err == nil {
		t.Fatal("browser --benchmark succeeded")
	}
	command = jevCmd()
	command.SetArgs([]string{"model", "classify", "--benchmark", "-1"})
	if err := command.Execute(); err == nil {
		t.Fatal("negative --benchmark succeeded")
	}
	command = jevCmd()
	command.SetArgs([]string{"model", "classify", "--threads", "-1"})
	if err := command.Execute(); err == nil {
		t.Fatal("negative --threads succeeded")
	}
}
