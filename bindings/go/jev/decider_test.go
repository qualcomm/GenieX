// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package jev

import (
	"context"
	"errors"
	"strings"
	"testing"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

func ptr(value int) *int { return &value }

func observation() Observation {
	return Observation{
		URL:            "https://example.test/page",
		Title:          "Example",
		ScreenshotPath: "C:/temp/screenshot.png",
		Elements: []Element{
			{Index: 1, Tag: "a", Name: "Documentation", Href: "https://example.test/docs"},
			{Index: 2, Tag: "textarea", Name: "Notes"},
		},
	}
}

func TestParseAction(t *testing.T) {
	obs := observation()
	for _, test := range []struct {
		name string
		body string
		want ActionKind
		ok   bool
	}{
		{"valid click", `{"action":"click","index":1}`, ActionClick, true},
		{"fenced action", "```json\n{\"action\":\"read\",\"index\":1}\n```", ActionRead, true},
		{"read without index", `{"action":"read"}`, "", false},
		{"unknown field", `{"action":"click","index":1,"selector":"#x"}`, "", false},
		{"two values", `{"action":"read","index":1}{"action":"read","index":1}`, "", false},
		{"invalid URL", `{"action":"navigate","url":"javascript:alert(1)"}`, "", false},
		{"invalid target", `{"action":"click","index":100}`, "", false},
		{"invalid scroll", `{"action":"scroll","direction":"left","amount":1}`, "", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			action, err := ParseAction(test.body, obs)
			if test.ok {
				if err != nil {
					t.Fatalf("ParseAction() error = %v", err)
				}
				if action.Action != test.want {
					t.Fatalf("action = %q, want %q", action.Action, test.want)
				}
				return
			}
			if err == nil {
				t.Fatalf("ParseAction(%q) succeeded", test.body)
			}
		})
	}
}

func TestSnapshotFingerprintIsStableAndOrderIndependent(t *testing.T) {
	first, err := SnapshotFingerprint("https://example.test/page", "Example", []Element{{Index: 2, Tag: "textarea"}, {Index: 1, Tag: "a"}})
	if err != nil {
		t.Fatalf("SnapshotFingerprint() error = %v", err)
	}
	second, err := SnapshotFingerprint("https://example.test/page", "Example", []Element{{Index: 1, Tag: "a"}, {Index: 2, Tag: "textarea"}})
	if err != nil {
		t.Fatalf("SnapshotFingerprint() error = %v", err)
	}
	changed, err := SnapshotFingerprint("https://example.test/other", "Example", []Element{{Index: 1, Tag: "a"}, {Index: 2, Tag: "textarea"}})
	if err != nil {
		t.Fatalf("SnapshotFingerprint() error = %v", err)
	}
	if first != second || first == changed {
		t.Fatalf("fingerprints = (%q, %q, %q)", first, second, changed)
	}
}

func TestActionValidate(t *testing.T) {
	obs := observation()
	for _, test := range []struct {
		name   string
		action Action
		ok     bool
	}{
		{"valid type", Action{Action: ActionType, Index: ptr(2), Text: "note"}, true},
		{"read target", Action{Action: ActionRead, Index: ptr(1)}, true},
		{"read without target", Action{Action: ActionRead}, false},
		{"type link", Action{Action: ActionType, Index: ptr(1), Text: "note"}, false},
		{"bounded wait", Action{Action: ActionWait, Milliseconds: 30000}, true},
		{"long wait", Action{Action: ActionWait, Milliseconds: 30001}, false},
		{"finish", Action{Action: ActionFinish, FinalAnswer: "done"}, true},
		{"empty finish", Action{Action: ActionFinish}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := test.action.Validate(obs)
			if (err == nil) != test.ok {
				t.Fatalf("Validate() error = %v, want success=%v", err, test.ok)
			}
		})
	}
}

type fakeVLM struct {
	templateInput geniex_sdk.VlmApplyChatTemplateInput
	generateInput geniex_sdk.VlmGenerateInput
	response      *geniex_sdk.VlmGenerateOutput
	generateErr   error
}

func (f *fakeVLM) ApplyChatTemplate(input geniex_sdk.VlmApplyChatTemplateInput) (*geniex_sdk.VlmApplyChatTemplateOutput, error) {
	f.templateInput = input
	return &geniex_sdk.VlmApplyChatTemplateOutput{FormattedText: "formatted"}, nil
}

func (f *fakeVLM) Generate(input geniex_sdk.VlmGenerateInput) (*geniex_sdk.VlmGenerateOutput, error) {
	f.generateInput = input
	return f.response, f.generateErr
}

func TestDeciderBuildsConstrainedVisualRequest(t *testing.T) {
	model := &fakeVLM{response: &geniex_sdk.VlmGenerateOutput{FullText: `{"action":"read","index":1}`}}
	response, err := (VLMDecider{VLM: model, RuntimeID: geniex_sdk.RuntimeLlamaCpp}).DecideAction(
		context.Background(), "read the page", observation(), []string{"scrolled"}, "bad JSON")
	if err != nil {
		t.Fatalf("DecideAction() error = %v", err)
	}
	if response != `{"action":"read","index":1}` {
		t.Fatalf("response = %q", response)
	}
	if len(model.templateInput.Messages) != 2 || len(model.templateInput.Messages[1].Contents) != 2 {
		t.Fatalf("template input = %#v", model.templateInput)
	}
	if model.templateInput.Messages[1].Contents[1].Text != observation().ScreenshotPath {
		t.Fatalf("image = %#v", model.templateInput.Messages[1].Contents[1])
	}
	if model.generateInput.Config.ImagePaths[0] != observation().ScreenshotPath {
		t.Fatalf("image paths = %#v", model.generateInput.Config.ImagePaths)
	}
	if model.generateInput.Config.SamplerConfig.GrammarString != Grammar {
		t.Fatalf("sampler = %#v", model.generateInput.Config.SamplerConfig)
	}
	if !strings.Contains(model.templateInput.Messages[1].Contents[0].Text, "Previous response was rejected") {
		t.Fatalf("prompt missing correction = %q", model.templateInput.Messages[1].Contents[0].Text)
	}
}

func TestDeciderReturnsPartialOutput(t *testing.T) {
	model := &fakeVLM{
		response:    &geniex_sdk.VlmGenerateOutput{FullText: `{"action":"read","index":1}`},
		generateErr: errors.New("cut off"),
	}
	response, err := (VLMDecider{VLM: model}).DecideAction(context.Background(), "read", observation(), nil, "")
	if err != nil || response != `{"action":"read","index":1}` {
		t.Fatalf("DecideAction() = (%q, %v)", response, err)
	}
}
