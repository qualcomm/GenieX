// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package jev

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func pointer(value int) *int { return &value }

func testObservation() Observation {
	return Observation{
		URL: "https://example.test/start",
		Elements: []Element{
			{Index: 1, Tag: "a", Name: "Read docs", Href: "https://example.test/docs"},
			{Index: 2, Tag: "button", Name: "Buy now"},
			{Index: 3, Tag: "input", Type: "password", Name: "Password"},
			{Index: 4, Tag: "textarea", Name: "Notes"},
		},
	}
}

func TestParseActionRejectsUnsafeOrInvalidActions(t *testing.T) {
	observation := testObservation()
	for _, test := range []struct {
		name string
		body string
	}{
		{"unknown field", `{"action":"click","index":1,"selector":"#x"}`},
		{"invalid target", `{"action":"click","index":99}`},
		{"read without index", `{"action":"read"}`},
		{"javascript URL", `{"action":"navigate","url":"javascript:alert(1)"}`},
		{"type button", `{"action":"type","index":2,"text":"x"}`},
		{"bad scroll", `{"action":"scroll","direction":"left","amount":1}`},
		{"empty finish", `{"action":"finish","final_answer":""}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := ParseAction(test.body, observation); err == nil {
				t.Fatalf("ParseAction(%s) succeeded", test.body)
			}
		})
	}
}

func TestApprovalPolicyFailsClosed(t *testing.T) {
	observation := testObservation()
	observation.Elements = append(observation.Elements,
		Element{Index: 5, Tag: "input", Type: "text", Name: "Credit card"},
		Element{Index: 6, Tag: "input", Type: "file", Name: "Upload receipt"},
		Element{Index: 7, Tag: "a", Name: "Documentation", Href: "https://example.test/docs", Target: "_blank"},
		Element{Index: 8, Tag: "button", Name: "Delete account"},
		Element{Index: 9, Tag: "input", Type: "submit", Name: "Send"},
	)
	for _, test := range []struct {
		name     string
		action   Action
		required bool
	}{
		{"read password", Action{Action: ActionRead, Index: pointer(3)}, true},
		{"read plain text", Action{Action: ActionRead, Index: pointer(4)}, false},
		{"read payment text", Action{Action: ActionRead, Index: pointer(5)}, true},
		{"scroll", Action{Action: ActionScroll, Direction: "down", Amount: 100}, false},
		{"wait", Action{Action: ActionWait, Milliseconds: 1}, false},
		{"history back", Action{Action: ActionGoBack}, false},
		{"history forward", Action{Action: ActionGoForward}, false},
		{"finish", Action{Action: ActionFinish, FinalAnswer: "done"}, false},
		{"same origin navigation", Action{Action: ActionNavigate, URL: "https://example.test/docs"}, false},
		{"cross origin navigation", Action{Action: ActionNavigate, URL: "https://other.test"}, true},
		{"same origin link", Action{Action: ActionClick, Index: pointer(1)}, false},
		{"purchase button", Action{Action: ActionClick, Index: pointer(2)}, true},
		{"password", Action{Action: ActionType, Index: pointer(3), Text: "secret"}, true},
		{"plain text", Action{Action: ActionType, Index: pointer(4), Text: "note"}, false},
		{"payment text", Action{Action: ActionType, Index: pointer(5), Text: "4111"}, true},
		{"file chooser", Action{Action: ActionClick, Index: pointer(6)}, true},
		{"new window", Action{Action: ActionClick, Index: pointer(7)}, true},
		{"delete", Action{Action: ActionClick, Index: pointer(8)}, true},
		{"submit", Action{Action: ActionClick, Index: pointer(9)}, true},
		{"unknown action", Action{Action: ActionKind("new_action")}, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			required, _ := ApprovalRequired(ApprovalRead, test.action, observation)
			if required != test.required {
				t.Fatalf("ApprovalRequired() = %v, want %v", required, test.required)
			}
		})
	}
}

type fakeBrowser struct {
	observation Observation
	executed    []Action
	outcome     string
}

func (b *fakeBrowser) Observe(context.Context) (Observation, error) { return b.observation, nil }
func (b *fakeBrowser) Execute(_ context.Context, action Action, _ Observation) (string, error) {
	b.executed = append(b.executed, action)
	if b.outcome != "" {
		return b.outcome, nil
	}
	return "ok", nil
}
func (b *fakeBrowser) Close() error { return nil }

type sequenceDecider struct{ responses []string }

func (d *sequenceDecider) DecideAction(context.Context, string, Observation, []string, string) (string, error) {
	if len(d.responses) == 0 {
		return "", errors.New("no response")
	}
	response := d.responses[0]
	d.responses = d.responses[1:]
	return response, nil
}

type fakeApprover struct{ decision ApprovalDecision }

func (a fakeApprover) Approve(ApprovalRequest) (ApprovalDecision, error) { return a.decision, nil }

func TestAgentRejectsReadWithoutIndexBeforeDispatch(t *testing.T) {
	browser := &fakeBrowser{observation: testObservation()}
	agent := Agent{
		Browser: browser,
		Decider: &sequenceDecider{responses: []string{
			`{"action":"read"}`,
		}},
		Config: AgentConfig{Task: "read docs", MaxSteps: 1, MaxRetries: 0, ApprovalMode: ApprovalRead},
	}
	if _, err := agent.Run(context.Background()); err == nil {
		t.Fatal("Run() accepted an unindexed read")
	}
	if len(browser.executed) != 0 {
		t.Fatalf("unindexed read was dispatched: %#v", browser.executed)
	}
}

func TestAgentRetriesThenRequiresApproval(t *testing.T) {
	browser := &fakeBrowser{observation: testObservation()}
	agent := Agent{
		Browser: browser,
		Decider: &sequenceDecider{responses: []string{
			`{"action":"click","index":99}`,
			`{"action":"click","index":2}`,
			`{"action":"finish","final_answer":"done"}`,
		}},
		Approver: fakeApprover{decision: ApprovalApprove},
		Config: AgentConfig{Task: "buy", MaxSteps: 3, MaxRetries: 1, ApprovalMode: ApprovalRead},
	}
	result, err := agent.Run(context.Background())
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if result.FinalAnswer != "done" || len(browser.executed) != 1 {
		t.Fatalf("result = %#v, executed = %#v", result, browser.executed)
	}
}

func TestAgentReobservesAfterStaleAction(t *testing.T) {
	browser := &staleOnceBrowser{observation: testObservation()}
	decider := &sequenceDecider{responses: []string{
		`{"action":"click","index":1}`,
		`{"action":"click","index":1}`,
		`{"action":"finish","final_answer":"done"}`,
	}}
	agent := Agent{
		Browser:  browser,
		Decider:  decider,
		Approver: fakeApprover{decision: ApprovalApprove},
		Config:   AgentConfig{Task: "read docs", MaxSteps: 3, ApprovalMode: ApprovalRead},
	}
	result, err := agent.Run(context.Background())
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if result.FinalAnswer != "done" || len(browser.executed) != 1 || browser.observations != 3 {
		t.Fatalf("result = %#v, executed = %#v, observations = %d", result, browser.executed, browser.observations)
	}
	if !strings.Contains(strings.Join(result.Steps, "\n"), "re-observe stale click") {
		t.Fatalf("stale action was not recorded: %#v", result.Steps)
	}
}

type staleOnceBrowser struct {
	observation  Observation
	executed     []Action
	observations int
	stale        bool
}

func (b *staleOnceBrowser) Observe(context.Context) (Observation, error) {
	b.observations++
	return b.observation, nil
}

func (b *staleOnceBrowser) Execute(_ context.Context, action Action, _ Observation) (string, error) {
	if !b.stale {
		b.stale = true
		return "", ErrStaleObservation
	}
	b.executed = append(b.executed, action)
	return "ok", nil
}

func (b *staleOnceBrowser) Close() error { return nil }

func TestAgentDoesNotExposePasswordInLedger(t *testing.T) {
	for _, test := range []struct {
		name     string
		action   string
		outcome  string
		contains string
	}{
		{"type", `{"action":"type","index":3,"text":"top-secret"}`, "ok", "top-secret"},
		{"read", `{"action":"read","index":3}`, "top-secret", "top-secret"},
	} {
		t.Run(test.name, func(t *testing.T) {
			browser := &fakeBrowser{observation: testObservation(), outcome: test.outcome}
			agent := Agent{
				Browser: browser,
				Decider: &sequenceDecider{responses: []string{
					test.action,
					`{"action":"finish","final_answer":"done"}`,
				}},
				Approver: fakeApprover{decision: ApprovalApprove},
				Config:   AgentConfig{Task: "log in", MaxSteps: 2, ApprovalMode: ApprovalRead},
			}
			result, err := agent.Run(context.Background())
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			if strings.Contains(strings.Join(result.Steps, "\n"), test.contains) {
				t.Fatalf("ledger leaked secret: %#v", result.Steps)
			}
		})
	}
}
