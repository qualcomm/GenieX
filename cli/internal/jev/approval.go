// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package jev

import (
	"fmt"
	"net/url"
	"strings"
)

// ApprovalMode controls which browser operations require a human decision.
type ApprovalMode string

const (
	ApprovalRead ApprovalMode = "read"
	ApprovalNone ApprovalMode = "none"
	ApprovalAll  ApprovalMode = "all"
)

// ApprovalDecision is returned by the terminal/UI implementation.
type ApprovalDecision string

const (
	ApprovalApprove ApprovalDecision = "approve"
	ApprovalSkip    ApprovalDecision = "skip"
	ApprovalAbort   ApprovalDecision = "abort"
)

// ApprovalRequest explains a proposed action before it can have an external
// side effect. Text is redacted by callers when it is potentially secret.
type ApprovalRequest struct {
	Action Action
	Target *Element
	Reason string
}

// ApprovalGate makes per-action confirmation independently testable.
type ApprovalGate interface {
	Approve(ApprovalRequest) (ApprovalDecision, error)
}

// ApprovalRequired reports whether an action must be presented to the user.
// It treats ambiguity as consequential, so new actions and target forms fail
// closed until an explicit policy is added.
func ApprovalRequired(mode ApprovalMode, action Action, observation Observation) (bool, string) {
	if mode == ApprovalNone {
		return true, "approval mode requires confirmation for every action"
	}
	if mode == ApprovalAll {
		return false, "--auto=all selected"
	}

	var element Element
	var found bool
	if action.Index != nil {
		element, found = observation.Element(*action.Index)
	}

	switch action.Action {
	case ActionScroll, ActionWait, ActionGoBack, ActionGoForward, ActionFinish:
		return false, "read-only action"
	case ActionRead:
		if found && isSensitiveElement(element) {
			return true, "reading a password or payment field requires confirmation"
		}
		return false, "read-only action"
	case ActionNavigate:
		if sameOrigin(observation.URL, action.URL) {
			return false, "same-origin navigation"
		}
		return true, "cross-origin navigation"
	case ActionClick:
		if !found {
			return true, "target cannot be classified"
		}
		return elementRequiresApproval(element, observation.URL)
	case ActionType:
		if !found {
			return true, "target cannot be classified"
		}
		if action.Submit || elementRequiresApprovalForType(element) {
			return true, "typing may submit a form or enter sensitive data"
		}
		return false, "non-submitting text entry"
	default:
		return true, "unknown action policy"
	}
}

func elementRequiresApproval(element Element, currentURL string) (bool, string) {
	if element.Form || element.Target == "_blank" {
		return true, "form or new-window target"
	}
	if element.Download {
		return true, "download target"
	}
	if strings.EqualFold(element.Tag, "a") && element.Href != "" {
		if sameOrigin(currentURL, element.Href) {
			return false, "same-origin link navigation"
		}
		return true, "cross-origin link navigation"
	}
	if strings.EqualFold(element.Tag, "input") && strings.EqualFold(element.Type, "file") {
		return true, "file chooser"
	}
	if isSubmitControl(element) {
		return true, "form submission"
	}
	if hasDangerousIntent(element) {
		return true, "target label indicates a consequential action"
	}
	if strings.EqualFold(element.Tag, "button") || strings.EqualFold(element.Role, "button") {
		return true, "button purpose cannot be proven read-only"
	}
	return true, "target purpose cannot be classified"
}

func elementRequiresApprovalForType(element Element) bool {
	return isSensitiveElement(element) || element.Form || hasDangerousIntent(element)
}

func isSensitiveElement(element Element) bool {
	return strings.EqualFold(element.Type, "password") || isPaymentField(element)
}

func isSubmitControl(element Element) bool {
	if strings.EqualFold(element.Tag, "button") && element.Form {
		return true
	}
	if !strings.EqualFold(element.Tag, "input") {
		return false
	}
	switch strings.ToLower(element.Type) {
	case "submit", "image", "button":
		return true
	default:
		return false
	}
}

func isPaymentField(element Element) bool {
	text := strings.ToLower(strings.Join([]string{element.Name, element.Role, element.Type}, " "))
	for _, term := range []string{"card", "credit", "cvv", "cvc", "payment", "billing"} {
		if strings.Contains(text, term) {
			return true
		}
	}
	return false
}

func hasDangerousIntent(element Element) bool {
	text := strings.ToLower(strings.Join([]string{element.Name, element.Role, element.Type}, " "))
	for _, term := range []string{
		"sign in", "signin", "log in", "login", "password", "buy", "purchase", "pay", "checkout", "order",
		"transfer", "send", "confirm", "delete", "remove", "clear", "unsubscribe", "download", "upload",
		"subscribe", "submit", "save", "publish", "post", "share",
	} {
		if strings.Contains(text, term) {
			return true
		}
	}
	return false
}

func sameOrigin(current, next string) bool {
	from, err := url.Parse(current)
	if err != nil {
		return false
	}
	to, err := url.Parse(next)
	if err != nil {
		return false
	}
	return from.Scheme == to.Scheme && from.Host == to.Host && from.Host != ""
}

// RedactedText hides content intended for password/payment controls from traces
// and approval prompts.
func RedactedText(action Action, target *Element) string {
	return RedactSensitiveText(action.Text, target)
}

// RedactSensitiveText hides text associated with password/payment controls.
func RedactSensitiveText(text string, target *Element) string {
	if target != nil && isSensitiveElement(*target) {
		return "<redacted>"
	}
	return text
}

// ApprovalSummary builds a concise terminal description without exposing
// sensitive text.
func ApprovalSummary(request ApprovalRequest) string {
	target := "page"
	if request.Target != nil {
		target = fmt.Sprintf("%s %q", request.Target.Tag, request.Target.Name)
	}
	return fmt.Sprintf("%s on %s: %s", request.Action.Action, target, request.Reason)
}
