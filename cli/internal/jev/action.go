// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// Package jev implements the safe, local browser-agent loop used by geniex jev.
package jev

import bindingjev "github.com/qualcomm/GenieX/bindings/go/jev"

// These aliases keep the CLI host policy and CDP adapter on the public, stable
// JEV decision wire protocol. Browser-specific target freshness and side-effect
// approval remain internal to the CLI.
type (
	ActionKind  = bindingjev.ActionKind
	Action      = bindingjev.Action
	Element     = bindingjev.Element
	Observation = bindingjev.Observation
)

// SnapshotFingerprint identifies trusted snapshot state without exposing it to
// the model. The browser adapter uses it to reject stale actions before CDP
// dispatch.
func SnapshotFingerprint(url, title string, elements []Element) (string, error) {
	return bindingjev.SnapshotFingerprint(url, title, elements)
}

const (
	ActionNavigate  = bindingjev.ActionNavigate
	ActionClick     = bindingjev.ActionClick
	ActionType      = bindingjev.ActionType
	ActionScroll    = bindingjev.ActionScroll
	ActionWait      = bindingjev.ActionWait
	ActionRead      = bindingjev.ActionRead
	ActionGoBack    = bindingjev.ActionGoBack
	ActionGoForward = bindingjev.ActionGoForward
	ActionFinish    = bindingjev.ActionFinish
)
