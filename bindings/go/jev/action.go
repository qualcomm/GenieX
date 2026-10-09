// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// Package jev provides non-executing structured model decisions: strict typed
// JSON decoding, closed-set classification and multiple choice, and constrained
// browser-action candidates. Hosts own browser access, freshness validation,
// approval, and every externally visible action.
package jev

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/url"
	"sort"
	"strings"
)

// ActionKind is the closed set of actions a host may choose to support.
type ActionKind string

const (
	ActionNavigate  ActionKind = "navigate"
	ActionClick     ActionKind = "click"
	ActionType      ActionKind = "type"
	ActionScroll    ActionKind = "scroll"
	ActionWait      ActionKind = "wait"
	ActionRead      ActionKind = "read"
	ActionGoBack    ActionKind = "go_back"
	ActionGoForward ActionKind = "go_forward"
	ActionFinish    ActionKind = "finish"
)

// Action is a single constrained model decision. It intentionally has no
// selector, JavaScript, shell, or arbitrary protocol-command field.
type Action struct {
	Action       ActionKind `json:"action"`
	Index        *int       `json:"index,omitempty"`
	URL          string     `json:"url,omitempty"`
	Text         string     `json:"text,omitempty"`
	Submit       bool       `json:"submit,omitempty"`
	Direction    string     `json:"direction,omitempty"`
	Amount       int        `json:"amount,omitempty"`
	Milliseconds int        `json:"milliseconds,omitempty"`
	FinalAnswer  string     `json:"final_answer,omitempty"`
}

// Element is host-provided metadata for a currently actionable indexed target.
// It is deliberately generic; a browser host may populate it from its trusted
// DOM snapshot, while other hosts may use an equivalent indexed control map.
type Element struct {
	Index       int    `json:"index"`
	Tag         string `json:"tag"`
	Role        string `json:"role,omitempty"`
	Name        string `json:"name,omitempty"`
	Type        string `json:"type,omitempty"`
	Href        string `json:"href,omitempty"`
	Target      string `json:"target,omitempty"`
	Download    bool   `json:"download,omitempty"`
	Form        bool   `json:"form,omitempty"`
	Disabled    bool   `json:"disabled,omitempty"`
	ContentEdit bool   `json:"content_editable,omitempty"`
}

// Observation is the trusted state presented for one decision. ScreenshotPath
// is passed to the VLM as media and is never rendered in the text prompt.
type Observation struct {
	URL            string    `json:"url"`
	Title          string    `json:"title,omitempty"`
	ScreenshotPath string    `json:"-"`
	Elements       []Element `json:"elements"`
	// Fingerprint is trusted host metadata for the snapshot state. It is never
	// model-selectable and is excluded from browser prompts.
	Fingerprint string `json:"fingerprint,omitempty"`
}

// SnapshotFingerprint returns a stable hash of trusted page metadata and the
// current visible control map. It is host-only freshness metadata, not model
// input. Element order is normalized so equivalent snapshots share a hash.
func SnapshotFingerprint(url, title string, elements []Element) (string, error) {
	normalized := append([]Element(nil), elements...)
	sort.Slice(normalized, func(i, j int) bool { return normalized[i].Index < normalized[j].Index })
	payload, err := json.Marshal(struct {
		URL      string    `json:"url"`
		Title    string    `json:"title"`
		Elements []Element `json:"elements"`
	}{URL: url, Title: title, Elements: normalized})
	if err != nil {
		return "", fmt.Errorf("marshal observation fingerprint: %w", err)
	}
	sum := sha256.Sum256(payload)
	return hex.EncodeToString(sum[:]), nil
}

// Element returns a target from this observation only.
func (o Observation) Element(index int) (Element, bool) {
	for _, element := range o.Elements {
		if element.Index == index {
			return element, true
		}
	}
	return Element{}, false
}

// Validate checks action shape and generic scalar bounds. Hosts must also
// validate that a referenced target remains fresh and is executable immediately
// before performing an external action.
func (a Action) Validate(observation Observation) error {
	needsElement := func() (Element, error) {
		if a.Index == nil {
			return Element{}, fmt.Errorf("action %q requires an element index", a.Action)
		}
		element, ok := observation.Element(*a.Index)
		if !ok {
			return Element{}, fmt.Errorf("element index %d is not in the current observation", *a.Index)
		}
		if element.Disabled {
			return Element{}, fmt.Errorf("element index %d is disabled", *a.Index)
		}
		return element, nil
	}

	switch a.Action {
	case ActionNavigate:
		if a.URL == "" {
			return fmt.Errorf("navigate requires a URL")
		}
		u, err := url.Parse(a.URL)
		if err != nil || !u.IsAbs() || (u.Scheme != "http" && u.Scheme != "https") {
			return fmt.Errorf("navigate URL must be an absolute http(s) URL")
		}
	case ActionClick:
		if _, err := needsElement(); err != nil {
			return err
		}
	case ActionType:
		element, err := needsElement()
		if err != nil {
			return err
		}
		if !isTextEntry(element) {
			return fmt.Errorf("element index %d is not a text-entry control", element.Index)
		}
		if len(a.Text) > 4096 {
			return fmt.Errorf("text input exceeds 4096 characters")
		}
	case ActionScroll:
		if a.Direction != "up" && a.Direction != "down" {
			return fmt.Errorf("scroll direction must be up or down")
		}
		if a.Amount < 1 || a.Amount > 5000 {
			return fmt.Errorf("scroll amount must be between 1 and 5000")
		}
	case ActionWait:
		if a.Milliseconds < 1 || a.Milliseconds > 30000 {
			return fmt.Errorf("wait duration must be between 1 and 30000 milliseconds")
		}
	case ActionRead:
		if _, err := needsElement(); err != nil {
			return err
		}
	case ActionGoBack, ActionGoForward:
	case ActionFinish:
		if strings.TrimSpace(a.FinalAnswer) == "" {
			return fmt.Errorf("finish requires a final_answer")
		}
	default:
		return fmt.Errorf("unsupported action %q", a.Action)
	}
	return nil
}

func isTextEntry(element Element) bool {
	if element.ContentEdit || element.Tag == "textarea" {
		return true
	}
	if element.Tag != "input" {
		return false
	}
	switch strings.ToLower(element.Type) {
	case "", "text", "search", "email", "tel", "url", "password", "number":
		return true
	default:
		return false
	}
}
