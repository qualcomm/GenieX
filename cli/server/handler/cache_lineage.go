// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"

	"github.com/qualcomm/GenieX/cli/server/types"
)

const (
	managedCacheSessionHeader   = "GenieX-Cache-Session"
	managedCacheParentHeader    = "GenieX-Cache-Parent"
	managedCacheProtocolHeader  = "GenieX-Cache-Protocol"
	managedCacheProtocolVersion = "2"
)

var (
	managedSessionPattern = regexp.MustCompile(`^[0-9a-f]{32}$`)
	revisionPattern       = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
	errStaleCacheTxn      = errors.New("managed cache transaction is not active")
)

var managedChatLineage = newLineageStore()

// invalidateManagedLineageForUnmanagedRequest is the single invalidation
// boundary for every endpoint that can touch the shared model handle without
// participating in the managed-cache transaction protocol. In particular,
// raw KeepCache completion and logits requests can mutate KV state without a
// model-generation change, so generation checks alone cannot detect them.
func invalidateManagedLineageForUnmanagedRequest() {
	managedChatLineage.Clear()
}

// managedCacheMetadata is intentionally small and is emitted as an extension
// on the final OpenAI response/chunk. A missing record means the request never
// committed and its model state must not be reused.
type managedCacheMetadata struct {
	Mode     string `json:"mode"`
	Status   string `json:"status"`
	Revision string `json:"revision"`
	Reason   string `json:"reason"`
	Reusable bool   `json:"reusable"`
}

type lineageMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// lineageIdentity contains every request property that can change how the
// retained prompt state is interpreted. The loaded-model generation is bound
// separately because it changes when the server reloads or resets the handle.
type lineageIdentity struct {
	Model           string                  `json:"model"`
	Artifact        lineageArtifactIdentity `json:"artifact"`
	ModelParam      types.ModelParam        `json:"model_param"`
	EnableThink     bool                    `json:"enable_think"`
	ReasoningFormat string                  `json:"reasoning_format"`
	GrammarPath     string                  `json:"grammar_path"`
	GrammarString   string                  `json:"grammar_string"`
}

type lineageRequest struct {
	Session  string
	Parent   string
	Identity lineageIdentity
	Messages []lineageMessage
}

type lineageDecision struct {
	TxnID  uint64
	Reuse  bool
	Status string
	Reason string
}

type pendingLineage struct {
	ID         uint64
	Request    lineageRequest
	Decision   lineageDecision
	Generation uint64
}

type committedLineage struct {
	Session    string
	Revision   string
	Identity   lineageIdentity
	Messages   []lineageMessage
	Generation uint64
	Reusable   bool
}

// lineageStore owns a single cache lineage because geniex serve owns one
// mutable model handle. Its mutex makes the state machine independently safe;
// the HTTP server's request GIL remains the outer model-handle guard.
type lineageStore struct {
	mu        sync.Mutex
	nextTxn   uint64
	committed *committedLineage
	pending   *pendingLineage
}

func newLineageStore() *lineageStore { return &lineageStore{} }

func (s *lineageStore) Begin(req lineageRequest) (lineageDecision, error) {
	if !managedSessionPattern.MatchString(req.Session) {
		return lineageDecision{}, fmt.Errorf("%s must be 32 lowercase hexadecimal characters", managedCacheSessionHeader)
	}
	if req.Parent != "" && !revisionPattern.MatchString(req.Parent) {
		return lineageDecision{}, fmt.Errorf("%s must be empty or sha256:<64 lowercase hexadecimal characters>", managedCacheParentHeader)
	}
	if len(req.Messages) == 0 {
		return lineageDecision{}, errors.New("managed caching requires at least one message")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// An abandoned in-process transaction is not a reusable state. This should
	// not occur under the request GIL, but clearing it makes direct callers and
	// future server changes fail closed.
	if s.pending != nil {
		s.pending = nil
		s.committed = nil
	}

	s.nextTxn++
	decision := lineageDecision{TxnID: s.nextTxn, Status: "reset", Reason: "branch"}
	current := s.committed
	switch {
	case current == nil && req.Parent == "":
		decision.Status = "cold"
		decision.Reason = "first_request"
	case current == nil:
		decision.Reason = "parent_mismatch"
	case current.Session != req.Session:
		decision.Reason = "session_switch"
	case current.Identity != req.Identity:
		// Identity changes are an incompatible transcript branch. Keep the
		// public reason vocabulary deliberately small and versioned.
		decision.Reason = "branch"
	case current.Revision != req.Parent:
		decision.Reason = "parent_mismatch"
	case strictMessageExtension(current.Messages, req.Messages):
		if current.Reusable {
			decision.Reuse = true
			decision.Status = "reused"
			decision.Reason = "exact_extension"
		} else {
			decision.Reason = "previous_not_reusable"
		}
	default:
		decision.Reason = "branch"
	}

	if !decision.Reuse {
		s.committed = nil
	}
	s.pending = &pendingLineage{ID: decision.TxnID, Request: cloneLineageRequest(req), Decision: decision}
	return decision, nil
}

// BindGeneration associates a transaction with the exact loaded/reset model
// generation. A model reload between calls turns a planned reuse into a safe
// cold reset; it can never remain reported as a hit.
func (s *lineageStore) BindGeneration(txnID, generation uint64) (lineageDecision, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.pending == nil || s.pending.ID != txnID {
		return lineageDecision{}, errStaleCacheTxn
	}
	if s.pending.Decision.Reuse && (s.committed == nil || s.committed.Generation != generation) {
		s.committed = nil
		s.pending.Decision.Reuse = false
		s.pending.Decision.Status = "reset"
		s.pending.Decision.Reason = "parent_mismatch"
	}
	s.pending.Generation = generation
	return s.pending.Decision, nil
}

func (s *lineageStore) Commit(txnID uint64, assistantContent string, reusable bool) (managedCacheMetadata, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.pending == nil || s.pending.ID != txnID {
		return managedCacheMetadata{}, errStaleCacheTxn
	}
	pending := s.pending
	messages := append([]lineageMessage(nil), pending.Request.Messages...)
	messages = append(messages, lineageMessage{Role: "assistant", Content: assistantContent})
	revision, err := computeLineageRevision(pending.Request.Identity, messages)
	if err != nil {
		s.pending = nil
		s.committed = nil
		return managedCacheMetadata{}, err
	}
	s.committed = &committedLineage{
		Session:    pending.Request.Session,
		Revision:   revision,
		Identity:   pending.Request.Identity,
		Messages:   messages,
		Generation: pending.Generation,
		Reusable:   reusable,
	}
	s.pending = nil
	return managedCacheMetadata{
		Mode:     "managed",
		Status:   pending.Decision.Status,
		Revision: revision,
		Reason:   pending.Decision.Reason,
		Reusable: reusable,
	}, nil
}

// Abort invalidates both the provisional request and the previously committed
// lineage. Generation may already have mutated the model's KV state, so keeping
// either would permit a false reuse after cancellation or error.
func (s *lineageStore) Abort(txnID uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.pending != nil && s.pending.ID == txnID {
		s.pending = nil
	}
	s.committed = nil
}

func (s *lineageStore) Clear() {
	s.mu.Lock()
	s.pending = nil
	s.committed = nil
	s.mu.Unlock()
}

func strictMessageExtension(prefix, candidate []lineageMessage) bool {
	if len(candidate) <= len(prefix) {
		return false
	}
	for i := range prefix {
		if prefix[i] != candidate[i] {
			return false
		}
	}
	return true
}

func computeLineageRevision(identity lineageIdentity, messages []lineageMessage) (string, error) {
	envelope := struct {
		Version  string           `json:"version"`
		Identity lineageIdentity  `json:"identity"`
		Messages []lineageMessage `json:"messages"`
	}{Version: "geniex.managed-cache/2", Identity: identity, Messages: messages}
	raw, err := json.Marshal(envelope)
	if err != nil {
		return "", fmt.Errorf("canonicalize managed cache state: %w", err)
	}
	digest := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}

func cloneLineageRequest(req lineageRequest) lineageRequest {
	req.Messages = append([]lineageMessage(nil), req.Messages...)
	return req
}

func normalizedReasoningFormat(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "none" {
		return ""
	}
	return value
}

func lineageRequestFromChat(param ChatCompletionRequest, modelParam types.ModelParam, artifact lineageArtifactIdentity, session, parent string) (lineageRequest, error) {
	if len(param.Tools) != 0 {
		return lineageRequest{}, errors.New("managed caching does not support native tool-call messages in version 2")
	}
	if reasoningSeparated(param.ReasoningFormat) {
		return lineageRequest{}, errors.New("managed caching requires inline or disabled reasoning in version 2")
	}
	if modelParam.Spec.Type != "" {
		return lineageRequest{}, errors.New("managed caching does not support speculative decoding in version 2")
	}
	messages := make([]lineageMessage, 0, len(param.Messages))
	for i, message := range param.Messages {
		if len(message.GetToolCalls()) != 0 || message.GetToolCallID() != nil {
			return lineageRequest{}, fmt.Errorf("managed caching does not support tool-call message at index %d", i)
		}
		role := message.GetRole()
		if role == nil || (*role != "system" && *role != "user" && *role != "assistant") {
			return lineageRequest{}, fmt.Errorf("managed caching requires system, user, or assistant role at index %d", i)
		}
		content, ok := message.GetContent().AsAny().(*string)
		if !ok || content == nil {
			return lineageRequest{}, fmt.Errorf("managed caching requires scalar text content at index %d", i)
		}
		messages = append(messages, lineageMessage{Role: *role, Content: *content})
	}
	return lineageRequest{
		Session: session,
		Parent:  parent,
		Identity: lineageIdentity{
			Model:           string(param.Model),
			Artifact:        artifact,
			ModelParam:      modelParam,
			EnableThink:     param.EnableThink,
			ReasoningFormat: normalizedReasoningFormat(param.ReasoningFormat),
			GrammarPath:     param.GrammarPath,
			GrammarString:   param.GrammarString,
		},
		Messages: messages,
	}, nil
}
