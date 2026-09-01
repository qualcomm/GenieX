// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"sync"
	"testing"

	"github.com/qualcomm/GenieX/cli/server/types"
)

const testSessionA = "0123456789abcdef0123456789abcdef"
const testSessionB = "fedcba9876543210fedcba9876543210"

func testLineageRequest(session, parent string, messages ...lineageMessage) lineageRequest {
	return lineageRequest{
		Session: session,
		Parent:  parent,
		Identity: lineageIdentity{
			Model: "llama-8b",
			Artifact: lineageArtifactIdentity{
				Runtime:     "qairt",
				ModelPath:   "fixture/model.bin",
				ModelDigest: "sha256:fixture",
			},
			EnableThink: false,
		},
		Messages: messages,
	}
}

func seedLineage(t *testing.T, store *lineageStore) (managedCacheMetadata, []lineageMessage) {
	t.Helper()
	messages := []lineageMessage{{Role: "system", Content: "safe"}, {Role: "user", Content: "first"}}
	decision, err := store.Begin(testLineageRequest(testSessionA, "", messages...))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.BindGeneration(decision.TxnID, 7); err != nil {
		t.Fatal(err)
	}
	metadata, err := store.Commit(decision.TxnID, "answer", true)
	if err != nil {
		t.Fatal(err)
	}
	return metadata, append(messages, lineageMessage{Role: "assistant", Content: "answer"})
}

func TestManagedCacheFirstRequestAndExactExtension(t *testing.T) {
	store := newLineageStore()
	first, committed := seedLineage(t, store)
	if first.Status != "cold" || first.Reason != "first_request" {
		t.Fatalf("first metadata = %+v", first)
	}

	next := append(committed, lineageMessage{Role: "user", Content: "second"})
	decision, err := store.Begin(testLineageRequest(testSessionA, first.Revision, next...))
	if err != nil {
		t.Fatal(err)
	}
	if !decision.Reuse || decision.Status != "reused" || decision.Reason != "exact_extension" {
		t.Fatalf("extension decision = %+v", decision)
	}
	decision, err = store.BindGeneration(decision.TxnID, 7)
	if err != nil {
		t.Fatal(err)
	}
	metadata, err := store.Commit(decision.TxnID, "next answer", true)
	if err != nil {
		t.Fatal(err)
	}
	if metadata.Status != "reused" || metadata.Revision == first.Revision {
		t.Fatalf("extension metadata = %+v", metadata)
	}
	if !first.Reusable || !metadata.Reusable {
		t.Fatalf("normal completions were marked non-reusable: first=%+v next=%+v", first, metadata)
	}
}

func TestManagedCacheNonReusableCommitForcesTheNextExactExtensionCold(t *testing.T) {
	store := newLineageStore()
	messages := []lineageMessage{{Role: "user", Content: "first"}}
	first, err := store.Begin(testLineageRequest(testSessionA, "", messages...))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = store.BindGeneration(first.TxnID, 7); err != nil {
		t.Fatal(err)
	}
	metadata, err := store.Commit(first.TxnID, "truncated answer", false)
	if err != nil {
		t.Fatal(err)
	}
	if metadata.Reusable {
		t.Fatalf("truncated commit was marked reusable: %+v", metadata)
	}

	committed := append(
		append([]lineageMessage(nil), messages...),
		lineageMessage{Role: "assistant", Content: "truncated answer"},
		lineageMessage{Role: "user", Content: "continue"},
	)
	next, err := store.Begin(testLineageRequest(
		testSessionA, metadata.Revision, committed...,
	))
	if err != nil {
		t.Fatal(err)
	}
	if next.Reuse || next.Status != "reset" || next.Reason != "previous_not_reusable" {
		t.Fatalf("post-truncation decision = %+v", next)
	}
}

func TestManagedCacheRejectsEveryDivergence(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*lineageRequest)
		reason string
	}{
		{"session switch", func(r *lineageRequest) { r.Session = testSessionB }, "session_switch"},
		{"parent mismatch", func(r *lineageRequest) { r.Parent = "sha256:" + fmt.Sprintf("%064d", 0) }, "parent_mismatch"},
		{"edited prefix", func(r *lineageRequest) { r.Messages[1].Content = "changed" }, "branch"},
		{"deleted prefix", func(r *lineageRequest) { r.Messages = r.Messages[1:] }, "branch"},
		{"reordered prefix", func(r *lineageRequest) { r.Messages[0], r.Messages[1] = r.Messages[1], r.Messages[0] }, "branch"},
		{"identity change", func(r *lineageRequest) { r.Identity.Artifact.Runtime = "llama_cpp" }, "branch"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			store := newLineageStore()
			first, committed := seedLineage(t, store)
			req := testLineageRequest(testSessionA, first.Revision, append(committed, lineageMessage{Role: "user", Content: "new"})...)
			tc.mutate(&req)
			decision, err := store.Begin(req)
			if err != nil {
				t.Fatal(err)
			}
			if decision.Reuse || decision.Status != "reset" || decision.Reason != tc.reason {
				t.Fatalf("decision = %+v, want reset/%s", decision, tc.reason)
			}
		})
	}
}

func TestManagedCacheModelReloadDowngradesAHit(t *testing.T) {
	store := newLineageStore()
	first, committed := seedLineage(t, store)
	req := testLineageRequest(testSessionA, first.Revision, append(committed, lineageMessage{Role: "user", Content: "new"})...)
	decision, err := store.Begin(req)
	if err != nil {
		t.Fatal(err)
	}
	decision, err = store.BindGeneration(decision.TxnID, 8)
	if err != nil {
		t.Fatal(err)
	}
	if decision.Reuse || decision.Status != "reset" || decision.Reason != "parent_mismatch" {
		t.Fatalf("decision = %+v", decision)
	}
}

func TestManagedCacheAbortErasesThePriorCommit(t *testing.T) {
	store := newLineageStore()
	first, committed := seedLineage(t, store)
	req := testLineageRequest(testSessionA, first.Revision, append(committed, lineageMessage{Role: "user", Content: "new"})...)
	decision, err := store.Begin(req)
	if err != nil {
		t.Fatal(err)
	}
	store.Abort(decision.TxnID)

	cold, err := store.Begin(testLineageRequest(testSessionA, first.Revision, req.Messages...))
	if err != nil {
		t.Fatal(err)
	}
	if cold.Reuse || cold.Reason != "parent_mismatch" {
		t.Fatalf("post-abort decision = %+v", cold)
	}
}

func TestUnmanagedEndpointInvalidationErasesThePriorCommit(t *testing.T) {
	previous := managedChatLineage
	managedChatLineage = newLineageStore()
	t.Cleanup(func() { managedChatLineage = previous })

	first, committed := seedLineage(t, managedChatLineage)
	invalidateManagedLineageForUnmanagedRequest()

	next := append(committed, lineageMessage{Role: "user", Content: "after-unmanaged-call"})
	decision, err := managedChatLineage.Begin(testLineageRequest(testSessionA, first.Revision, next...))
	if err != nil {
		t.Fatal(err)
	}
	if decision.Reuse || decision.Status != "reset" || decision.Reason != "parent_mismatch" {
		t.Fatalf("post-unmanaged decision = %+v", decision)
	}
}

func TestManagedCacheAbortNextCommitMatchesAColdBaseline(t *testing.T) {
	store := newLineageStore()
	first, committed := seedLineage(t, store)
	nextMessages := append(committed, lineageMessage{Role: "user", Content: "retry-me"})
	pending, err := store.Begin(testLineageRequest(testSessionA, first.Revision, nextMessages...))
	if err != nil {
		t.Fatal(err)
	}
	store.Abort(pending.TxnID)

	retry, err := store.Begin(testLineageRequest(testSessionA, first.Revision, nextMessages...))
	if err != nil {
		t.Fatal(err)
	}
	if retry.Reuse || retry.Reason != "parent_mismatch" {
		t.Fatalf("retry decision = %+v", retry)
	}
	if _, err = store.BindGeneration(retry.TxnID, 8); err != nil {
		t.Fatal(err)
	}
	retried, err := store.Commit(retry.TxnID, "deterministic-answer", true)
	if err != nil {
		t.Fatal(err)
	}

	coldStore := newLineageStore()
	cold, err := coldStore.Begin(testLineageRequest(testSessionA, "", nextMessages...))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = coldStore.BindGeneration(cold.TxnID, 1); err != nil {
		t.Fatal(err)
	}
	baseline, err := coldStore.Commit(cold.TxnID, "deterministic-answer", true)
	if err != nil {
		t.Fatal(err)
	}
	if retried.Revision != baseline.Revision {
		t.Fatalf("post-abort revision %q != cold baseline %q", retried.Revision, baseline.Revision)
	}
}

func TestManagedCacheSessionCanariesNeverCrossLineages(t *testing.T) {
	for _, sessionCount := range []int{1, 2, 4, 8} {
		t.Run(fmt.Sprintf("sessions-%d", sessionCount), func(t *testing.T) {
			store := newLineageStore()
			generation := uint64(1)
			for round := 0; round < 3; round++ {
				for index := 0; index < sessionCount; index++ {
					session := fmt.Sprintf("%032x", index+1)
					canary := fmt.Sprintf("private-canary-%d-%d", index, round)
					decision, err := store.Begin(testLineageRequest(
						session, "", lineageMessage{Role: "user", Content: canary},
					))
					if err != nil {
						t.Fatal(err)
					}
					if decision.Reuse {
						t.Fatalf("session %s reused state containing another canary", session)
					}
					generation++
					if _, err = store.BindGeneration(decision.TxnID, generation); err != nil {
						t.Fatal(err)
					}
					if _, err = store.Commit(decision.TxnID, "answer-for-"+canary, true); err != nil {
						t.Fatal(err)
					}
				}
			}
		})
	}
}

func TestManagedCacheValidatesHeaderShapes(t *testing.T) {
	store := newLineageStore()
	for _, tc := range []lineageRequest{
		testLineageRequest("not-a-session", "", lineageMessage{Role: "user", Content: "x"}),
		testLineageRequest(testSessionA, "sha256:ABC", lineageMessage{Role: "user", Content: "x"}),
		testLineageRequest(testSessionA, ""),
	} {
		if _, err := store.Begin(tc); err == nil {
			t.Fatalf("Begin(%+v) accepted invalid request", tc)
		}
	}
}

func TestManagedCacheConcurrentBeginsFailClosed(t *testing.T) {
	store := newLineageStore()
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			session := fmt.Sprintf("%032x", i+1)
			decision, err := store.Begin(testLineageRequest(session, "", lineageMessage{Role: "user", Content: fmt.Sprint(i)}))
			if err == nil {
				store.Abort(decision.TxnID)
			}
		}(i)
	}
	wg.Wait()
}

func TestManagedCacheHasZeroFalseHitsAcrossRandomizedBranches(t *testing.T) {
	rng := rand.New(rand.NewSource(20260830))
	for iteration := 0; iteration < 1000; iteration++ {
		store := newLineageStore()
		turnCount := 1 + rng.Intn(12)
		initial := []lineageMessage{{Role: "system", Content: "policy-canary"}}
		for turn := 0; turn < turnCount; turn++ {
			initial = append(initial, lineageMessage{Role: "user", Content: fmt.Sprintf("request-%d-%d", iteration, turn)})
			if turn+1 < turnCount {
				initial = append(initial, lineageMessage{Role: "assistant", Content: fmt.Sprintf("answer-%d-%d", iteration, turn)})
			}
		}

		first, err := store.Begin(testLineageRequest(testSessionA, "", initial...))
		if err != nil {
			t.Fatal(err)
		}
		if _, err = store.BindGeneration(first.TxnID, 9); err != nil {
			t.Fatal(err)
		}
		committedMetadata, err := store.Commit(first.TxnID, fmt.Sprintf("committed-%d", iteration), true)
		if err != nil {
			t.Fatal(err)
		}
		candidate := append([]lineageMessage(nil), initial...)
		candidate = append(candidate,
			lineageMessage{Role: "assistant", Content: fmt.Sprintf("committed-%d", iteration)},
			lineageMessage{Role: "user", Content: "extension"},
		)
		req := testLineageRequest(testSessionA, committedMetadata.Revision, candidate...)

		switch rng.Intn(6) {
		case 0:
			req.Messages[rng.Intn(len(req.Messages)-1)].Content += "-mutated"
		case 1:
			index := rng.Intn(len(req.Messages) - 1)
			req.Messages = append(req.Messages[:index:index], req.Messages[index+1:]...)
		case 2:
			req.Messages[0], req.Messages[1] = req.Messages[1], req.Messages[0]
		case 3:
			req.Session = testSessionB
		case 4:
			req.Parent = "sha256:" + fmt.Sprintf("%064x", iteration+1)
		case 5:
			req.Identity.GrammarString = fmt.Sprintf("grammar-%d", iteration)
		}

		decision, err := store.Begin(req)
		if err != nil {
			t.Fatal(err)
		}
		if decision.Reuse {
			t.Fatalf("iteration %d produced a false cache hit: %+v", iteration, decision)
		}
	}
}

func TestManagedCacheRevisionIsDeterministic(t *testing.T) {
	identity := testLineageRequest(testSessionA, "", lineageMessage{}).Identity
	messages := []lineageMessage{{Role: "system", Content: "policy"}, {Role: "user", Content: "hello"}}
	first, err := computeLineageRevision(identity, messages)
	if err != nil {
		t.Fatal(err)
	}
	second, err := computeLineageRevision(identity, append([]lineageMessage(nil), messages...))
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Fatalf("revisions differ: %q != %q", first, second)
	}
}

func TestLineageRequestRejectsUnsupportedMessageForms(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{
			name: "native tools",
			body: `{"model":"fixture","messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]}`,
		},
		{
			name: "multimodal content",
			body: `{"model":"fixture","messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`,
		},
		{
			name: "tool role",
			body: `{"model":"fixture","messages":[{"role":"tool","tool_call_id":"call_1","content":"result"}]}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			param := defaultChatCompletionRequest()
			if err := json.Unmarshal([]byte(tc.body), &param); err != nil {
				t.Fatal(err)
			}
			if _, err := lineageRequestFromChat(param, types.ModelParam{}, lineageArtifactIdentity{Runtime: "qairt"}, testSessionA, ""); err == nil {
				t.Fatal("unsupported request was accepted")
			}
		})
	}
}

func TestLineageRequestRejectsUnboundSpeculativeDraftModel(t *testing.T) {
	param := defaultChatCompletionRequest()
	if err := json.Unmarshal([]byte(`{"model":"fixture","messages":[{"role":"user","content":"hello"}]}`), &param); err != nil {
		t.Fatal(err)
	}
	modelParam := types.ModelParam{Spec: types.SpecParam{Type: "draft", DraftModel: "unbound.gguf"}}
	if _, err := lineageRequestFromChat(param, modelParam, lineageArtifactIdentity{Runtime: "llama_cpp"}, testSessionA, ""); err == nil {
		t.Fatal("speculative decoding was accepted without a bound draft artifact")
	}
}

func FuzzManagedCacheNeverReusesAnEditedPrefix(f *testing.F) {
	f.Add("changed")
	f.Fuzz(func(t *testing.T, replacement string) {
		if replacement == "first" {
			t.Skip()
		}
		store := newLineageStore()
		first, committed := seedLineage(t, store)
		candidate := append([]lineageMessage(nil), committed...)
		candidate[1].Content = replacement
		candidate = append(candidate, lineageMessage{Role: "user", Content: "next"})
		decision, err := store.Begin(testLineageRequest(testSessionA, first.Revision, candidate...))
		if err != nil {
			t.Fatal(err)
		}
		if decision.Reuse {
			t.Fatalf("edited prefix reused for replacement %q", replacement)
		}
	})
}
