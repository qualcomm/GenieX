// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package types

import (
	"encoding/json"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestNewResponseMetadata(t *testing.T) {
	seen := map[string]bool{}
	for _, prefix := range []string{"cmpl", "chatcmpl"} {
		t.Run(prefix, func(t *testing.T) {
			for range 2 {
				before := time.Now().Unix()
				meta := NewResponseMetadata(prefix, "test-model")
				after := time.Now().Unix()
				suffix, ok := strings.CutPrefix(meta.ID, prefix+"-")
				if !ok || len(suffix) != 16 {
					t.Fatalf("response ID = %q, want %s- followed by 16 hex digits", meta.ID, prefix)
				}
				if _, err := strconv.ParseUint(suffix, 16, 64); err != nil {
					t.Fatalf("response ID = %q: %v", meta.ID, err)
				}
				if seen[meta.ID] {
					t.Fatalf("reused response ID %q", meta.ID)
				}
				seen[meta.ID] = true
				if meta.Model != "test-model" || meta.Created < before || meta.Created > after {
					t.Errorf("response metadata = %+v, want test-model and creation time in [%d, %d]", meta, before, after)
				}
			}
		})
	}
}

func TestResponseMetadataJSON(t *testing.T) {
	meta := ResponseMetadata{ID: "cmpl-0123456789abcdef", Created: 1700000000, Model: "test-model"}
	body, err := json.Marshal(meta)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]json.RawMessage
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	want := map[string]json.RawMessage{
		"id":      json.RawMessage(`"cmpl-0123456789abcdef"`),
		"created": json.RawMessage(`1700000000`),
		"model":   json.RawMessage(`"test-model"`),
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("response metadata JSON = %s, want id, created and model fields", body)
	}
	var decoded ResponseMetadata
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded != meta {
		t.Errorf("decoded metadata = %+v, want %+v", decoded, meta)
	}
}
