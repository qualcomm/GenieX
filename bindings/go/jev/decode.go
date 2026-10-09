// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package jev

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// DecodeStrict decodes exactly one JSON object into out. It accepts one fenced
// json object for model compatibility, but rejects prose, unknown fields,
// multiple JSON values, arrays, and trailing data.
func DecodeStrict(response string, out any) error {
	if out == nil {
		return fmt.Errorf("structured decision destination is required")
	}
	body := strings.TrimSpace(response)
	if strings.HasPrefix(body, "```json") && strings.HasSuffix(body, "```") {
		body = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(body, "```json"), "```"))
	}
	if !strings.HasPrefix(body, "{") || !strings.HasSuffix(body, "}") {
		return fmt.Errorf("structured decision must be a JSON object")
	}

	if err := rejectDuplicateJSONFields(body); err != nil {
		return err
	}

	decoder := json.NewDecoder(bytes.NewBufferString(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(out); err != nil {
		return fmt.Errorf("decode structured decision: %w", err)
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		return fmt.Errorf("structured decision contains multiple JSON values")
	}
	return nil
}

// DecodeStructured decodes one strict JSON object then applies host-owned
// semantic validation. The validator is intentionally separate: syntax alone
// cannot establish a decision's business or side-effect safety.
func rejectDuplicateJSONFields(body string) error {
	decoder := json.NewDecoder(bytes.NewBufferString(body))
	return validateJSONObject(decoder)
}

func validateJSONObject(decoder *json.Decoder) error {
	token, err := decoder.Token()
	if err != nil {
		return fmt.Errorf("decode structured decision: %w", err)
	}
	if delimiter, ok := token.(json.Delim); !ok || delimiter != '{' {
		return fmt.Errorf("structured decision must be a JSON object")
	}
	seen := map[string]struct{}{}
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return fmt.Errorf("decode structured decision: %w", err)
		}
		key, ok := token.(string)
		if !ok {
			return fmt.Errorf("structured decision object key is invalid")
		}
		if _, duplicate := seen[key]; duplicate {
			return fmt.Errorf("structured decision contains duplicate field %q", key)
		}
		seen[key] = struct{}{}
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return fmt.Errorf("decode structured decision: %w", err)
		}
	}
	if _, err := decoder.Token(); err != nil {
		return fmt.Errorf("decode structured decision: %w", err)
	}
	return nil
}

func DecodeStructured(response string, out any, validate func() error) error {
	if err := DecodeStrict(response, out); err != nil {
		return err
	}
	if validate != nil {
		if err := validate(); err != nil {
			return err
		}
	}
	return nil
}
