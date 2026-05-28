// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model_hub

import (
	"errors"
	"fmt"
	"net/http"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub/aihub"
)

// Hub-level sentinels surfaced to the CLI top level so PrintError can
// render a friendly hint. Producers wrap with %w; consumers match with
// errors.Is. These are hub-agnostic — both the HuggingFace and AI Hub
// backends translate their transport errors into these.
var (
	ErrUnreachable   = errors.New("hub unreachable")
	ErrAuthRequired  = errors.New("hub auth required")
	ErrModelNotFound = errors.New("model not found on hub")
)

// TranslateAIHubError converts the AI Hub backend's local errors into the
// hub-agnostic sentinels above. Any other error is returned unchanged.
// Exported so callers that talk to aihub.Client directly (e.g. the
// CLI's device picker) can surface the same hints.
func TranslateAIHubError(err error) error {
	if errors.Is(err, aihub.ErrModelNotFound) {
		return fmt.Errorf("%w: %v", ErrModelNotFound, err)
	}
	var he *aihub.HTTPError
	if errors.As(err, &he) {
		switch he.Status {
		case http.StatusNotFound:
			return fmt.Errorf("%w: %s", ErrModelNotFound, he.URL)
		case http.StatusUnauthorized, http.StatusForbidden:
			return fmt.Errorf("%w: %s", ErrAuthRequired, he.URL)
		default:
			return fmt.Errorf("%w: %v", ErrUnreachable, he)
		}
	}
	return err
}

// wrapTransport tags a transport-layer error (DNS / dial / timeout) as
// ErrUnreachable. Errors already carrying a hub sentinel pass through
// untouched, so middleware-produced errors keep their original tag.
func wrapTransport(url string, err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, ErrUnreachable) || errors.Is(err, ErrAuthRequired) || errors.Is(err, ErrModelNotFound) {
		return err
	}
	return fmt.Errorf("%w: %s: %v", ErrUnreachable, url, err)
}
