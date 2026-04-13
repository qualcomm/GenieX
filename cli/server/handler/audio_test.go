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
// See the License for the language governing permissions and
// limitations under the License.

package handler

import (
	"encoding/base64"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	geniex_bridge "github.com/qcom-it-nexa-ai/geniex/bindings/go"
)

func TestSpeechStreamSSE(t *testing.T) {
	gin.SetMode(gin.TestMode)

	f, err := os.CreateTemp("", "audio_*.raw")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())
	const payload = "fake-audio-bytes"
	if _, err := f.Write([]byte(payload)); err != nil {
		t.Fatal(err)
	}
	f.Close()

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	profile := geniex_bridge.ProfileData{
		PromptTokens:    10,
		GeneratedTokens: 20,
	}

	speechStreamSSE(c, f.Name(), profile)

	if !strings.HasPrefix(w.Header().Get("Content-Type"), "text/event-stream") {
		t.Errorf("Content-Type: got %s", w.Header().Get("Content-Type"))
	}
	body := w.Body.String()
	if !strings.Contains(body, "speech.audio.delta") {
		t.Error("response missing speech.audio.delta")
	}
	if !strings.Contains(body, "speech.audio.done") {
		t.Error("response missing speech.audio.done")
	}
	if !strings.Contains(body, "input_tokens") || !strings.Contains(body, "output_tokens") {
		t.Error("response missing usage fields")
	}
	b64Payload := base64.StdEncoding.EncodeToString([]byte(payload))
	if !strings.Contains(body, b64Payload) {
		t.Errorf("response should contain base64 of audio chunk %q", b64Payload)
	}
}

func TestSpeechStreamSSE_FileNotFound(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	speechStreamSSE(c, "/nonexistent/path.wav", geniex_bridge.ProfileData{})

	if w.Code != 500 {
		t.Errorf("expected 500, got %d", w.Code)
	}
	if !strings.Contains(w.Body.String(), "error") {
		t.Error("expected error in body")
	}
}
