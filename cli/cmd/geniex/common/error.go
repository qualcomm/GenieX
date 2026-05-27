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

package common

import (
	"errors"
	"fmt"
	"log/slog"
	"os"

	geniex_sdk "github.com/qcom-it-nexa-ai/geniex/bindings/go"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/render"
)

// errorHints maps known sentinel errors to a multi-line hint shown in
// place of the raw error message. Keep the list short — only well-
// understood, user-actionable cases belong here.
const (
	hintParamNotSupported = `⚠️ A flag you passed is not supported by the plugin.

👉 Run 'geniex infer -h' to see which flags are plugin-specific.`

	hintNotSupport = `⚠️ Oops. This model type is not supported yet.

👉 Try these:
- Check back later for updates.
- See help in our discord or slack.`

	hintModelLoad = `⚠️ Oops. Model failed to load.

👉 Try these:
- Redownload the model.
- Verify your system meets the model's requirements.
- Check your NPU / GPU driver version and update it if it's out of date.
- See help in our discord or slack.`

	hintPluginLoad = `⚠️ Oops. Plugin failed to load.

👉 Try these:
- Ensure all plugin dependencies are correct.
- See help in our discord or slack.`

	hintPluginInvalid = `⚠️ Oops. Plugin is invalid.

👉 Try these:
- This model may not be compatible with your system. Try another model.
- See help in our discord or slack.`

	hintContextLength = `Context length exceeded, please start a new conversation.`

	hintHubUnreachable = `⚠️ Unable to reach the model hub while resolving metadata.

Possible causes: network timeout, corporate proxy, or firewall.

👉 Try these:
- Check that you can open huggingface.co (or aihub.qualcomm.com) in a browser.
- If you're behind a proxy, set HTTPS_PROXY before running geniex.
- Use a local model path if it's already downloaded.
- If the issue persists, see help in our discord or slack.`

	hintHubAuthRequired = `⚠️ The model hub rejected the request (auth required).

👉 Try these:
- For HuggingFace gated models, set HF_TOKEN to a token with access.
- Verify the token has not expired or been revoked.`

	hintModelNotFound = `⚠️ Model not found on the hub.

👉 Try these:
- Check the spelling of the model name.
- Run 'geniex model list' to see available AI Hub models.
- For HuggingFace models, pass the full repo path (e.g. unsloth/Qwen3-4B-GGUF).`

	hintServerUnreachable = `⚠️ Could not reach the geniex server.

👉 Try these:
- Run 'geniex serve' in another terminal to start the server.
- If the server is on a different host or port, update 'host' via 'geniex config set host <addr>'.`

	hintPrecisionNotFound = `⚠️ The requested precision is not available locally.

👉 Try these:
- Run 'geniex list' to see what's been downloaded for this model.
- Run 'geniex pull <model>:<precision>' to download it.
- Drop the ':<precision>' suffix to be prompted from what's already downloaded.`
)

// CLI-side sentinels. Hub sentinels live in model_hub; SDK sentinels in
// bindings/go. Producers wrap with %w; PrintError matches via errors.Is.
var (
	// ErrServerUnreachable: dial failure against the geniex HTTP server
	// (geniex run path).
	ErrServerUnreachable = errors.New("server unreachable")
	// ErrPrecisionNotFound: the user-specified precision is missing from
	// the model's local manifest (or listed but not downloaded).
	ErrPrecisionNotFound = errors.New("precision not found")
)

var errorHints = []struct {
	sentinel error
	hint     string
}{
	{geniex_sdk.ErrCommonParamNotSupported, hintParamNotSupported},
	{geniex_sdk.ErrCommonNotSupport, hintNotSupport},
	{geniex_sdk.ErrCommonModelLoad, hintModelLoad},
	{geniex_sdk.ErrCommonPluginLoad, hintPluginLoad},
	{geniex_sdk.ErrCommonPluginInvalid, hintPluginInvalid},
	{geniex_sdk.ErrLlmTokenizationContextLength, hintContextLength},
	{model_hub.ErrUnreachable, hintHubUnreachable},
	{model_hub.ErrAuthRequired, hintHubAuthRequired},
	{model_hub.ErrModelNotFound, hintModelNotFound},
	{ErrServerUnreachable, hintServerUnreachable},
	{ErrPrecisionNotFound, hintPrecisionNotFound},
}

// PrintError renders err for the user on stderr in the theme's error style
// and logs the raw wrapped error via slog so the cause chain remains visible
// in the log stream. If err matches a known sentinel, the friendly hint is
// shown; otherwise the raw error message is shown.
func PrintError(err error) {
	if err == nil {
		return
	}
	slog.Error("cli error", "err", err)
	theme := render.GetTheme()
	for _, h := range errorHints {
		if errors.Is(err, h.sentinel) {
			fmt.Fprintln(os.Stderr, theme.Error.Sprint(h.hint))
			return
		}
	}
	fmt.Fprintln(os.Stderr, theme.Error.Sprintf("Error: %s", err))
}
