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
	"io"
	"log/slog"
	"os"

	"github.com/lmittmann/tint"

	geniex_sdk "github.com/qcom-it-nexa-ai/geniex/bindings/go"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/config"
)

const (
	LogLevelNone  string = "none"
	LogLevelTrace string = "trace"
	LogLevelDebug string = "debug"
	LogLevelInfo  string = "info"
	LogLevelWarn  string = "warn"
	LogLevelError string = "error"
)

func ApplyLogLevel() {
	options := tint.Options{AddSource: true}

	if os.Getenv("NO_COLOR") == "1" {
		options.NoColor = true
	}

	level := config.Get().Log

	switch level {
	case LogLevelNone:
		geniex_sdk.SetLog(false)
		slog.SetDefault(slog.New(slog.NewTextHandler(io.Discard, nil)))
		return
	case LogLevelTrace:
		options.Level = slog.LevelDebug
		slog.SetDefault(slog.New(tint.NewHandler(os.Stderr, &options)))
		return
	case LogLevelDebug:
		options.Level = slog.LevelDebug
	case LogLevelInfo:
		options.Level = slog.LevelInfo
	case LogLevelWarn:
		options.Level = slog.LevelWarn
	case LogLevelError:
		options.Level = slog.LevelError
	}

	geniex_sdk.SetLog(true)
	slog.SetDefault(slog.New(tint.NewHandler(os.Stderr, &options)))
}
