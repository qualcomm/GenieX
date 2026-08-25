// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package common

import (
	"io"
	"log/slog"
	"os"

	"github.com/lmittmann/tint"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/internal/config"
)

const (
	LogLevelNone  string = "none"
	LogLevelTrace string = "trace"
	LogLevelDebug string = "debug"
	LogLevelInfo  string = "info"
	LogLevelWarn  string = "warn"
	LogLevelError string = "error"
)

// ApplySlog configures slog from the resolved log level, without touching the
// SDK callback. Safe to call before flags are parsed.
func ApplySlog() {
	level := config.Get().Log
	if level == LogLevelNone {
		slog.SetDefault(slog.New(slog.NewTextHandler(io.Discard, nil)))
		return
	}

	options := tint.Options{
		AddSource: true,
		Level:     slogLevels[level],
		NoColor:   os.Getenv("NO_COLOR") == "1",
	}
	slog.SetDefault(slog.New(tint.NewHandler(os.Stderr, &options)))
}

// ApplyLogLevel runs after flags are parsed; the only place that sets the SDK
// callback. trace keeps the SDK's native handler; other levels forward to slog.
func ApplyLogLevel() {
	ApplySlog()

	if level := config.Get().Log; level != LogLevelTrace {
		geniex_sdk.SetLog(level != LogLevelNone)
	}
}

var slogLevels = map[string]slog.Level{
	LogLevelTrace: slog.LevelDebug,
	LogLevelDebug: slog.LevelDebug,
	LogLevelInfo:  slog.LevelInfo,
	LogLevelWarn:  slog.LevelWarn,
	LogLevelError: slog.LevelError,
}
