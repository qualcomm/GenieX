// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package server

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/qualcomm/GenieX/cli/internal/config"
)

// WebUIProps implements the subset of llama-server's /props contract used by
// the unmodified llama.cpp Web UI. Values mirror GenieX request defaults so
// loading the UI does not silently change sampling behavior.
func WebUIProps(c *gin.Context) {
	cfg := config.Get()
	modelAlias := "GenieX"
	if cfg.Model != "" {
		modelAlias = cfg.Model
	}
	c.JSON(http.StatusOK, map[string]any{
		"role":               "model",
		"model_alias":        modelAlias,
		"model_path":         "GenieX llama_cpp plugin",
		"total_slots":        1,
		"modalities":         map[string]bool{"vision": false, "audio": false, "video": false},
		"chat_template":      "",
		"bos_token":          "",
		"eos_token":          "",
		"build_info":         "GenieX llama_cpp Web UI compatibility",
		"cors_proxy_enabled": false,
		"default_generation_settings": map[string]any{
			"id":            0,
			"id_task":       -1,
			"n_ctx":         cfg.NCtx,
			"speculative":   false,
			"is_processing": false,
			"prompt":        "",
			"params": map[string]any{
				"n_predict":          2048,
				"max_tokens":         2048,
				"seed":               0,
				"temperature":        0.0,
				"dynatemp_range":     0.0,
				"dynatemp_exponent":  1.0,
				"top_k":              0,
				"top_p":              0.0,
				"min_p":              0.0,
				"typical_p":          1.0,
				"repeat_last_n":      0,
				"repeat_penalty":     1.0,
				"presence_penalty":   0.0,
				"frequency_penalty":  0.0,
				"dry_multiplier":     0.0,
				"dry_base":           1.75,
				"dry_allowed_length": 2,
				"dry_penalty_last_n": 0,
				"samplers":           []string{},
				"backend_sampling":   false,
				"stream":             true,
				"timings_per_token":  false,
			},
			"next_token": map[string]any{
				"has_next_token": false,
				"has_new_line":   false,
				"n_remain":       0,
				"n_decoded":      0,
				"stopping_word":  "",
			},
		},
	})
}
