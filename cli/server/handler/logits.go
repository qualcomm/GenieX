// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"log/slog"
	"net/http"

	"github.com/gin-gonic/gin"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/internal/config"
	"github.com/qualcomm/GenieX/cli/internal/store"
	"github.com/qualcomm/GenieX/cli/server/service"
	"github.com/qualcomm/GenieX/cli/server/types"
)

// ForwardLogitsRequest is the body for POST /v1/logits.
//
// This is a prefill-only forward pass (raw logits), not the OpenAI generative
// logprobs semantics: the model runs one non-autoregressive pass over the input
// and returns the raw LM-head logits. Input is pre-tokenized (InputIDs); the
// server has no public tokenizer to accept text here.
type ForwardLogitsRequest struct {
	Model    string  `json:"model"`
	InputIDs []int32 `json:"input_ids"`
	// LastOnly emits only the last token's logits row (next-token / MMLU
	// scoring); the default (false) emits every position's row for perplexity.
	LastOnly bool `json:"last_only"`
	// TopN keeps only the top-N logits per row (the SDK does the reduction).
	// nil / omitted defaults to 20; an explicit 0 returns the full vocabulary.
	TopN *int `json:"top_n"`

	// Compute-unit / layer overrides, same semantics as chat completions.
	NCtx    int32  `json:"nctx"`
	Ngl     int32  `json:"ngl"`
	Compute string `json:"compute"`
}

const defaultLogitsTopN = 20

// ForwardLogitsResponse carries the per-position rows. Each row is top-N
// [token_id, logit] pairs sorted by descending logit (like geniex-bench). When
// top_n is 0 each row is the full vocabulary, ordered by token id, and token id
// is the column index — TokenIDs is then omitted.
type ForwardLogitsResponse struct {
	Model     string       `json:"model"`
	NRows     int          `json:"n_rows"`
	VocabSize int          `json:"vocab_size"`
	TopN      int          `json:"top_n"`
	Rows      []ForwardRow `json:"rows"`
}

// ForwardRow is one position's logits. For top-N output, Logits[i] pairs with
// TokenIDs[i]. For full-vocab output, TokenIDs is nil and the index is the token.
type ForwardRow struct {
	TokenIDs []int32   `json:"token_ids,omitempty"`
	Logits   []float32 `json:"logits"`
}

func ForwardLogits(c *gin.Context) {
	cfg := config.Get()
	req := ForwardLogitsRequest{NCtx: cfg.NCtx, Ngl: cfg.Ngl, Compute: cfg.Compute}
	if err := c.ShouldBindJSON(&req); err != nil {
		slog.Error("Failed to bind JSON", "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	if len(req.InputIDs) == 0 {
		c.JSON(http.StatusBadRequest, map[string]any{"error": "input_ids must be non-empty"})
		return
	}

	name, _ := geniex_sdk.SplitNamePrecision(req.Model)
	modelType, err := geniex_sdk.ModelGetType(name)
	if err != nil {
		slog.Error("Failed to get model type", "model", req.Model, "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	if modelType != geniex_sdk.ModelTypeLLM {
		c.JSON(http.StatusBadRequest, map[string]any{"error": "logits is only supported for LLM models"})
		return
	}
	paths, err := geniex_sdk.ModelGetPaths(name)
	if err != nil {
		slog.Error("Failed to resolve model paths", "model", req.Model, "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	modelParam, err := service.ResolveModelParam(paths.RuntimeID, paths.ModelName, req.NCtx, req.Ngl, req.Compute, store.Get().ResolveChipset(true), types.SpecParam{})
	if err != nil {
		slog.Error("Failed to resolve model params", "model", req.Model, "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	p, err := service.KeepAliveGet[geniex_sdk.LLM](
		req.Model,
		modelParam,
		c.GetHeader("GenieX-KeepCache") != "true",
	)
	if writeKeepAliveError(c, err) {
		return
	}

	topN := defaultLogitsTopN
	if req.TopN != nil {
		if *req.TopN < 0 {
			c.JSON(http.StatusBadRequest, map[string]any{"error": "top_n must be >= 0"})
			return
		}
		topN = *req.TopN
	}

	res, err := p.ForwardLogits(req.InputIDs, !req.LastOnly, topN)
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
		return
	}

	rows := make([]ForwardRow, res.NRows)
	for r := 0; r < res.NRows; r++ {
		lo := r * res.RowWidth
		hi := lo + res.RowWidth
		row := ForwardRow{Logits: res.Logits[lo:hi]}
		if res.TokenIDs != nil {
			row.TokenIDs = res.TokenIDs[lo:hi]
		}
		rows[r] = row
	}

	c.JSON(http.StatusOK, ForwardLogitsResponse{
		Model:     req.Model,
		NRows:     res.NRows,
		VocabSize: res.VocabSize,
		TopN:      topN,
		Rows:      rows,
	})
}
