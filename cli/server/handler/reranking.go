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

package handler

import (
	"log/slog"
	"net/http"

	"github.com/gin-gonic/gin"

	geniex_bridge "github.com/qcom-it-nexa-ai/geniex/bindings/go"
	"github.com/qcom-it-nexa-ai/geniex/internal/types"
	"github.com/qcom-it-nexa-ai/geniex/server/service"
)

type RerankingRequest struct {
	Model           string   `json:"model" binding:"required"`
	Query           string   `json:"query"`
	Documents       []string `json:"documents"`
	BatchSize       int32    `json:"batch_size"`
	NormalizeMethod string   `json:"normalize_method"`
	Normalize       bool     `json:"normalize"`
}

type RerankResponse struct {
	Result []float32 `json:"result"`
}

func Reranking(c *gin.Context) {
	param := RerankingRequest{}
	if err := c.ShouldBindJSON(&param); err != nil {
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	slog.Info("Reranking request received",
		"model", param.Model,
		"query", param.Query,
		"documents", param.Documents,
	)

	p, err := service.KeepAliveGet[geniex_bridge.Reranker](
		string(param.Model),
		types.ModelParam{},
		false,
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_bridge.SDKErrorCode(err)})
		return
	}

	if param.Query == "" || len(param.Documents) == 0 {
		if param.Query != "" || len(param.Documents) != 0 {
			c.JSON(http.StatusBadRequest, map[string]any{"error": "both query and documents must be provided"})
			return
		}
		c.JSON(http.StatusOK, nil)
		return
	}

	res, err := p.Rerank(geniex_bridge.RerankerRerankInput{
		Query:     param.Query,
		Documents: param.Documents,
		Config: &geniex_bridge.RerankConfig{
			BatchSize:       param.BatchSize,
			Normalize:       param.Normalize,
			NormalizeMethod: param.NormalizeMethod,
		},
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_bridge.SDKErrorCode(err)})
	} else {
		c.JSON(http.StatusOK, RerankResponse{Result: res.Scores})
	}
}
