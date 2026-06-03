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
	"net/http"
	"slices"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v3"

	geniex_sdk "github.com/qcom-it-nexa-ai/geniex/bindings/go"
)

func ListModels(c *gin.Context) {
	models, err := geniex_sdk.ModelListDetailed()
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}

	res := make([]openai.Model, 0, len(models))
	for _, m := range models {
		for _, q := range m.Precisions {
			id := m.Name
			if q != geniex_sdk.QuantNA {
				id += ":" + q
			}
			res = append(res, openai.Model{
				ID:      id,
				OwnedBy: strings.Split(m.Name, "/")[0],
			})
		}
	}

	c.JSON(http.StatusOK, map[string]any{
		"object": "list",
		"data":   res,
	})
}

func RetrieveModel(c *gin.Context) {
	name, quant := geniex_sdk.SplitNameQuant(strings.TrimPrefix(c.Param("model"), "/"))

	models, err := geniex_sdk.ModelListDetailed()
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}

	idx := slices.IndexFunc(models, func(m geniex_sdk.ModelDetail) bool { return m.Name == name })
	if idx < 0 {
		c.JSON(http.StatusNotFound, nil)
		return
	}
	m := models[idx]

	if quant == "" {
		precisions := slices.Clone(m.Precisions)
		slices.Sort(precisions)
		if len(precisions) == 0 {
			c.JSON(http.StatusNotFound, nil)
			return
		}
		quant = precisions[0]
	} else if !slices.Contains(m.Precisions, quant) {
		c.JSON(http.StatusNotFound, nil)
		return
	}

	id := name
	if quant != geniex_sdk.QuantNA {
		id += ":" + quant
	}
	c.JSON(http.StatusOK, openai.Model{
		ID:      id,
		OwnedBy: strings.Split(m.Name, "/")[0],
	})
}
