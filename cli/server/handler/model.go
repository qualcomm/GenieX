// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"net/http"
	"slices"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v3"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
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
			if q != geniex_sdk.PrecisionNA {
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
	name, quant := geniex_sdk.SplitNamePrecision(strings.TrimPrefix(c.Param("model"), "/"))

	// ModelGetDetailed canonicalizes the name in the SDK, so an alias form the
	// store never records under (a bare AI Hub id) still resolves.
	m, err := geniex_sdk.ModelGetDetailed(name)
	if err != nil {
		if geniex_sdk.IsModelNotFound(err) {
			c.JSON(http.StatusNotFound, nil)
		} else {
			c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error()})
		}
		return
	}

	if quant == "" {
		precisions := slices.Sorted(slices.Values(m.Precisions))
		if len(precisions) == 0 {
			c.JSON(http.StatusNotFound, nil)
			return
		}
		quant = precisions[0]
	} else if i := slices.IndexFunc(m.Precisions, func(p string) bool {
		return strings.EqualFold(p, quant)
	}); i < 0 {
		c.JSON(http.StatusNotFound, nil)
		return
	} else {
		// echo the precision as the store spells it, not as the caller cased it
		quant = m.Precisions[i]
	}

	id := m.Name
	if quant != geniex_sdk.PrecisionNA {
		id += ":" + quant
	}
	c.JSON(http.StatusOK, openai.Model{
		ID:      id,
		OwnedBy: strings.Split(m.Name, "/")[0],
	})
}
