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
	"errors"
	"net/http"
	"os"
	"slices"
	"strings"

	"github.com/bytedance/sonic"
	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v3"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/store"
)

func ListModels(c *gin.Context) {
	s := store.Get()

	models, err := s.List()
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}

	res := make([]openai.Model, 0, len(models))
	for _, m := range models {
		for q, f := range m.ModelFile {
			if !f.Downloaded {
				continue
			}
			id := m.Name
			if q != "N/A" {
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
	name := strings.TrimPrefix(c.Param("model"), "/")
	name, quant := model_hub.NormalizeModelName(name)

	// check model exist
	s := store.Get()
	manifest, err := s.GetManifest(name)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			c.JSON(http.StatusNotFound, nil)
		} else {
			c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error()})
		}
		return
	}

	// fill quant if not specified
	if quant == "" {
		quants := make([]string, 0, len(manifest.ModelFile))
		for quant, v := range manifest.ModelFile {
			if v.Downloaded {
				quants = append(quants, quant)
				break
			}
		}
		slices.Sort(quants)
		quant = quants[0]
	}

	// check quant exist
	if _, ok := manifest.ModelFile[quant]; !ok {
		c.JSON(http.StatusNotFound, nil)
		return
	}

	// compact with openai format
	var res map[string]any
	ms, _ := sonic.Marshal(manifest)
	_ = sonic.Unmarshal(ms, &res)
	model := openai.Model{}
	model.ID = name
	if quant != "N/A" {
		model.ID += ":" + quant
	}
	model.OwnedBy = strings.Split(manifest.Name, "/")[0]
	ms, _ = sonic.Marshal(model)
	_ = sonic.Unmarshal(ms, &res)

	c.JSON(http.StatusOK, res)
}
