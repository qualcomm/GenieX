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

package server

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/qcom-it-nexa-ai/geniex/server/docs"
	"github.com/qcom-it-nexa-ai/geniex/server/handler"
	"github.com/qcom-it-nexa-ai/geniex/server/middleware"
)

func RegisterRoot(r *gin.Engine) {
	r.Use(middleware.CORS)
	r.GET("/", func(c *gin.Context) {
		c.String(200, "GenieX-CLI is running")
	})
}

// http://localhost:18181/docs/ui/
func RegisterSwagger(r *gin.Engine) {
	g := r.Group("/docs")
	g.GET("/swagger.yaml", docs.SwaggerYAMLHandler())
	g.StaticFS("/ui", docs.FS)
}

func RegisterAPIv1(r *gin.Engine) {
	g := r.Group("/v1")
	g.Use(middleware.CORS, middleware.GIL)

	// ==== legacy ====
	g.POST("/completions", func(c *gin.Context) {
		c.JSON(http.StatusGone, map[string]any{"error": "this endpoint is deprecated, please use /chat/completions instead"})
	})

	// ==== openai compatible ====
	g.POST("/chat/completions", handler.ChatCompletions)
	g.POST("/embeddings", handler.Embeddings)
	g.POST("/audio/speech", handler.Speech)
	g.POST("/audio/transcriptions", handler.Transcriptions)
	g.POST("/images/generations", handler.ImageGenerations)
	// ==== geniex specific ====
	g.POST("/audio/diarize", handler.Diarize)
	g.POST("/reranking", handler.Reranking)
	g.POST("/cv", handler.CV)

	// ==== model management ====
	g.GET("/models/*model", handler.RetrieveModel)
	g.GET("/models", handler.ListModels)
}
