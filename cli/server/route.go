// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package server

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/qualcomm/GenieX/cli/server/docs"
	"github.com/qualcomm/GenieX/cli/server/handler"
	"github.com/qualcomm/GenieX/cli/server/middleware"
)

func RegisterRoot(r *gin.Engine) {
	r.Use(middleware.CORS)
	r.GET("/", func(c *gin.Context) {
		c.Redirect(http.StatusFound, "/docs/ui/")
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
	g.GET("/", func(c *gin.Context) {
		c.String(200, "GenieX-CLI is running")
	})

	g.Use(middleware.CORS, middleware.GIL)

	// ==== openai compatible ====
	g.POST("/completions", handler.Completions)
	g.POST("/chat/completions", handler.ChatCompletions)

	// ==== raw logits (prefill-only forward pass; not OpenAI generative logprobs) ====
	g.POST("/logits", handler.ForwardLogits)

	// ==== model management ====
	g.GET("/models/*model", handler.RetrieveModel)
	g.GET("/models", handler.ListModels)
}
