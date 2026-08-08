// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package server

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/qualcomm/GenieX/cli/internal/config"
	"github.com/qualcomm/GenieX/cli/server/docs"
	"github.com/qualcomm/GenieX/cli/server/handler"
	"github.com/qualcomm/GenieX/cli/server/middleware"
)

func RegisterRoot(r *gin.Engine) error {
	r.Use(middleware.CORS)
	webuiDir := config.Get().WebUIDir
	if webuiDir == "" {
		r.GET("/", func(c *gin.Context) {
			c.Redirect(http.StatusFound, "/docs/ui/")
		})
		return nil
	}

	root, err := filepath.Abs(webuiDir)
	if err != nil {
		return fmt.Errorf("resolve %q: %w", webuiDir, err)
	}
	index := filepath.Join(root, "index.html")
	info, err := os.Stat(index)
	if err != nil || !info.Mode().IsRegular() {
		return fmt.Errorf("llama.cpp Web UI index not found: %s", index)
	}

	r.GET("/", func(c *gin.Context) {
		c.File(index)
	})
	r.GET("/props", WebUIProps)
	r.HEAD("/props", WebUIProps)
	r.GET("/health", func(c *gin.Context) { c.String(http.StatusOK, "ok") })
	r.GET("/slots", webUIDisabledFeature)
	r.GET("/tools", webUIDisabledFeature)
	r.POST("/tools", webUIDisabledFeature)

	// Preserve the upstream build exactly as generated. Unknown browser routes
	// fall back to index.html for the Svelte SPA, while unknown API routes keep
	// a real 404 instead of accidentally returning HTML.
	r.NoRoute(func(c *gin.Context) {
		if c.Request.Method != http.MethodGet && c.Request.Method != http.MethodHead {
			c.Status(http.StatusNotFound)
			return
		}
		path := c.Request.URL.Path
		if strings.HasPrefix(path, "/v1/") || strings.HasPrefix(path, "/docs/") {
			c.Status(http.StatusNotFound)
			return
		}

		rel := strings.TrimPrefix(filepath.Clean("/"+path), string(filepath.Separator))
		candidate := filepath.Join(root, rel)
		withinRoot, relErr := filepath.Rel(root, candidate)
		if relErr == nil && withinRoot != ".." && !strings.HasPrefix(withinRoot, ".."+string(filepath.Separator)) {
			if asset, statErr := os.Stat(candidate); statErr == nil && asset.Mode().IsRegular() {
				c.File(candidate)
				return
			}
		}
		c.File(index)
	})
	return nil
}

func webUIDisabledFeature(c *gin.Context) {
	c.JSON(http.StatusForbidden, map[string]any{"error": "this feature is disabled"})
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

	// ==== legacy ====
	g.POST("/completions", func(c *gin.Context) {
		c.JSON(http.StatusGone, map[string]any{"error": "this endpoint is deprecated, please use /chat/completions instead"})
	})

	// ==== openai compatible ====
	g.POST("/chat/completions", handler.ChatCompletions)

	// ==== raw logits (prefill-only forward pass; not OpenAI generative logprobs) ====
	g.POST("/logits", handler.ForwardLogits)

	// ==== model management ====
	g.GET("/models/*model", handler.RetrieveModel)
	g.GET("/models", handler.ListModels)
}
