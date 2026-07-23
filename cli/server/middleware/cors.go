// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package middleware

import (
	"github.com/gin-gonic/gin"

	"github.com/qualcomm/GenieX/cli/internal/config"
)

func CORS(c *gin.Context) {
	h := c.Writer.Header()
	h.Set("Access-Control-Allow-Origin", config.Get().Origins)
	h.Set("Access-Control-Allow-Methods", "OPTIONS, GET, POST")
	h.Set("Access-Control-Allow-Headers", "Authorization, Content-Type, GenieX-KeepCache")
	h.Set("Access-Control-Max-Age", "86400")

	if c.Request.Method == "OPTIONS" {
		c.AbortWithStatus(204)
		return
	}

	c.Next()
}
