// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package server

import (
	"fmt"
	"net"
	"os"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/qualcomm/GenieX/cli/internal/config"
	"github.com/qualcomm/GenieX/cli/internal/render"
	"github.com/qualcomm/GenieX/cli/server/service"
)

// hostBindingHint suggests --host 0.0.0.0 when bound to loopback, else "".
// Malformed hosts are treated as non-loopback — better silent than misleading.
func hostBindingHint(host string) string {
	h, port, err := net.SplitHostPort(host)
	if err != nil {
		return ""
	}
	loopback := strings.EqualFold(h, "localhost")
	if ip := net.ParseIP(h); ip != nil && ip.IsLoopback() {
		loopback = true
	}
	if !loopback {
		return ""
	}
	return fmt.Sprintf("Bound to loopback only. To expose on your network, restart with --host 0.0.0.0:%s", port)
}

// @Title		GenieX Server
// @Version	0.0.0
// @BasePath	/v1
func Serve() {
	service.Init()
	defer service.DeInit()

	gin.SetMode(gin.ReleaseMode)
	engine := gin.Default()

	if err := RegisterRoot(engine); err != nil {
		fmt.Println(render.GetTheme().Error.Sprintf("Web UI configuration error: %v", err))
		return
	}
	RegisterAPIv1(engine)
	RegisterSwagger(engine)

	cfg := config.Get()
	var err error

	// Determine whether to serve over HTTPS
	if cfg.HTTPS {
		certFile := cfg.CertFile
		keyFile := cfg.KeyFile

		// Verify that certificate and key files exist
		if _, err := os.Stat(certFile); os.IsNotExist(err) {
			fmt.Println(render.GetTheme().Error.Sprintf("HTTPS Certificate file not found: %s", certFile))
			return
		}
		if _, err := os.Stat(keyFile); os.IsNotExist(err) {
			fmt.Println(render.GetTheme().Error.Sprintf("HTTPS Key file not found: %s", keyFile))
			return
		}

		fmt.Println(render.GetTheme().Info.Sprintf("HTTPS enabled: cert=%s key=%s", certFile, keyFile))
		fmt.Println(render.GetTheme().Info.Sprintf("Local hosting on https://%s/", cfg.Host))
		if hint := hostBindingHint(cfg.Host); hint != "" {
			fmt.Println(render.GetTheme().Info.Sprint(hint))
		}
		err = engine.RunTLS(cfg.Host, certFile, keyFile)
	} else {
		fmt.Println(render.GetTheme().Info.Sprintf("Local hosting on http://%s/", cfg.Host))
		if hint := hostBindingHint(cfg.Host); hint != "" {
			fmt.Println(render.GetTheme().Info.Sprint(hint))
		}
		err = engine.Run(cfg.Host)
	}

	if err != nil {
		fmt.Println(render.GetTheme().Error.Sprintf("HTTP/HTTPS Server Error: %v", err))
	}
}
