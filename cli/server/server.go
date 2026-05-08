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
	"fmt"
	"os"

	"github.com/gin-gonic/gin"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/config"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/render"
	"github.com/qcom-it-nexa-ai/geniex/cli/server/service"
)

// @Title		GenieX Server
// @Version	0.0.0
// @BasePath	/v1
func Serve() {
	service.Init()
	defer service.DeInit()

	gin.SetMode(gin.ReleaseMode)
	engine := gin.Default()

	RegisterRoot(engine)
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
		fmt.Println(render.GetTheme().Info.Sprintf("Local hosting on https://%s/docs/ui", cfg.Host))
		err = engine.RunTLS(cfg.Host, certFile, keyFile)
	} else {
		fmt.Println(render.GetTheme().Info.Sprintf("Local hosting on http://%s/docs/ui", cfg.Host))
		err = engine.Run(cfg.Host)
	}

	if err != nil {
		fmt.Println(render.GetTheme().Error.Sprintf("HTTP/HTTPS Server Error: %v", err))
	}
}
