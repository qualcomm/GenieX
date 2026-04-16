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

package docs

import (
	"embed"
	"io/fs"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
)

//go:embed ui/*
var StaticFiles embed.FS

//go:embed swagger.yaml
var StaticYAML []byte

type embedFileSystem struct {
	http.FileSystem
}

func (e embedFileSystem) Exists(prefix string, path string) bool {
	if path != "/" {
		path = strings.TrimSuffix(path, "/")
	}
	_, err := e.Open(path)
	return err == nil
}

func getSwaggerSubFS() fs.FS {
	sub, _ := fs.Sub(StaticFiles, "ui")
	return sub
}

var FS = embedFileSystem{http.FS(getSwaggerSubFS())}

func SwaggerYAMLHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Content-Type", "application/x-yaml")
		c.Data(http.StatusOK, "application/x-yaml", StaticYAML)
	}
}
