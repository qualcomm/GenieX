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

package model_hub

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"time"

	"github.com/bytedance/sonic"
	"resty.dev/v3"

	"github.com/qcom-it-nexa-ai/geniex/internal/downloader"
)

const MS_ENDPOINT = "https://modelscope.cn"

type ModelScope struct {
	downloader *downloader.HTTPDownloader
}

func NewModelScope() *ModelScope {
	return &ModelScope{downloader: downloader.NewDownloader("")}
}

func (d *ModelScope) ChinaMainlandOnly() bool {
	return true
}

func (d *ModelScope) MaxConcurrency() int {
	return 8
}

func (d *ModelScope) CheckAvailable(ctx context.Context, name string) error {
	client := resty.New()
	defer client.Close()
	client.SetTimeout(5 * time.Second)
	client.AddResponseMiddleware(code2error)

	_, err := client.R().Get(fmt.Sprintf("%s/api/v1/models/%s/revisions", MS_ENDPOINT, name))
	if err != nil {
		return err
	}

	return nil
}

func (d *ModelScope) ModelInfo(ctx context.Context, name string) ([]ModelFileInfo, error) {
	return d.modelInfo(ctx, name, "")
}

func (d *ModelScope) modelInfo(ctx context.Context, name string, root string) ([]ModelFileInfo, error) {
	info := struct {
		Data struct {
			Files []struct {
				Path string
				Size int64
				Type string
			}
		}
	}{}

	client := resty.New()
	defer client.Close()
	client.SetTimeout(10 * time.Second)
	client.AddResponseMiddleware(code2error)

	resp, err := client.R().
		Get(fmt.Sprintf("%s/api/v1/models/%s/repo/files?Root=%s", MS_ENDPOINT, name, root))
	if err != nil {
		return nil, err
	}

	if err := sonic.UnmarshalString(resp.String(), &info); err != nil {
		return nil, err
	}

	res := make([]ModelFileInfo, 0)
	for _, file := range info.Data.Files {
		// blob, tree
		switch file.Type {
		case "tree":
			subFiles, err := d.modelInfo(ctx, name, file.Path)
			if err != nil {
				return nil, err
			}
			res = append(res, subFiles...)
		case "blob":
			res = append(res, ModelFileInfo{
				Name: file.Path,
				Size: file.Size,
			})
		default:
			slog.Warn("modelscope unknown file type", "model", name, "file", file.Path, "type", file.Type)
			continue
		}
	}
	return res, nil
}

func (d *ModelScope) GetFileContent(ctx context.Context, modelName, fileName string, offset, limit int64, writer io.Writer) error {
	return d.downloader.DownloadChunk(ctx, fmt.Sprintf("%s/models/%s/resolve/main/%s", MS_ENDPOINT, modelName, fileName), offset, limit, writer)
}
