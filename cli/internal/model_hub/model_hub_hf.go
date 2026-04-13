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
	"net/http"
	"sync"
	"time"

	"github.com/bytedance/sonic"
	"golang.org/x/sync/errgroup"
	"resty.dev/v3"

	"github.com/qcom-it-nexa-ai/geniex/internal/config"
	"github.com/qcom-it-nexa-ai/geniex/internal/downloader"
	"github.com/qcom-it-nexa-ai/geniex/internal/render"
)

const HF_ENDPOINT = "https://huggingface.co"

type HuggingFace struct {
	downloader *downloader.HTTPDownloader
}

func NewHuggingFace() *HuggingFace {
	return &HuggingFace{downloader: downloader.NewDownloader(config.Get().HFToken)}
}

func (d *HuggingFace) ChinaMainlandOnly() bool {
	return false
}

func (d *HuggingFace) MaxConcurrency() int {
	if config.Get().HFToken != "" {
		return 8
	} else {
		fmt.Println(render.GetTheme().Warning.Sprintf("GENIEX_HFTOKEN not set. Set it for speeding up downloads from https://huggingface.co/settings/tokens"))
		return 1
	}
}

func (d *HuggingFace) CheckAvailable(ctx context.Context, name string) error {
	return nil
}

func (d *HuggingFace) ModelInfo(ctx context.Context, name string) ([]ModelFileInfo, error) {
	info := struct {
		Siblings []struct {
			RFileName string `json:"rfilename"`
		} `json:"siblings"`
	}{}

	client := resty.New()
	defer client.Close()
	client.SetTimeout(10 * time.Second)
	client.AddResponseMiddleware(code2error)

	req := client.R()
	if config.Get().HFToken != "" {
		req.SetHeader("Authorization", "Bearer "+config.Get().HFToken)
	}
	resp, err := req.Get(fmt.Sprintf("%s/api/models/%s", HF_ENDPOINT, name))
	if err != nil {
		return nil, err
	}

	if err := sonic.UnmarshalString(resp.String(), &info); err != nil {
		return nil, err
	}

	res := make([]ModelFileInfo, len(info.Siblings))
	var resLock sync.Mutex

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(d.MaxConcurrency())

	for i := range info.Siblings {
		i := i
		g.Go(func() error {
			req := client.R()
			if config.Get().HFToken != "" {
				req.SetHeader("Authorization", "Bearer "+config.Get().HFToken)
			}
			req.SetHeader("Accept-Encoding", "identity")

			resp, err := req.SetContext(gctx).Head(fmt.Sprintf("%s/%s/resolve/main/%s", HF_ENDPOINT, name, info.Siblings[i].RFileName))
			if err != nil {
				return err
			}
			if resp.StatusCode() != http.StatusOK || resp.RawResponse.ContentLength < 0 {
				return fmt.Errorf("Get file [%s] info error: %s", info.Siblings[i].RFileName, resp.Status())
			}

			resLock.Lock()
			res[i] = ModelFileInfo{
				Name: info.Siblings[i].RFileName,
				Size: resp.RawResponse.ContentLength,
			}
			resLock.Unlock()

			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}

	return res, nil
}

func (d *HuggingFace) GetFileContent(ctx context.Context, modelName, fileName string, offset, limit int64, writer io.Writer) error {
	return d.downloader.DownloadChunk(ctx, fmt.Sprintf("%s/%s/resolve/main/%s", HF_ENDPOINT, modelName, fileName), offset, limit, writer)
}
