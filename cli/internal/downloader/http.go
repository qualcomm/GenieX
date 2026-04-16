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

package downloader

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"time"

	"github.com/valyala/fasthttp"
)

type HTTPDownloader struct {
	authToken    string
	maxRedirects int

	maxRetries   int
	retryDelayMs int

	fasthttp.Client
}

func NewDownloader(authToken string) *HTTPDownloader {
	return &HTTPDownloader{
		authToken:    authToken,
		maxRedirects: 3,
		maxRetries:   3,
		retryDelayMs: 1000,
		Client: fasthttp.Client{
			NoDefaultUserAgentHeader:  true,
			MaxIdemponentCallAttempts: 3,
			ReadBufferSize:            64 * 1024,
			WriteBufferSize:           64 * 1024,
		},
	}
}

func (d *HTTPDownloader) DownloadChunk(ctx context.Context, url string, offset, limit int64, writer io.Writer) error {
	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	for range d.maxRedirects {
		req.Reset()
		resp.Reset()

		req.SetRequestURI(url)
		req.Header.SetMethod(fasthttp.MethodGet)
		req.Header.Set("User-Agent", "GenieX-CLI/0.0")
		if d.authToken != "" {
			req.Header.Set("Authorization", "Bearer "+d.authToken)
		}
		if limit > 0 {
			req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", offset, offset+limit-1))
		} else {
			req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
		}

		var lastErr error
		baseDelay := d.retryDelayMs
		for retry := 0; retry < d.maxRetries; retry++ {
			if retry > 0 {
				select {
				case <-ctx.Done():
					return ctx.Err()
				default:
				}
				time.Sleep(time.Duration(baseDelay) * time.Millisecond)
			}
			if err := d.Client.Do(req, resp); err != nil {
				lastErr = err
				if errors.Is(err, fasthttp.ErrTimeout) || errors.Is(err, io.EOF) {
					slog.Warn("Request failed, retrying", "error", err, "retry", retry+1)
					continue
				}
				// Other errors are returned directly
				return err
			} else {
				lastErr = nil
				break
			}
		}
		if lastErr != nil {
			return fmt.Errorf("download failed after %d retries: %w", d.maxRetries, lastErr)
		}

		if resp.StatusCode() >= 300 && resp.StatusCode() < 400 {
			location := resp.Header.Peek("Location")
			if len(location) == 0 {
				return fmt.Errorf("redirect status %d with no Location", resp.StatusCode())
			}
			url = resolveRelativeURL(url, string(location))
			continue
		}

		if resp.StatusCode() != fasthttp.StatusPartialContent && resp.StatusCode() != fasthttp.StatusOK {
			return fmt.Errorf("unexpected status code: %d", resp.StatusCode())
		}

		_, err := writer.Write(resp.Body())
		if err != nil {
			return err
		}
		return nil
	}

	return fmt.Errorf("exceeded max redirects (%d)", d.maxRedirects)
}

func resolveRelativeURL(base, location string) string {
	u, err := url.Parse(location)
	if err != nil {
		return location
	}
	if u.IsAbs() {
		return location
	}

	baseURL, err := url.Parse(base)
	if err != nil {
		return location
	}
	return baseURL.ResolveReference(u).String()
}
