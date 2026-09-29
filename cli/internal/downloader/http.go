// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package downloader

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"time"

	"github.com/valyala/fasthttp"
	"github.com/valyala/fasthttp/fasthttpproxy"
)

type HTTPDownloader struct {
	// AuthToken is sent as a Bearer token and stripped when a redirect crosses
	// to a different host (presigned S3/Azure URLs 400/404 on unexpected auth).
	AuthToken string
	// Headers are applied to every request and follow all redirects — use for
	// static, non-sensitive headers like Accept.
	Headers map[string]string

	maxRedirects int
	maxRetries   int
	retryDelayMs int

	fasthttp.Client
}

func NewDownloader() *HTTPDownloader {
	return &HTTPDownloader{
		maxRedirects: 3,
		maxRetries:   3,
		retryDelayMs: 1000,
		Client: fasthttp.Client{
			NoDefaultUserAgentHeader:  true,
			MaxIdemponentCallAttempts: 3,
			ReadBufferSize:            64 * 1024,
			WriteBufferSize:           64 * 1024,
			// Respect HTTP_PROXY / HTTPS_PROXY / NO_PROXY (and lowercase variants).
			Dial: fasthttpproxy.FasthttpProxyHTTPDialerTimeout(10 * time.Second),
		},
	}
}

// sameHost reports whether two URLs share the same host (case-insensitive).
// Used to decide whether Authorization headers should survive a redirect.
func sameHost(a, b string) bool {
	ua, err := url.Parse(a)
	if err != nil {
		return false
	}
	ub, err := url.Parse(b)
	if err != nil {
		return false
	}
	return ua.Hostname() == ub.Hostname()
}

func (d *HTTPDownloader) DownloadChunk(ctx context.Context, reqURL string, offset, limit int64, writer io.Writer) error {
	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	originalURL := reqURL
	for range d.maxRedirects {
		req.Reset()
		resp.Reset()

		req.SetRequestURI(reqURL)
		req.Header.SetMethod(fasthttp.MethodGet)
		req.Header.Set("User-Agent", "GenieX-CLI/0.0")
		for k, v := range d.Headers {
			req.Header.Set(k, v)
		}
		// Strip Authorization when following a cross-host redirect: presigned
		// S3/Azure URLs 400/404 on unexpected auth headers.
		if d.AuthToken != "" && sameHost(reqURL, originalURL) {
			req.Header.Set("Authorization", "Bearer "+d.AuthToken)
		}
		if limit > 0 {
			req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", offset, offset+limit-1))
		} else {
			req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
		}

		var lastErr error
		baseDelay := d.retryDelayMs
		for retry := 0; retry < d.maxRetries; retry++ {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}
			if retry > 0 {
				time.Sleep(time.Duration(baseDelay) * time.Millisecond)
			}
			resp.Reset()
			if err := d.Client.Do(req, resp); err != nil {
				lastErr = err
				slog.Warn("Request failed, retrying", "error", err, "retry", retry+1)
				continue
			}
			lastErr = nil
			break
		}
		if lastErr != nil {
			return fmt.Errorf("download failed after %d retries: %w", d.maxRetries, lastErr)
		}

		if resp.StatusCode() >= 300 && resp.StatusCode() < 400 {
			location := resp.Header.Peek("Location")
			if len(location) == 0 {
				return fmt.Errorf("redirect status %d with no Location", resp.StatusCode())
			}
			reqURL = resolveRelativeURL(reqURL, string(location))
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
