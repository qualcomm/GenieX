// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package downloader

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
)

func TestSameHost(t *testing.T) {
	cases := []struct {
		a, b string
		want bool
	}{
		{"https://example.com/a", "https://example.com/b", true},
		{"https://example.com:443/a", "https://example.com/b", true}, // port ignored
		{"https://example.com/a", "https://OTHER.com/b", false},
		{"https://example.com/a", "", false},
		{"::bad::", "https://example.com", false},
	}
	for _, c := range cases {
		if got := sameHost(c.a, c.b); got != c.want {
			t.Errorf("sameHost(%q,%q) = %v, want %v", c.a, c.b, got, c.want)
		}
	}
}

// redirectServer returns a pair (originServer, targetServer) where origin
// 302-redirects to target. The target records whether Authorization was seen.
func redirectServer(t *testing.T, targetHost string) (origin *httptest.Server, target *httptest.Server, authSeen *atomic.Bool) {
	t.Helper()
	authSeen = &atomic.Bool{}

	target = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			authSeen.Store(true)
		}
		w.Header().Set("Content-Range", fmt.Sprintf("bytes 0-4/%d", 5))
		w.WriteHeader(http.StatusPartialContent)
		w.Write([]byte("hello"))
	}))

	origin = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Rewrite the target URL to use targetHost (e.g. "localhost") so it
		// differs from the origin's 127.0.0.1 and triggers cross-host logic.
		loc := target.URL
		if targetHost != "" {
			loc = strings.Replace(target.URL, "127.0.0.1", targetHost, 1)
		}
		http.Redirect(w, r, loc, http.StatusFound)
	}))
	return
}

func TestDownloadChunk_CrossHostRedirectStripsAuth(t *testing.T) {
	// origin = 127.0.0.1, target = localhost → different Hostname → strip.
	origin, target, authSeen := redirectServer(t, "localhost")
	defer origin.Close()
	defer target.Close()

	d := NewDownloader()
	d.AuthToken = "secret-token"
	var buf bytes.Buffer
	if err := d.DownloadChunk(context.Background(), origin.URL, 0, 5, &buf); err != nil {
		t.Fatalf("DownloadChunk: %v", err)
	}
	if buf.String() != "hello" {
		t.Errorf("body = %q, want %q", buf.String(), "hello")
	}
	if authSeen.Load() {
		t.Error("Authorization header leaked to cross-host redirect target")
	}
}

func TestDownloadChunk_SameHostRedirectPreservesAuth(t *testing.T) {
	// origin and target share hostname (127.0.0.1) — only ports differ.
	origin, target, authSeen := redirectServer(t, "")
	defer origin.Close()
	defer target.Close()

	d := NewDownloader()
	d.AuthToken = "secret-token"
	var buf bytes.Buffer
	if err := d.DownloadChunk(context.Background(), origin.URL, 0, 5, &buf); err != nil {
		t.Fatalf("DownloadChunk: %v", err)
	}
	if !authSeen.Load() {
		t.Error("Authorization header was dropped on same-host redirect")
	}
}

func TestDownloadChunk_RetriesOnNonTimeoutError(t *testing.T) {
	var attempts atomic.Int32
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if attempts.Add(1) == 1 {
			hj := w.(http.Hijacker)
			conn, _, err := hj.Hijack()
			if err != nil {
				t.Errorf("hijack: %v", err)
				return
			}
			defer conn.Close()
			conn.Write([]byte("HTTP/1.1 200 OK\r\nContent-Length: bogus\r\n\r\nhello"))
			return
		}
		w.Header().Set("Content-Range", "bytes 0-4/5")
		w.WriteHeader(http.StatusPartialContent)
		w.Write([]byte("hello"))
	}))
	defer backend.Close()

	d := NewDownloader()
	d.retryDelayMs = 10
	var buf bytes.Buffer
	if err := d.DownloadChunk(context.Background(), backend.URL, 0, 5, &buf); err != nil {
		t.Fatalf("DownloadChunk: %v", err)
	}
	if buf.String() != "hello" {
		t.Errorf("body = %q, want %q", buf.String(), "hello")
	}
	if got := attempts.Load(); got < 2 {
		t.Errorf("attempts = %d, want >= 2 (should have retried after the malformed response)", got)
	}
}

func TestDownloadChunk_HonorsHTTPProxyEnv(t *testing.T) {
	// Backend serves the bytes the downloader expects.
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Range", "bytes 0-4/5")
		w.WriteHeader(http.StatusPartialContent)
		w.Write([]byte("proxy"))
	}))
	defer backend.Close()

	backendURL, _ := url.Parse(backend.URL)

	// fasthttp's proxy dialer always issues CONNECT, even for plain-HTTP
	// targets. Stand up a minimal CONNECT tunnel that pipes bytes to the
	// backend regardless of the authority the client requests.
	var proxied atomic.Bool
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodConnect {
			http.Error(w, "expected CONNECT", http.StatusMethodNotAllowed)
			return
		}
		proxied.Store(true)

		upstream, err := net.Dial("tcp", backendURL.Host)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		hj, ok := w.(http.Hijacker)
		if !ok {
			http.Error(w, "no hijacker", http.StatusInternalServerError)
			return
		}
		client, _, err := hj.Hijack()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer client.Close()
		defer upstream.Close()

		fmt.Fprint(client, "HTTP/1.1 200 Connection established\r\n\r\n")
		go io.Copy(upstream, client)
		io.Copy(client, upstream)
	}))
	defer proxy.Close()

	t.Setenv("HTTP_PROXY", proxy.URL)
	t.Setenv("NO_PROXY", "")

	d := NewDownloader()
	// Target must not be a loopback host — httpproxy bypasses the proxy for
	// localhost / 127.0.0.0/8. example.com is public, but we never actually
	// resolve it because the CONNECT tunnel above routes every request to
	// `backend` regardless of the requested authority.
	var buf bytes.Buffer
	if err := d.DownloadChunk(context.Background(), "http://example.com/file", 0, 5, &buf); err != nil {
		t.Fatalf("DownloadChunk via proxy: %v", err)
	}
	if !proxied.Load() {
		t.Fatal("request did not traverse HTTP_PROXY")
	}
	if buf.String() != "proxy" {
		t.Errorf("body = %q, want %q", buf.String(), "proxy")
	}
}
