// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// Package browser contains the narrow CDP implementation used by JEV.
package browser

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/qualcomm/GenieX/cli/internal/jev"
	"golang.org/x/net/websocket"
)

// Config controls a Chrome/Edge CDP session. BrowserPath is optional when
// AttachURL is set; otherwise a platform default is selected.
type Config struct {
	Browser     string
	BrowserPath string
	AttachURL   string
	ProfileDir  string
	Headless    bool
	InitialURL  string
	TraceDir    string
}

// Browser is a single-page CDP client. It never exposes arbitrary CDP calls to
// the model: only the action implementation below dispatches browser events.
type Browser struct {
	ws                   *websocket.Conn
	process              *os.Process
	processDone          <-chan struct{}
	profileDir           string
	removeDir            bool
	traceDir             string
	observationDir       string
	removeObservationDir bool
	mu                   sync.Mutex
	nextID               int
}

type versionInfo struct {
	WebSocketDebuggerURL string `json:"webSocketDebuggerUrl"`
}

type targetInfo struct {
	Type                 string `json:"type"`
	URL                  string `json:"url"`
	WebSocketDebuggerURL string `json:"webSocketDebuggerUrl"`
}

type rpcMessage struct {
	ID     int             `json:"id,omitempty"`
	Method string          `json:"method,omitempty"`
	Params json.RawMessage `json:"params,omitempty"`
	Result json.RawMessage `json:"result,omitempty"`
	Error  *rpcError       `json:"error,omitempty"`
}

type rpcError struct {
	Message string `json:"message"`
}

const cdpClientOrigin = "http://127.0.0.1"

// Open attaches to an existing browser or launches a clean isolated instance.
func Open(ctx context.Context, config Config) (*Browser, error) {
	endpoint := config.AttachURL
	if endpoint != "" {
		if err := validateAttachURL(endpoint); err != nil {
			return nil, err
		}
	}
	browser := &Browser{traceDir: config.TraceDir}
	if endpoint == "" {
		port, err := freePort()
		if err != nil {
			return nil, err
		}
		profileDir := config.ProfileDir
		if profileDir == "" {
			profileDir, err = os.MkdirTemp("", "geniex-jev-profile-")
			if err != nil {
				return nil, fmt.Errorf("create browser profile: %w", err)
			}
			browser.removeDir = true
		}
		browser.profileDir = profileDir
		path := config.BrowserPath
		if path == "" {
			path, err = defaultBrowserPath(config.Browser)
			if err != nil {
				browser.Close()
				return nil, err
			}
		}
		process, done, err := startBrowser(ctx, path, launchArgs(port, profileDir, config.Headless, config.InitialURL)...)
		if err != nil {
			browser.Close()
			return nil, fmt.Errorf("start browser %q: %w", path, err)
		}
		browser.process = process
		browser.processDone = done
		endpoint = fmt.Sprintf("http://127.0.0.1:%d", port)
	}

	pageURL, err := waitForPage(ctx, endpoint, config.AttachURL != "", config.InitialURL)
	if err != nil {
		browser.Close()
		return nil, err
	}
	if err := validateAttachURL(pageURL); err != nil {
		browser.Close()
		return nil, fmt.Errorf("invalid CDP page endpoint: %w", err)
	}
	ws, err := websocket.Dial(pageURL, "", cdpClientOrigin)
	if err != nil {
		browser.Close()
		return nil, fmt.Errorf("connect to browser CDP: %w", err)
	}
	browser.ws = ws
	return browser, nil
}

// startBrowser launches a browser executable selected by the local operator or
// defaultBrowserPath. The model and visited page never influence executablePath.
func startBrowser(ctx context.Context, executablePath string, args ...string) (*os.Process, <-chan struct{}, error) {
	attributes := &os.ProcAttr{Files: []*os.File{os.Stdin, os.Stdout, os.Stderr}}
	process, err := os.StartProcess(executablePath, append([]string{executablePath}, args...), attributes)
	if err != nil {
		return nil, nil, err
	}
	done := make(chan struct{})
	go func() {
		_, _ = process.Wait()
		close(done)
	}()
	go func() {
		select {
		case <-ctx.Done():
			_ = process.Kill()
		case <-done:
		}
	}()
	return process, done, nil
}

func launchArgs(port int, profileDir string, headless bool, initialURL string) []string {
	args := []string{
		fmt.Sprintf("--remote-debugging-port=%d", port),
		"--remote-allow-origins=" + cdpClientOrigin,
		"--no-first-run",
		"--no-default-browser-check",
		"--user-data-dir=" + profileDir,
	}
	if headless {
		args = append(args, "--headless=new")
	}
	if initialURL != "" {
		args = append(args, initialURL)
	}
	return args
}

func (b *Browser) Close() error {
	var firstErr error
	if b.ws != nil {
		if err := b.ws.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		b.ws = nil
	}
	if b.process != nil {
		if err := b.process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) && firstErr == nil {
			firstErr = err
		}
		if b.processDone != nil {
			<-b.processDone
		}
	}
	if b.removeDir && b.profileDir != "" {
		if err := os.RemoveAll(b.profileDir); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if b.removeObservationDir && b.observationDir != "" {
		if err := os.RemoveAll(b.observationDir); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

func (b *Browser) Observe(ctx context.Context) (jev.Observation, error) {
	url, title, elements, err := b.snapshot(ctx)
	if err != nil {
		return jev.Observation{}, err
	}
	fingerprint, err := jev.SnapshotFingerprint(url, title, elements)
	if err != nil {
		return jev.Observation{}, err
	}
	screenshot, err := b.call(ctx, "Page.captureScreenshot", map[string]any{"format": "png"})
	if err != nil {
		return jev.Observation{}, err
	}
	var image struct{ Data string `json:"data"` }
	if err := json.Unmarshal(screenshot, &image); err != nil {
		return jev.Observation{}, fmt.Errorf("decode browser screenshot: %w", err)
	}
	png, err := base64.StdEncoding.DecodeString(image.Data)
	if err != nil {
		return jev.Observation{}, fmt.Errorf("decode browser screenshot image: %w", err)
	}
	dir, err := b.observationOutputDir()
	if err != nil {
		return jev.Observation{}, err
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return jev.Observation{}, err
	}
	file, err := os.CreateTemp(dir, "screenshot-*.png")
	if err != nil {
		return jev.Observation{}, err
	}
	if _, err := file.Write(png); err != nil {
		file.Close()
		return jev.Observation{}, err
	}
	if err := file.Close(); err != nil {
		return jev.Observation{}, err
	}
	return jev.Observation{URL: url, Title: title, ScreenshotPath: file.Name(), Elements: elements, Fingerprint: fingerprint}, nil
}

// observationOutputDir returns a persistent caller-owned trace directory or a
// private temporary directory that Browser.Close removes.
func (b *Browser) observationOutputDir() (string, error) {
	if b.traceDir != "" {
		return b.traceDir, nil
	}
	if b.observationDir != "" {
		return b.observationDir, nil
	}
	dir, err := os.MkdirTemp("", "geniex-jev-observation-")
	if err != nil {
		return "", fmt.Errorf("create observation directory: %w", err)
	}
	b.observationDir = dir
	b.removeObservationDir = true
	return dir, nil
}

func (b *Browser) snapshot(ctx context.Context) (string, string, []jev.Element, error) {
	value, err := b.evaluate(ctx, snapshotScript)
	if err != nil {
		return "", "", nil, err
	}
	var snapshot struct {
		URL      string         `json:"url"`
		Title    string         `json:"title"`
		Elements []jev.Element `json:"elements"`
	}
	if err := json.Unmarshal(value, &snapshot); err != nil {
		return "", "", nil, fmt.Errorf("decode browser snapshot: %w", err)
	}
	return snapshot.URL, snapshot.Title, snapshot.Elements, nil
}

func (b *Browser) ensureFresh(ctx context.Context, observation jev.Observation) error {
	if observation.Fingerprint == "" {
		return fmt.Errorf("%w: observation has no fingerprint", jev.ErrStaleObservation)
	}
	url, title, elements, err := b.snapshot(ctx)
	if err != nil {
		return fmt.Errorf("refresh browser snapshot: %w", err)
	}
	fingerprint, err := jev.SnapshotFingerprint(url, title, elements)
	if err != nil {
		return err
	}
	if fingerprint != observation.Fingerprint {
		return fmt.Errorf("%w: page changed since decision", jev.ErrStaleObservation)
	}
	return nil
}

func (b *Browser) Execute(ctx context.Context, action jev.Action, observation jev.Observation) (string, error) {
	if action.Action == jev.ActionRead && action.Index == nil {
		return "", fmt.Errorf("read action requires an element index")
	}
	if err := b.ensureFresh(ctx, observation); err != nil {
		return "", err
	}
	var expression string
	switch action.Action {
	case jev.ActionNavigate:
		_, err := b.call(ctx, "Page.navigate", map[string]any{"url": action.URL})
		return "navigated", err
	case jev.ActionGoBack:
		expression = "history.back(); 'went back'"
	case jev.ActionGoForward:
		expression = "history.forward(); 'went forward'"
	case jev.ActionScroll:
		delta := action.Amount
		if action.Direction == "up" {
			delta = -delta
		}
		expression = fmt.Sprintf("window.scrollBy({top:%d,behavior:'instant'}); 'scrolled'", delta)
	case jev.ActionWait:
		expression = fmt.Sprintf("new Promise(resolve => setTimeout(() => resolve('waited'), %d))", action.Milliseconds)
	case jev.ActionRead:
		if action.Index == nil {
			return "", fmt.Errorf("read action requires an element index")
		}
		expression = targetExpression(*action.Index, "el => el.innerText || (el.type === 'password' ? '<redacted>' : el.value) || el.getAttribute('aria-label') || 'read'")
	case jev.ActionClick:
		expression = targetExpression(*action.Index, "el => { el.click(); return 'clicked'; }")
	case jev.ActionType:
		expression = targetExpression(*action.Index, fmt.Sprintf("el => { el.focus(); el.value = %s; el.dispatchEvent(new Event('input', {bubbles:true})); el.dispatchEvent(new Event('change', {bubbles:true})); %s return 'typed'; }", jsString(action.Text), submitExpression(action.Submit)))
	default:
		return "", fmt.Errorf("browser cannot execute %q", action.Action)
	}
	value, err := b.evaluate(ctx, expression)
	if err != nil {
		return "", err
	}
	var outcome string
	if err := json.Unmarshal(value, &outcome); err != nil {
		return "", fmt.Errorf("decode browser action outcome: %w", err)
	}
	return outcome, nil
}

func (b *Browser) evaluate(ctx context.Context, expression string) (json.RawMessage, error) {
	result, err := b.call(ctx, "Runtime.evaluate", map[string]any{"expression": expression, "returnByValue": true, "awaitPromise": true})
	if err != nil {
		return nil, err
	}
	return decodeEvaluateResult(result)
}

func decodeEvaluateResult(raw json.RawMessage) (json.RawMessage, error) {
	var result struct {
		Result struct {
			Value json.RawMessage `json:"value"`
		} `json:"result"`
		ExceptionDetails *struct {
			Text      string `json:"text"`
			Exception *struct {
				Description string `json:"description"`
			} `json:"exception"`
		} `json:"exceptionDetails"`
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("decode Runtime.evaluate result: %w", err)
	}
	if result.ExceptionDetails == nil {
		return result.Result.Value, nil
	}

	message := result.ExceptionDetails.Text
	if result.ExceptionDetails.Exception != nil && result.ExceptionDetails.Exception.Description != "" {
		message = result.ExceptionDetails.Exception.Description
	}
	if strings.Contains(message, "target is stale") || strings.Contains(message, "target is occluded") {
		return nil, fmt.Errorf("%w: %s", jev.ErrStaleObservation, message)
	}
	return nil, fmt.Errorf("Runtime.evaluate: %s", message)
}

func (b *Browser) call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.ws == nil {
		return nil, fmt.Errorf("browser CDP connection is closed")
	}
	b.nextID++
	payload, err := json.Marshal(struct {
		ID     int    `json:"id"`
		Method string `json:"method"`
		Params any    `json:"params,omitempty"`
	}{b.nextID, method, params})
	if err != nil {
		return nil, err
	}
	if err := websocket.Message.Send(b.ws, string(payload)); err != nil {
		return nil, fmt.Errorf("send CDP %s: %w", method, err)
	}
	deadline := time.Now().Add(30 * time.Second)
	if err := b.ws.SetDeadline(deadline); err != nil {
		return nil, err
	}
	defer b.ws.SetDeadline(time.Time{})
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		var raw string
		if err := websocket.Message.Receive(b.ws, &raw); err != nil {
			return nil, fmt.Errorf("receive CDP %s: %w", method, err)
		}
		var message rpcMessage
		if err := json.Unmarshal([]byte(raw), &message); err != nil {
			return nil, fmt.Errorf("decode CDP response: %w", err)
		}
		if message.ID != b.nextID {
			continue // ignore page events and unrelated messages
		}
		if message.Error != nil {
			return nil, fmt.Errorf("CDP %s: %s", method, message.Error.Message)
		}
		return message.Result, nil
	}
}

func validateAttachURL(raw string) error {
	endpoint, err := url.ParseRequestURI(raw)
	if err != nil || !endpoint.IsAbs() || endpoint.Host == "" || endpoint.User != nil {
		return fmt.Errorf("attach endpoint must be an absolute loopback URL")
	}
	switch endpoint.Scheme {
	case "http", "https", "ws", "wss":
	default:
		return fmt.Errorf("attach endpoint scheme must be http, https, ws, or wss")
	}
	host := endpoint.Hostname()
	if strings.EqualFold(host, "localhost") {
		return nil
	}
	address := net.ParseIP(host)
	if address == nil || !address.IsLoopback() {
		return fmt.Errorf("attach endpoint host must be loopback")
	}
	return nil
}

func waitForPage(ctx context.Context, endpoint string, attached bool, initialURL string) (string, error) {
	endpoint = strings.TrimSuffix(endpoint, "/")
	parsed, err := url.ParseRequestURI(endpoint)
	if err != nil {
		return "", fmt.Errorf("invalid CDP endpoint %q: %w", endpoint, err)
	}
	if parsed.Scheme == "ws" || parsed.Scheme == "wss" {
		return endpoint, nil
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", fmt.Errorf("invalid CDP endpoint scheme %q", parsed.Scheme)
	}
	client := &http.Client{
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	for {
		request, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint+"/json/list", nil)
		if err == nil {
			response, requestErr := client.Do(request)
			if requestErr == nil {
				body, readErr := io.ReadAll(response.Body)
				response.Body.Close()
				if readErr == nil && response.StatusCode == http.StatusOK {
					var targets []targetInfo
					if json.Unmarshal(body, &targets) == nil {
						if pageURL := selectPageTarget(targets, attached, initialURL); pageURL != "" {
							return pageURL, nil
						}
					}
				}
			}
		}
		select {
		case <-ctx.Done():
			if initialURL != "" && !attached {
				return "", fmt.Errorf("wait for browser to open initial URL %q: %w", initialURL, ctx.Err())
			}
			return "", fmt.Errorf("wait for CDP browser: %w", ctx.Err())
		case <-time.After(150 * time.Millisecond):
		}
	}
}

// selectPageTarget avoids attaching a newly launched session to Chrome's
// transient about:blank target. Existing attached sessions retain the caller's
// prior target selection semantics.
func selectPageTarget(targets []targetInfo, attached bool, initialURL string) string {
	var firstPage string
	var loadedPage string
	for _, target := range targets {
		if target.Type != "page" || target.WebSocketDebuggerURL == "" {
			continue
		}
		if firstPage == "" {
			firstPage = target.WebSocketDebuggerURL
		}
		if !attached && initialURL != "" && target.URL == initialURL {
			return target.WebSocketDebuggerURL
		}
		if loadedPage == "" && isHTTPURL(target.URL) {
			loadedPage = target.WebSocketDebuggerURL
		}
	}
	if attached || initialURL == "" {
		return firstPage
	}
	return loadedPage
}

func isHTTPURL(value string) bool {
	parsed, err := url.Parse(value)
	return err == nil && (parsed.Scheme == "http" || parsed.Scheme == "https") && parsed.Host != ""
}

func freePort() (int, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port, nil
}

func defaultBrowserPath(browser string) (string, error) {
	browser = strings.ToLower(browser)
	if browser != "" && browser != "auto" && browser != "chrome" && browser != "edge" {
		return "", fmt.Errorf("browser must be auto, chrome, or edge")
	}
	candidates := []string{"google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "msedge"}
	if browser == "chrome" {
		candidates = []string{"google-chrome", "google-chrome-stable", "chromium", "chromium-browser"}
	} else if browser == "edge" {
		candidates = []string{"msedge"}
	}
	if runtime.GOOS == "windows" {
		chrome := []string{filepath.Join(os.Getenv("ProgramFiles"), "Google", "Chrome", "Application", "chrome.exe"), "chrome.exe"}
		edge := []string{filepath.Join(os.Getenv("ProgramFiles(x86)"), "Microsoft", "Edge", "Application", "msedge.exe"), "msedge.exe"}
		switch browser {
		case "chrome":
			candidates = chrome
		case "edge":
			candidates = edge
		default:
			candidates = append(chrome, edge...)
		}
	}
	for _, candidate := range candidates {
		if strings.Contains(candidate, string(filepath.Separator)) {
			if _, err := os.Stat(candidate); err == nil {
				return candidate, nil
			}
			continue
		}
		if path, err := exec.LookPath(candidate); err == nil {
			return path, nil
		}
	}
	return "", fmt.Errorf("could not find Chrome or Edge; pass --browser-path or --attach")
}

func targetExpression(index int, body string) string {
	return fmt.Sprintf(`(() => { const el = document.querySelector('[data-geniex-jev-index=%q]'); if (!el || el.disabled || el.getClientRects().length === 0) throw new Error('target is stale'); const r = el.getBoundingClientRect(); if (document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2) !== el) throw new Error('target is occluded'); return (%s)(el); })()`, fmt.Sprintf("%d", index), body)
}

func jsString(value string) string {
	encoded, _ := json.Marshal(value)
	return string(encoded)
}

func submitExpression(submit bool) string {
	if !submit {
		return ""
	}
	return "if (el.form) el.form.requestSubmit();"
}

const snapshotScript = `(() => {
  const selectors = 'a[href],button,input,textarea,select,[contenteditable=true],[role=button],[onclick]';
  let index = 0;
  const elements = [];
  for (const el of document.querySelectorAll(selectors)) {
    const style = getComputedStyle(el), rect = el.getBoundingClientRect();
    if (style.display === 'none' || style.visibility === 'hidden' || rect.width < 2 || rect.height < 2 || el.disabled) continue;
    const center = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
    if (center !== el && !el.contains(center)) continue;
    const id = ++index;
    el.setAttribute('data-geniex-jev-index', String(id));
    const text = ['input', 'textarea', 'select'].includes(el.tagName.toLowerCase()) ? '' : el.innerText;
    const name = (el.getAttribute('aria-label') || text || el.placeholder || el.title || '').trim().slice(0, 160);
    elements.push({index:id, tag:el.tagName.toLowerCase(), role:el.getAttribute('role') || '', name, type:el.type || '', href:el.href || '', target:el.target || '', download:el.hasAttribute('download'), form:!!el.form, disabled:!!el.disabled, content_editable:el.isContentEditable});
  }
  return {url:location.href, title:document.title, elements};
})()`

// SortElements gives deterministic fixtures and trace output to callers that
// receive element maps from non-CDP test doubles.
func SortElements(elements []jev.Element) {
	sort.Slice(elements, func(i, j int) bool { return elements[i].Index < elements[j].Index })
}
