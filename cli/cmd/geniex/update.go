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

package main

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/bytedance/sonic"
	"github.com/spf13/cobra"
	"golang.org/x/mod/semver"
	"golang.org/x/sync/errgroup"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/downloader"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/render"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/store"
)

const (
	githubAPIURL = "https://api.github.com/repos/qcom-ai-hub/geniex/releases/latest"
	userAgent    = "GenieX-Updater/1.0"

	updateCheckInterval  = 24 * time.Hour
	notificationInterval = 8 * time.Hour

	defaultChunkSize  = 4 * 1024 * 1024
	defaultNumWorkers = 16

	linuxInstallScriptURL = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/install.sh"
)

func update() *cobra.Command {
	return &cobra.Command{
		GroupID: "management",
		Use:     "update",
		Short:   "update geniex",
		Long:    "Update geniex to the latest version",
		Run: func(cmd *cobra.Command, args []string) {
			if err := runUpdate(cmd, args); err != nil {
				fmt.Println(render.GetTheme().Error.Sprintf("Update failed: %s", err))
			}
		},
	}
}

func runUpdate(_ *cobra.Command, _ []string) error {
	rls, err := getLastRelease()
	if err != nil {
		return err
	}

	latest := rls.Name
	cmp, err := compareVersion(Version, latest)
	if err != nil {
		return err
	}
	if cmp >= 0 {
		fmt.Println("Already up-to-date.")
		return nil
	}

	// Linux auto-update is not yet wired to the new tar.gz layout;
	// point users at the canonical install script for now.
	if runtime.GOOS == "linux" {
		fmt.Println(render.GetTheme().Warning.Sprintf(
			"New version %s available. Re-run the install script to upgrade:", latest))
		fmt.Println(render.GetTheme().Success.Sprintf(
			"  curl -fsSL %s | bash", linuxInstallScriptURL))
		return nil
	}
	if runtime.GOOS != "windows" {
		return fmt.Errorf("auto-update is not supported on %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	assetName := fmt.Sprintf("geniex-cli-setup-windows-%s-%s.exe", runtime.GOARCH, latest)
	var ast asset
	for _, a := range rls.Assets {
		if a.Name == assetName {
			ast = a
			break
		}
	}
	if ast.Name == "" {
		return fmt.Errorf("asset %s not found in release", assetName)
	}

	fmt.Println(
		render.GetTheme().Warning.Sprint("New version found, file: "),
		render.GetTheme().Success.Sprint(ast.Name),
		render.GetTheme().Warning.Sprint(", version: "),
		render.GetTheme().Success.Sprint(latest))

	dst := filepath.Join(os.TempDir(), ast.Name)
	progress := make(chan int64)
	bar := render.NewProgressBar(int64(ast.Size), "downloading")

	var dlErr error
	go func() {
		dlErr = downloadPkg(ast.URL, dst, int64(ast.Size), progress)
	}()
	for pg := range progress {
		bar.Add(pg)
	}
	bar.Exit()
	if dlErr != nil {
		return dlErr
	}

	if err := exec.Command(dst).Start(); err != nil {
		return err
	}
	fmt.Println("update package is ready to install")
	return nil
}

// GitHub release API

// Release `name` is the display version (e.g. "v0.1.4"). We prefer it over
// `tag_name` because releases published without a git tag surface a synthetic
// `tag_name` like "untagged-<sha>".
type release struct {
	Name   string  `json:"name"`
	Assets []asset `json:"assets"`
}

type asset struct {
	// URL is the GitHub API endpoint. Use it (with Accept:
	// application/octet-stream) for private repos — browser_download_url
	// 404s when the release was published without a real git tag.
	URL    string `json:"url"`
	Name   string `json:"name"`
	Size   int    `json:"size"`
	Digest string `json:"digest"`
}

// resolveGitHubToken returns a PAT for authenticating the release-lookup call.
// geniex's own var wins over the generic one used by gh / CI.
func resolveGitHubToken() string {
	if t := os.Getenv("GENIEX_GITHUB_TOKEN"); t != "" {
		return t
	}
	return os.Getenv("GITHUB_TOKEN")
}

func getLastRelease() (release, error) {
	var rls release

	req, err := http.NewRequest("GET", githubAPIURL, nil)
	if err != nil {
		return rls, err
	}
	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("Accept", "application/vnd.github+json")
	if tok := resolveGitHubToken(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}

	resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
	if err != nil {
		return rls, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return rls, fmt.Errorf("get latest release failed: %d", resp.StatusCode)
	}
	return rls, sonic.ConfigDefault.NewDecoder(resp.Body).Decode(&rls)
}

// compareVersion compares two SemVer strings.
// Returns: -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2.
// Accepts bare versions ("1.2.3") by normalizing to the "v" prefix semver expects.
func compareVersion(v1, v2 string) (int, error) {
	n1, n2 := normalizeSemver(v1), normalizeSemver(v2)
	for _, p := range [...]struct{ orig, norm string }{{v1, n1}, {v2, n2}} {
		if !semver.IsValid(p.norm) {
			return 0, fmt.Errorf("invalid format: %s", p.orig)
		}
	}
	return semver.Compare(n1, n2), nil
}

func normalizeSemver(v string) string {
	if !strings.HasPrefix(v, "v") {
		return "v" + v
	}
	return v
}

// Parallel chunked downloader

func downloadPkg(url, dst string, size int64, progress chan int64) error {
	defer close(progress)

	file, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	chunkSize := int64(defaultChunkSize)
	numWorkers := min(int((size+chunkSize-1)/chunkSize), defaultNumWorkers)
	slog.Debug("downloading package", "url", url, "size", size, "chunkSize", chunkSize, "numWorkers", numWorkers)

	g, ctx := errgroup.WithContext(context.Background())
	g.SetLimit(numWorkers)
	dl := downloader.NewDownloader()
	dl.AuthToken = resolveGitHubToken()
	dl.Headers = map[string]string{"Accept": "application/octet-stream"}

	for offset := int64(0); offset < size; offset += chunkSize {
		offset := offset
		g.Go(func() error {
			limit := min(chunkSize, size-offset)
			buf := bytes.NewBuffer(make([]byte, 0, int(limit)))
			if err := dl.DownloadChunk(ctx, url, offset, limit, buf); err != nil {
				return fmt.Errorf("failed to download chunk at offset %d: %w", offset, err)
			}
			if _, err := file.WriteAt(buf.Bytes(), offset); err != nil {
				return fmt.Errorf("failed to write chunk at offset %d: %w", offset, err)
			}
			progress <- int64(buf.Len())
			return nil
		})
	}

	return g.Wait()
}

// Update check store

type updateCheck struct {
	LastCheck     time.Time `json:"last_check"`
	LastNotify    time.Time `json:"last_notify"`
	LatestVersion string    `json:"latest_version"`
}

func updateCheckPath() string {
	return filepath.Join(store.Get().DataPath(), "update_check")
}

func getUpdateCheck() updateCheck {
	var ck updateCheck
	data, err := os.ReadFile(updateCheckPath())
	if err != nil {
		return ck
	}
	sonic.Unmarshal(data, &ck)
	return ck
}

func setUpdateCheck(ck updateCheck) {
	data, _ := sonic.Marshal(ck)
	if err := os.WriteFile(updateCheckPath(), data, 0644); err != nil {
		slog.Debug("update check save failed", "error", err)
	}
}

// Background check & pre-launch notice

func checkUpdate() {
	ck := getUpdateCheck()
	if time.Since(ck.LastCheck) < updateCheckInterval {
		return
	}

	rls, err := getLastRelease()
	if err != nil {
		slog.Debug("update check failed", "error", err)
		return
	}

	ck.LastCheck = time.Now()
	ck.LatestVersion = rls.Name
	setUpdateCheck(ck)
}

func notifyUpdate() {
	ck := getUpdateCheck()
	if ck.LatestVersion == "" || time.Since(ck.LastNotify) < notificationInterval {
		return
	}
	cmp, err := compareVersion(Version, ck.LatestVersion)
	if err != nil || cmp >= 0 {
		return
	}

	ck.LastNotify = time.Now()
	setUpdateCheck(ck)

	fmt.Printf("\n\n%s %s → %s\n",
		render.GetTheme().Warning.Sprintf("A new version of geniex-cli is available:"),
		render.GetTheme().Success.Sprint(Version),
		render.GetTheme().Success.Sprint(ck.LatestVersion))
	fmt.Printf("%s\n\n",
		render.GetTheme().Warning.Sprint("To update, run: `geniex update`"))
}
