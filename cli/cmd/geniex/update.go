// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
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

	"github.com/qualcomm/GenieX/cli/internal/downloader"
	"github.com/qualcomm/GenieX/cli/internal/render"
	"github.com/qualcomm/GenieX/cli/internal/store"
)

const (
	// releaseBaseURL is the S3 prefix holding index.json and the per-version
	// manifest-<tag>.json files. manifest asset URLs are absolute, so this is
	// only used to resolve index.json and manifest names.
	releaseBaseURL = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/"
	indexURL       = releaseBaseURL + "index.json"
	userAgent      = "GenieX-Updater/1.0"

	updateCheckInterval  = 24 * time.Hour
	notificationInterval = 8 * time.Hour

	defaultChunkSize  = 4 * 1024 * 1024
	defaultNumWorkers = 16

	linuxInstallScriptURL = releaseBaseURL + "install.sh"
)

func update() *cobra.Command {
	return &cobra.Command{
		GroupID: "management",
		Use:     "update",
		Short:   "update geniex",
		Long:    "Update geniex to the latest version",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runUpdate(cmd, args)
		},
	}
}

func runUpdate(_ *cobra.Command, _ []string) error {
	latest, err := getLatestVersion()
	if err != nil {
		return err
	}

	// Refresh the cache so other commands' notify banner matches this result.
	ck := getUpdateCheck()
	ck.LastCheck = time.Now()
	ck.LatestVersion = latest
	setUpdateCheck(ck)

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

	mf, err := getManifest(latest)
	if err != nil {
		return err
	}
	ast, err := mf.find("cli-installer", "windows", runtime.GOARCH)
	if err != nil {
		return err
	}

	fmt.Println(
		render.GetTheme().Warning.Sprint("New version found, file: "),
		render.GetTheme().Success.Sprint(ast.Name),
		render.GetTheme().Warning.Sprint(", version: "),
		render.GetTheme().Success.Sprint(latest))

	dst := filepath.Join(os.TempDir(), ast.Name)
	progress := make(chan int64)
	bar := render.NewProgressBar(int64(ast.Size), 0, "downloading")

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

	if err := verifySHA256(dst, ast.SHA256); err != nil {
		return err
	}

	if err := exec.Command(dst).Start(); err != nil {
		return err
	}
	fmt.Println("update package is ready to install")
	return nil
}

// S3 release index & manifests

// index is the top-level S3 manifest listing every published version.
type index struct {
	LatestStable string `json:"latest_stable"`
}

// manifest describes the downloadable assets for one version.
type manifest struct {
	Tag    string  `json:"tag"`
	Assets []asset `json:"assets"`
}

type asset struct {
	Name     string `json:"name"`
	URL      string `json:"url"`
	Size     int    `json:"size"`
	SHA256   string `json:"sha256"`
	Kind     string `json:"kind"`
	Platform string `json:"platform"`
	Arch     string `json:"arch"`
}

// find returns the asset matching kind/platform/arch, or an error if none.
func (m manifest) find(kind, platform, arch string) (asset, error) {
	for _, a := range m.Assets {
		if a.Kind == kind && a.Platform == platform && a.Arch == arch {
			return a, nil
		}
	}
	return asset{}, fmt.Errorf("no %s asset for %s/%s in release %s", kind, platform, arch, m.Tag)
}

// fetchJSON GETs url and decodes the JSON body into v.
func fetchJSON(url string, v any) error {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", userAgent)

	resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("get %s failed: %d", url, resp.StatusCode)
	}
	return sonic.ConfigDefault.NewDecoder(resp.Body).Decode(v)
}

// getLatestVersion returns the latest stable version tag from the S3 index.
func getLatestVersion() (string, error) {
	var idx index
	if err := fetchJSON(indexURL, &idx); err != nil {
		return "", err
	}
	if idx.LatestStable == "" {
		return "", fmt.Errorf("no stable release found in index")
	}
	return idx.LatestStable, nil
}

// getManifest fetches the per-version asset manifest for tag.
func getManifest(tag string) (manifest, error) {
	var mf manifest
	url := releaseBaseURL + "manifest-" + tag + ".json"
	if err := fetchJSON(url, &mf); err != nil {
		return mf, err
	}
	return mf, nil
}

// verifySHA256 checks that the file at path matches the expected hex digest.
func verifySHA256(path, expected string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	got := hex.EncodeToString(h.Sum(nil))
	if !strings.EqualFold(got, expected) {
		return fmt.Errorf("checksum mismatch: expected %s, got %s", expected, got)
	}
	return nil
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

	latest, err := getLatestVersion()
	if err != nil {
		slog.Debug("update check failed", "error", err)
		return
	}

	ck.LastCheck = time.Now()
	ck.LatestVersion = latest
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
