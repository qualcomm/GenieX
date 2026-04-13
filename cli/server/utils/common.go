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

package utils

import (
	"bytes"
	"encoding/base64"
	"errors"
	"image"
	"image/png"
	"io"
	"mime"
	"net/http"
	"net/url"
	"os"
	"strings"

	_ "golang.org/x/image/webp"
)

func SaveURIToTempFile(uri string) (string, error) {
	u, err := url.Parse(uri)
	if err != nil {
		return "", err
	}

	var data []byte
	// windows drive letter
	if len(u.Scheme) == 1 && (u.Scheme[0] >= 'a' && u.Scheme[0] <= 'z' || u.Scheme[0] >= 'A' && u.Scheme[0] <= 'Z') {
		u.Scheme = "file"
		u.Path = uri
	}
	switch u.Scheme {
	case "http", "https":
		resp, err := http.Get(uri)
		if err != nil {
			return "", err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return "", errors.New("http download failed: " + resp.Status)
		}
		data, err = io.ReadAll(resp.Body)
		if err != nil {
			return "", err
		}
	case "data":
		parts := strings.SplitN(u.Opaque, ",", 2)
		if len(parts) != 2 {
			return "", errors.New("invalid data URI format")
		}
		// format: data:[<mediatype>][;base64],<data>
		if strings.Contains(parts[0], ";base64") {
			data, err = base64.StdEncoding.DecodeString(parts[1])
			if err != nil {
				return "", err
			}

		} else {
			// format: data:[<mediatype>],<data>
			decoded, err := url.QueryUnescape(parts[1])
			if err != nil {
				return "", err
			}
			data = []byte(decoded)
		}
	case "file", "":
		data, err = os.ReadFile(u.Path)
		if err != nil {
			return "", err
		}
	default:

		return "", errors.New("unsupported scheme: " + u.Scheme)
	}

	// Detect content type
	contentType := http.DetectContentType(data)

	// Convert WebP to PNG for compatibility with native SDK
	if strings.HasPrefix(contentType, "image/webp") || strings.HasSuffix(strings.ToLower(u.Path), ".webp") {
		img, _, err := image.Decode(bytes.NewReader(data))
		if err != nil {
			return "", errors.New("failed to decode WebP image: " + err.Error())
		}

		var buf bytes.Buffer
		if err := png.Encode(&buf, img); err != nil {
			return "", errors.New("failed to encode image as PNG: " + err.Error())
		}

		data = buf.Bytes()
		contentType = "image/png"
	}

	fileExt := ""
	if exts, err := mime.ExtensionsByType(contentType); err == nil && len(exts) > 0 {
		fileExt = exts[0]
	}

	tmpFile, err := os.CreateTemp("", "uri-*"+fileExt)
	if err != nil {
		return "", err
	}
	defer tmpFile.Close()

	_, err = tmpFile.Write(data)
	if err != nil {
		os.Remove(tmpFile.Name())
		return "", err
	}
	return tmpFile.Name(), nil
}

func NormalizeModelName(name string) (string, string) {
	if strings.Contains(name, ":") {
		parts := strings.SplitN(name, ":", 2)
		return parts[0], parts[1]
	}
	return name, ""
}
