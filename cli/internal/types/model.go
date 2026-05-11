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

package types

// ManifestFileName is the name of the per-model manifest file written into
// every model directory.
const ManifestFileName = "geniex.json"

type ModelType string

const (
	ModelTypeLLM ModelType = "llm"
	ModelTypeVLM ModelType = "vlm"
)

var AllModelTypes = []ModelType{
	ModelTypeLLM,
	ModelTypeVLM,
}

type ModelFileInfo struct {
	Name       string
	Downloaded bool
	Size       int64
}

type ModelManifest struct {
	Name      string // OrgName/RepoName
	ModelName string // model arch name like "qwen3-4b", "yolov12", etc.
	ModelType ModelType
	PluginId  string
	DeviceId  string

	ModelFile     map[string]ModelFileInfo // quant -> modelfile
	MMProjFile    ModelFileInfo
	TokenizerFile ModelFileInfo
	ExtraFiles    []ModelFileInfo
}

func (m ModelManifest) GetSize() int64 {
	var count int64

	for _, v := range m.ModelFile {
		if v.Downloaded {
			count += v.Size
		}
	}
	if m.MMProjFile.Downloaded {
		count += m.MMProjFile.Size
	}
	if m.TokenizerFile.Downloaded {
		count += m.TokenizerFile.Size
	}
	for _, v := range m.ExtraFiles {
		if v.Downloaded {
			count += v.Size
		}
	}

	return count
}

type ModelParam struct {
	NCtx       int32
	NGpuLayers int32
}

type DownloadInfo struct {
	// CurrentFileName   string
	// CurrentDownloaded int64
	// CurrentSize       int64
	TotalDownloaded int64
	TotalSize       int64
}
