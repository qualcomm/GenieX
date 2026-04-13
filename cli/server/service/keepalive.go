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

package service

import (
	"fmt"
	"log/slog"
	"reflect"
	"slices"
	"sync"
	"time"

	geniex_bridge "github.com/qcom-it-nexa-ai/geniex/bindings/go"
	"github.com/qcom-it-nexa-ai/geniex/internal/config"
	"github.com/qcom-it-nexa-ai/geniex/internal/store"
	"github.com/qcom-it-nexa-ai/geniex/internal/types"
	"github.com/qcom-it-nexa-ai/geniex/server/utils"
)

// KeepAliveGet retrieves a model from the keepalive cache or creates it if not found
// This avoids the overhead of repeatedly loading/unloading models from disk
func KeepAliveGet[T any](name string, param types.ModelParam, reset bool) (*T, error) {
	t, err := keepAliveGet[T](name, param, reset)
	if err != nil {
		return nil, err
	}
	return t.(*T), nil
}

var keepAlive keepAliveService

// current only support keepalive one model
type keepAliveService struct {
	models map[string]*modelKeepInfo // Map of model name to model info
	stopCh chan struct{}             // Channel to stop the cleanup goroutine

	sync.Mutex // Protects concurrent access to models map
}

// modelKeepInfo holds metadata for a cached model instance
type modelKeepInfo struct {
	model    keepable
	param    types.ModelParam
	lastTime time.Time
}

// keepable interface defines objects that can be managed by the keepalive service
// Objects must support cleanup and reset operations
type keepable interface {
	Destroy() error
}

type keepResetable interface {
	keepable
	Reset() error
}

// start begins the background cleanup process that removes unused models
// Runs a ticker every 5 seconds to check for models that exceed the keepalive timeout
func (keepAlive *keepAliveService) start() {
	keepAlive.models = make(map[string]*modelKeepInfo)
	keepAlive.stopCh = make(chan struct{})

	go func() {
		t := time.NewTicker(5 * time.Second)
		for {
			select {
			// Stop signal received - cleanup all models and exit
			case <-keepAlive.stopCh:
				keepAlive.Lock()
				for name, model := range keepAlive.models {
					model.model.Destroy()
					delete(keepAlive.models, name)
				}
				keepAlive.Unlock()
				return

			// Periodic cleanup - remove models that haven't been used recently
			case <-t.C:
				keepAlive.Lock()
				for name, model := range keepAlive.models {
					if time.Since(model.lastTime).Milliseconds()/1000 > config.Get().KeepAlive {
						model.model.Destroy()
						delete(keepAlive.models, name)
					}
				}
				keepAlive.Unlock()
			}
		}
	}()
}

// keepAliveGet retrieves a cached model or creates a new one if not found
// Ensures only one model is kept in memory at a time by clearing others
func keepAliveGet[T any](name string, param types.ModelParam, reset bool) (any, error) {
	keepAlive.Lock()
	defer keepAlive.Unlock()

	name, quant := utils.NormalizeModelName(name)
	slog.Debug("KeepAliveGet", "name", name, "quant", quant, "param", param)

	s := store.Get()

	manifest, err := s.GetManifest(name)
	if err != nil {
		return nil, err
	}

	var modelfile string
	if quant != "" {
		if fileinfo, exists := manifest.ModelFile[quant]; !exists {
			return nil, fmt.Errorf("quantization %s not found for model %s", quant, name)
		} else if !fileinfo.Downloaded {
			return nil, fmt.Errorf("quantization %s not downloaded for model %s", quant, name)
		} else {
			modelfile = s.ModelfilePath(manifest.Name, fileinfo.Name)
		}
	} else {
		// fallback to first downloaded model file
		quants := make([]string, 0, len(manifest.ModelFile))
		for quant, v := range manifest.ModelFile {
			if v.Downloaded {
				quants = append(quants, quant)
				break
			}
		}
		quant = slices.Min(quants)
		slog.Debug("KeepAliveGet quant fallback", "quant", quant)
		modelfile = s.ModelfilePath(manifest.Name, manifest.ModelFile[quant].Name) // at least one is downloaded
	}

	// Check if model already exists in cache
	model, ok := keepAlive.models[name+":"+quant]
	if ok && reflect.DeepEqual(model.param, param) {
		if reset {
			if r, ok := model.model.(keepResetable); ok {
				r.Reset()
			}
		}
		model.lastTime = time.Now()
		return model.model, nil
	}

	// Clear existing models to ensure only one is in memory
	// This prevents memory overflow but limits to single model usage
	// TODO: unload model due to free ram/vram
	for name, model := range keepAlive.models {
		model.model.Destroy()
		delete(keepAlive.models, name)
	}

	var t keepable
	var e error
	switch reflect.TypeFor[T]() {
	case reflect.TypeFor[geniex_bridge.LLM]():
		t, e = geniex_bridge.NewLLM(geniex_bridge.LlmCreateInput{
			ModelName: manifest.ModelName,
			ModelPath: modelfile,
			Config: geniex_bridge.ModelConfig{
				NCtx:       param.NCtx,
				NGpuLayers: param.NGpuLayers,
			},
			PluginID: manifest.PluginId,
			DeviceID: manifest.DeviceId,
		})
	case reflect.TypeFor[geniex_bridge.VLM]():
		var mmproj string
		if manifest.MMProjFile.Name != "" {
			mmproj = s.ModelfilePath(manifest.Name, manifest.MMProjFile.Name)
		}
		var tokenizer string
		if manifest.TokenizerFile.Name != "" {
			tokenizer = s.ModelfilePath(manifest.Name, manifest.TokenizerFile.Name)
		}
		t, e = geniex_bridge.NewVLM(geniex_bridge.VlmCreateInput{
			ModelName:     manifest.ModelName,
			ModelPath:     modelfile,
			MmprojPath:    mmproj,
			TokenizerPath: tokenizer,
			Config: geniex_bridge.ModelConfig{
				NCtx:       param.NCtx,
				NGpuLayers: param.NGpuLayers,
			},
			PluginID: manifest.PluginId,
			DeviceID: manifest.DeviceId,
		})
	case reflect.TypeFor[geniex_bridge.Embedder]():
		t, e = geniex_bridge.NewEmbedder(geniex_bridge.EmbedderCreateInput{
			ModelName: manifest.ModelName,
			ModelPath: modelfile,
			PluginID:  manifest.PluginId,
			DeviceID:  manifest.DeviceId,
		})
	case reflect.TypeFor[geniex_bridge.Reranker]():
		t, e = geniex_bridge.NewReranker(geniex_bridge.RerankerCreateInput{
			ModelName: manifest.ModelName,
			ModelPath: modelfile,
			PluginID:  manifest.PluginId,
			DeviceID:  manifest.DeviceId,
		})
	case reflect.TypeFor[geniex_bridge.TTS]():
		t, e = geniex_bridge.NewTTS(geniex_bridge.TtsCreateInput{
			ModelName: manifest.ModelName,
			ModelPath: modelfile,
			PluginID:  manifest.PluginId,
			DeviceID:  manifest.DeviceId,
		})
	case reflect.TypeFor[geniex_bridge.ASR]():
		t, e = geniex_bridge.NewASR(geniex_bridge.AsrCreateInput{
			ModelName: manifest.ModelName,
			ModelPath: modelfile,
			Config: geniex_bridge.ASRModelConfig{
				NCtx:       param.NCtx,
				NGpuLayers: param.NGpuLayers,
			},
			PluginID: manifest.PluginId,
			DeviceID: manifest.DeviceId,
		})
	case reflect.TypeFor[geniex_bridge.Diarize]():
		t, e = geniex_bridge.NewDiarize(geniex_bridge.DiarizeCreateInput{
			ModelName: manifest.ModelName,
			ModelPath: modelfile,
			Config: geniex_bridge.DiarizeModelConfig{
				NCtx:       param.NCtx,
				NGpuLayers: param.NGpuLayers,
			},
			PluginID: manifest.PluginId,
			DeviceID: manifest.DeviceId,
		})
	case reflect.TypeFor[geniex_bridge.CV]():
		t, e = geniex_bridge.NewCV(geniex_bridge.CVCreateInput{
			ModelName: manifest.ModelName,
			Config: geniex_bridge.CVModelConfig{
				DetModelPath: modelfile,
				RecModelPath: modelfile,
			},
			PluginID: manifest.PluginId,
			DeviceID: manifest.DeviceId,
		})
	case reflect.TypeFor[geniex_bridge.ImageGen]():
		// For image generation models, use the model directory path instead of specific file
		modelDir := s.ModelfilePath(manifest.Name, "")
		t, e = geniex_bridge.NewImageGen(geniex_bridge.ImageGenCreateInput{
			ModelName: manifest.ModelName,
			ModelPath: modelDir,
			PluginID:  manifest.PluginId,
			DeviceID:  manifest.DeviceId,
		})
	default:
		panic(fmt.Sprintf("not support type: %+#v", t))
	}
	if e != nil {
		return nil, e
	}
	model = &modelKeepInfo{
		model:    t,
		param:    param,
		lastTime: time.Now(),
	}
	keepAlive.models[name+":"+quant] = model

	return model.model, nil
}

// stop signals the cleanup goroutine to terminate
func (keepAlive *keepAliveService) stop() {
	keepAlive.stopCh <- struct{}{}
}
