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
	"sync"
	"time"

	geniex_sdk "github.com/qcom-it-nexa-ai/geniex/bindings/go"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/config"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/types"
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

	// The SDK resolves bare names / aliases and picks the default precision
	// when none is given; pass the request string through verbatim.
	paths, err := geniex_sdk.ModelGetPaths(name)
	if err != nil {
		return nil, err
	}
	slog.Debug("KeepAliveGet", "name", name, "param", param, "model_path", paths.ModelPath)

	modelfile := paths.ModelPath

	// Check if model already exists in cache
	model, ok := keepAlive.models[name]
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

	// Resolve llama_cpp-only defaults: NCtx and NGpuLayers are meaningful only for
	// llama_cpp. For other plugins (e.g. qairt) they must stay 0 so the plugin's
	// param-guard is not tripped by the request defaults.
	// TODO: Remove this resolution once the C API exposes geniex_get_last_error_detail()
	// and the plugin can report the exact unsupported param name directly.
	nctx, ngl := param.NCtx, param.NGpuLayers
	if paths.PluginID == geniex_sdk.PluginLlamaCpp {
		if nctx == 0 {
			nctx = 4096
		}
		if ngl == 0 {
			ngl = 999
		}
	}

	var t keepable
	var e error
	switch reflect.TypeFor[T]() {
	case reflect.TypeFor[geniex_sdk.LLM]():
		t, e = geniex_sdk.NewLLM(geniex_sdk.LlmCreateInput{
			ModelName: paths.ModelName,
			ModelPath: modelfile,
			Config: geniex_sdk.ModelConfig{
				NCtx:       nctx,
				NGpuLayers: ngl,
			},
			PluginID: paths.PluginID,
		})
	case reflect.TypeFor[geniex_sdk.VLM]():
		t, e = geniex_sdk.NewVLM(geniex_sdk.VlmCreateInput{
			ModelName:  paths.ModelName,
			ModelPath:  modelfile,
			MmprojPath: paths.MmprojPath,
			Config: geniex_sdk.ModelConfig{
				NCtx:       nctx,
				NGpuLayers: ngl,
			},
			PluginID: paths.PluginID,
		})
	default:
		return nil, fmt.Errorf("unsupported model type: %s", reflect.TypeFor[T]())
	}
	if e != nil {
		return nil, e
	}
	model = &modelKeepInfo{
		model:    t,
		param:    param,
		lastTime: time.Now(),
	}
	keepAlive.models[name] = model

	return model.model, nil
}

// stop signals the cleanup goroutine to terminate
func (keepAlive *keepAliveService) stop() {
	keepAlive.stopCh <- struct{}{}
}
