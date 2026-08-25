// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package service

import (
	"fmt"
	"log/slog"
	"os"
	"reflect"
	"time"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/internal/config"
	"github.com/qualcomm/GenieX/cli/internal/render"
	"github.com/qualcomm/GenieX/cli/server/middleware"
	"github.com/qualcomm/GenieX/cli/server/types"
)

// resolveDraftModelPath maps a spec_draft_model value to an absolute GGUF path:
// an existing filesystem path is returned as-is, otherwise it is a catalogue
// name (optionally :precision) looked up in the local cache. A cache miss is an
// error — the server never auto-pulls, so the draft must be pulled beforehand.
func resolveDraftModelPath(draft string) (string, error) {
	if draft == "" {
		return "", nil
	}
	if _, err := os.Stat(draft); err == nil {
		return draft, nil
	}
	name, precision := geniex_sdk.SplitNamePrecision(draft)
	key := name
	if precision != "" {
		key = name + ":" + precision
	}
	paths, err := geniex_sdk.ModelGetPaths(key)
	if err != nil {
		return "", fmt.Errorf("resolve draft model %q: %w", draft, err)
	}
	return paths.ModelPath, nil
}

// ResolveModelParam turns the already-resolved (nctx, ngl, compute) knobs into
// the ModelParam the cache keys on. Compute is resolved to a DeviceID by the
// SDK; nctx/ngl are llama_cpp-only and zeroed for other plugins.
func ResolveModelParam(runtimeID, modelName string, reqNCtx, reqNgl int32, reqCompute, chipset string, spec types.SpecParam) (types.ModelParam, error) {
	// Non-llama_cpp plugins (e.g. qairt) reject non-zero nctx; the SDK zeroes
	// ngl for them in geniex_resolve_device.
	nctx, ngl := reqNCtx, reqNgl
	if runtimeID != geniex_sdk.RuntimeLlamaCpp {
		nctx = 0
	}

	// Host-aware default (e.g. RB3 Gen 2 → cpu) before the SDK's npu fallback.
	// chipset is resolved by the caller (offline) so this stays store-free.
	if resolved, overridden := config.ComputeDefault(reqCompute, chipset); overridden {
		slog.Info("applied host-aware compute default", "compute", resolved)
		fmt.Println(render.GetTheme().Info.Sprintf("Defaulting to compute %s for this device.", resolved))
		reqCompute = resolved
	}

	resolved, err := geniex_sdk.ResolveDevice(geniex_sdk.ResolveDeviceInput{
		RuntimeID:   runtimeID,
		ModelName:   modelName,
		ComputeUnit: reqCompute,
		NglDefault:  ngl,
	})
	if err != nil {
		return types.ModelParam{}, err
	}
	if resolved.Warning != "" {
		slog.Warn("compute unit coerced", "warning", resolved.Warning)
		fmt.Println(render.GetTheme().Warning.Sprintf("Warning: %s", resolved.Warning))
	}

	mp := types.ModelParam{
		NCtx:       nctx,
		NGpuLayers: resolved.Ngl,
		DeviceID:   resolved.DeviceID,
	}
	// Spec is llama_cpp-only; leave it zero (disabled) for other plugins.
	if runtimeID == geniex_sdk.RuntimeLlamaCpp {
		mp.Spec = spec
	}
	return mp, nil
}

// KeepAliveGet returns the cached model of type T, loading it if needed, to
// avoid reloading from disk on every request.
func KeepAliveGet[T any](name string, param types.ModelParam, reset bool) (*T, error) {
	t, err := keepAliveGet[T](name, param, reset)
	if err != nil {
		return nil, err
	}
	// Stamp the idle timer at request end (only model requests reach here).
	middleware.RunOnRelease(func() { keepAlive.lastActivity = time.Now() })
	return t.(*T), nil
}

var keepAlive keepAliveService

// keepAliveService caches a single loaded model. All access is under the
// request GIL (middleware.GILock), so it needs no lock of its own.
type keepAliveService struct {
	name         string           // cache key of the loaded model, "" when none
	model        keepable         // nil when none
	param        types.ModelParam // params the cache keys on
	lastActivity time.Time        // when the last model request finished
	stopCh       chan struct{}
}

// keepable is a model the cache can free; keepResetable can also be reset.
type keepable interface {
	Destroy() error
}

type keepResetable interface {
	keepable
	Reset() error
}

// start runs the background sweep every 5 seconds until stopped.
func (keepAlive *keepAliveService) start() {
	keepAlive.lastActivity = time.Now()
	keepAlive.stopCh = make(chan struct{})

	go func() {
		t := time.NewTicker(5 * time.Second)
		for {
			select {
			case <-keepAlive.stopCh:
				return

			case <-t.C:
				keepAlive.sweep()
			}
		}
	}()
}

// sweep frees the model once idle past the timeout. It runs only when it can
// take the GIL, so an in-flight request defers it and the model is never freed
// mid-generation; idle is measured from the last model request's end (#1322).
func (keepAlive *keepAliveService) sweep() {
	if !middleware.GILock.TryLock() {
		return
	}
	defer middleware.GILock.Unlock()

	if time.Since(keepAlive.lastActivity).Milliseconds()/1000 > config.Get().KeepAlive {
		keepAlive.destroy()
	}
}

// destroy frees the cached model, if any. Caller holds GILock.
func (keepAlive *keepAliveService) destroy() {
	if keepAlive.model != nil {
		keepAlive.model.Destroy()
		keepAlive.model = nil
		keepAlive.name = ""
	}
}

// keepAliveGet reuses the cached model when name and params match, otherwise
// loads a fresh one. Runs under the request GIL, so no locking here.
func keepAliveGet[T any](name string, param types.ModelParam, reset bool) (any, error) {
	// The SDK resolves bare names / aliases and picks the default precision
	// when none is given; pass the request string through verbatim.
	paths, err := geniex_sdk.ModelGetPaths(name)
	if err != nil {
		return nil, err
	}
	slog.Debug("KeepAliveGet", "name", name, "param", param, "model_path", paths.ModelPath)

	modelfile := paths.ModelPath

	if keepAlive.name == name && reflect.DeepEqual(keepAlive.param, param) {
		if reset {
			if r, ok := keepAlive.model.(keepResetable); ok {
				r.Reset()
			}
		}
		return keepAlive.model, nil
	}

	// Drop the current model so only one stays in memory.
	// TODO: unload model due to free ram/vram
	keepAlive.destroy()

	// param already carries the resolved NCtx / NGpuLayers / DeviceID; the
	// cache keys on it, so no further resolution here.
	var t keepable
	var e error
	switch reflect.TypeFor[T]() {
	case reflect.TypeFor[geniex_sdk.LLM]():
		draftPath := ""
		if param.Spec.Type != "" && param.Spec.DraftModel != "" {
			p, perr := resolveDraftModelPath(param.Spec.DraftModel)
			if perr != nil {
				return nil, perr
			}
			draftPath = p
		}
		t, e = geniex_sdk.NewLLM(geniex_sdk.LlmCreateInput{
			ModelPath: modelfile,
			DeviceID:  param.DeviceID,
			Config: geniex_sdk.ModelConfig{
				NCtx:           param.NCtx,
				NGpuLayers:     param.NGpuLayers,
				SpecType:       param.Spec.Type,
				SpecDraftModel: draftPath,
				SpecNMax:       param.Spec.NMax,
				SpecNMin:       param.Spec.NMin,
				SpecPMin:       param.Spec.PMin,
			},
			RuntimeID: paths.RuntimeID,
		})
	case reflect.TypeFor[geniex_sdk.VLM]():
		t, e = geniex_sdk.NewVLM(geniex_sdk.VlmCreateInput{
			ModelPath:  modelfile,
			MmprojPath: paths.MmprojPath,
			DeviceID:   param.DeviceID,
			Config: geniex_sdk.ModelConfig{
				NCtx:       param.NCtx,
				NGpuLayers: param.NGpuLayers,
			},
			RuntimeID: paths.RuntimeID,
		})
	default:
		return nil, fmt.Errorf("unsupported model type: %s", reflect.TypeFor[T]())
	}
	if e != nil {
		return nil, e
	}
	keepAlive.name = name
	keepAlive.model = t
	keepAlive.param = param

	return t, nil
}

// stop ends the sweep goroutine and frees the cached model — here rather than in
// the goroutine, so it lands before the SDK deinit that follows.
func (keepAlive *keepAliveService) stop() {
	close(keepAlive.stopCh)
	middleware.GILock.Lock()
	defer middleware.GILock.Unlock()
	keepAlive.destroy()
}
