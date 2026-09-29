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
	"github.com/qualcomm/GenieX/cli/server/utils"
)

// resolveDraftModelPath resolves spec_draft_model to a GGUF path: an existing
// path passes through as-is, otherwise it's a catalogue name (optionally
// :precision) looked up in the local cache — never auto-pulled.
func resolveDraftModelPath(draft string) (string, error) {
	if draft == "" {
		return "", nil
	}
	if _, err := os.Stat(draft); err == nil {
		return draft, nil
	}
	name, precision := geniex_sdk.SplitNamePrecision(draft)
	paths, err := geniex_sdk.ModelGetPaths(geniex_sdk.JoinNamePrecision(name, precision))
	if err != nil {
		return "", fmt.Errorf("resolve draft model %q: %w", draft, err)
	}
	return paths.ModelPath, nil
}

// ResolveModelParam turns the model-load options into the ModelParam the cache
// keys on. Compute is resolved to a DeviceID by the SDK; nctx/ngl are
// llama_cpp-only and zeroed for other plugins. power_mode is shared by both
// plugins and resolved here, once, into the geniex_PowerMode the SDK expects.
func ResolveModelParam(runtimeID, modelName string, reqNCtx, reqNgl int32, reqCompute, reqVitCompute, reqPowerMode, chipset string, spec types.SpecParam) (types.ModelParam, error) {
	// Non-llama_cpp plugins (e.g. qairt) reject non-zero nctx; the SDK zeroes
	// ngl for them in geniex_resolve_device.
	nctx, ngl := reqNCtx, reqNgl
	if runtimeID != geniex_sdk.RuntimeLlamaCpp {
		nctx = 0
	}

	resolvedPowerMode, err := geniex_sdk.ResolvePowerMode(reqPowerMode)
	if err != nil {
		return types.ModelParam{}, err
	}

	// Runs before the SDK's npu fallback; chipset comes from the caller so this
	// stays store-free. Ubatch stays 0: ModelParam has no n_ubatch to key on.
	reqCompute, _ = config.ChipsetDefaults(reqCompute, 0, chipset)

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
		NCtx:        nctx,
		NGpuLayers:  resolved.Ngl,
		DeviceID:    resolved.DeviceID,
		VitDeviceID: reqVitCompute,
		PowerMode:   resolvedPowerMode,
	}
	// Spec is llama_cpp-only; leave it zero (disabled) for other plugins.
	if runtimeID == geniex_sdk.RuntimeLlamaCpp {
		mp.Spec = spec
	}
	return mp, nil
}

// AcquiredModel is a cached model and whether it has reusable request state.
type AcquiredModel[T any] struct {
	Model *T
	Fresh bool
}

// KeepAliveGet returns the cached model of type T, loading it if needed.
// session identifies the request's conversation; Fresh is true after a load
// or reset.
func KeepAliveGet[T any](name string, param types.ModelParam, session utils.SessionKey) (AcquiredModel[T], error) {
	t, fresh, err := keepAliveGet[T](name, param, session)
	if err != nil {
		return AcquiredModel[T]{}, err
	}
	// Stamp the idle timer at request end (only model requests reach here).
	middleware.RunOnRelease(func() { keepAlive.lastActivity = time.Now() })
	return AcquiredModel[T]{Model: t.(*T), Fresh: fresh}, nil
}

var keepAlive keepAliveService

// keepAliveService caches a single loaded model. All access is under the
// request GIL (middleware.GILock), so it needs no lock of its own.
type keepAliveService struct {
	name         string           // cache key of the loaded model, "" when none
	model        keepable         // nil when none
	param        types.ModelParam // params the cache keys on
	lastSession  utils.SessionKey // session served by the last request
	lastActivity time.Time        // when the last model request finished
	stopCh       chan struct{}
}

// keepable is a model the cache can free; resettable can also clear its
// KV cache / turn state without a full reload.
type keepable interface {
	Destroy() error
}

type resettable interface {
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

// sweep frees the model once idle past the timeout. Runs only when it can
// take the GIL, so it never fires mid-generation; idle is measured from the
// last request's end (#1322).
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

// keepAliveGet reuses the cached model when name/params match, resetting it
// when session isn't a continuation of the one last served, so a new
// conversation never inherits another's KV cache. Returns whether the model
// is fresh after a reset or load. Runs under the request GIL.
func keepAliveGet[T any](name string, param types.ModelParam, session utils.SessionKey) (any, bool, error) {
	// The SDK resolves bare names / aliases and picks the default precision
	// when none is given; pass the request string through verbatim.
	paths, err := geniex_sdk.ModelGetPaths(name)
	if err != nil {
		return nil, false, err
	}
	slog.Debug("KeepAliveGet", "name", name, "param", param, "model_path", paths.ModelPath)

	modelfile := paths.ModelPath

	if keepAlive.name == name && reflect.DeepEqual(keepAlive.param, param) {
		fresh := !utils.IsContinuation(keepAlive.lastSession, session)
		if fresh {
			if r, ok := keepAlive.model.(resettable); ok {
				if err := r.Reset(); err != nil {
					return nil, false, err
				}
			}
		}
		keepAlive.lastSession = session
		return keepAlive.model, fresh, nil
	}

	// Drop the current model so only one stays in memory.
	// TODO: unload model due to free ram/vram
	keepAlive.destroy()

	// param already carries the resolved NCtx/NGpuLayers/DeviceID; no further
	// resolution needed here.
	var t keepable
	var e error
	switch reflect.TypeFor[T]() {
	case reflect.TypeFor[geniex_sdk.LLM]():
		draftPath := ""
		if param.Spec.Type != "" && param.Spec.DraftModel != "" {
			p, perr := resolveDraftModelPath(param.Spec.DraftModel)
			if perr != nil {
				return nil, false, perr
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
				PowerMode:      param.PowerMode,
			},
			RuntimeID: paths.RuntimeID,
		})
	case reflect.TypeFor[geniex_sdk.VLM]():
		t, e = geniex_sdk.NewVLM(geniex_sdk.VlmCreateInput{
			ModelPath:   modelfile,
			MmprojPath:  paths.MmprojPath,
			DeviceID:    param.DeviceID,
			VitDeviceID: param.VitDeviceID,
			Config: geniex_sdk.ModelConfig{
				NCtx:       param.NCtx,
				NGpuLayers: param.NGpuLayers,
				PowerMode:  param.PowerMode,
			},
			RuntimeID: paths.RuntimeID,
		})
	default:
		return nil, false, fmt.Errorf("unsupported model type: %s", reflect.TypeFor[T]())
	}
	if e != nil {
		return nil, false, e
	}
	keepAlive.name = name
	keepAlive.model = t
	keepAlive.param = param
	keepAlive.lastSession = session

	return t, true, nil
}

// stop ends the sweep goroutine and frees the model here, not in the
// goroutine, so it lands before the SDK deinit that follows.
func (keepAlive *keepAliveService) stop() {
	close(keepAlive.stopCh)
	middleware.GILock.Lock()
	defer middleware.GILock.Unlock()
	keepAlive.destroy()
}
