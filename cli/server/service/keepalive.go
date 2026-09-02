// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package service

import (
	"fmt"
	"log/slog"
	"os"
	"reflect"
	"sync/atomic"
	"time"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/internal/config"
	"github.com/qualcomm/GenieX/cli/internal/render"
	"github.com/qualcomm/GenieX/cli/server/middleware"
	"github.com/qualcomm/GenieX/cli/server/types"
	"github.com/qualcomm/GenieX/cli/server/utils"
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
	paths, err := geniex_sdk.ModelGetPaths(geniex_sdk.JoinNamePrecision(name, precision))
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

// AcquiredModel is a cached model and whether it was loaded or reset for this
// request. Fresh tells VLM callers to replay all request media.
type AcquiredModel[T any] struct {
	Model *T
	Fresh bool
}

// KeepAliveGet applies GenieX's automatic conversation reset policy. A cached
// model is reused only when session exactly extends the last request seen here.
func KeepAliveGet[T any](name string, param types.ModelParam, session utils.SessionKey) (AcquiredModel[T], error) {
	t, fresh, err := keepAliveGet[T](name, param, session, acquireAutomatic)
	if err != nil {
		return AcquiredModel[T]{}, err
	}
	middleware.RunOnRelease(func() { keepAlive.lastActivity = time.Now() })
	return AcquiredModel[T]{Model: t.(*T), Fresh: fresh}, nil
}

// KeepAliveGetManaged applies the reset decision made by the transactional
// managed-cache lineage. It invalidates the automatic lineage so an ordinary
// request can never extend state changed by a managed request.
func KeepAliveGetManaged[T any](name string, param types.ModelParam, reset bool) (*T, error) {
	mode := acquireManagedReuse
	if reset {
		mode = acquireManagedReset
	}
	t, _, err := keepAliveGet[T](name, param, nil, mode)
	if err != nil {
		return nil, err
	}
	middleware.RunOnRelease(func() { keepAlive.lastActivity = time.Now() })
	return t.(*T), nil
}

// KeepAliveGetLegacy retains raw mutable state without lineage validation. It
// exists only for the explicitly enabled synthetic comparison runner.
func KeepAliveGetLegacy[T any](name string, param types.ModelParam) (*T, error) {
	t, _, err := keepAliveGet[T](name, param, nil, acquireLegacy)
	if err != nil {
		return nil, err
	}
	middleware.RunOnRelease(func() { keepAlive.lastActivity = time.Now() })
	return t.(*T), nil
}

var keepAlive keepAliveService
var keepAliveGeneration atomic.Uint64

// KeepAliveGeneration identifies the exact lifetime/reset generation of the
// mutable model state. Managed cache callers bind a committed transcript to
// this value so an idle unload, model reload, or out-of-band reset can never be
// mistaken for retained KV state.
func KeepAliveGeneration() uint64 { return keepAliveGeneration.Load() }

// ResetKeepAlive clears the loaded model's mutable state. If Reset is absent or
// fails, the model is destroyed instead; either outcome prevents a caller from
// reusing partially generated KV state. Caller holds middleware.GILock.
func ResetKeepAlive() error {
	if keepAlive.model == nil {
		return nil
	}
	if r, ok := keepAlive.model.(resettable); ok {
		if err := r.Reset(); err == nil {
			keepAlive.lastSession = nil
			keepAlive.lastSessionValid = false
			keepAliveGeneration.Add(1)
			return nil
		} else {
			keepAlive.destroy()
			return err
		}
	}
	keepAlive.destroy()
	return nil
}

// keepAliveService caches a single loaded model. All access is under the
// request GIL (middleware.GILock), so it needs no lock of its own.
type keepAliveService struct {
	name             string           // cache key of the loaded model, "" when none
	model            keepable         // nil when none
	param            types.ModelParam // params the cache keys on
	lastSession      utils.SessionKey // last request handled by automatic reset
	lastSessionValid bool             // false after managed or legacy state mutation
	lastActivity     time.Time        // when the last model request finished
	stopCh           chan struct{}
}

// keepable is a model the cache can free; resettable can also be reset.
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
		keepAlive.param = types.ModelParam{}
		keepAlive.lastSession = nil
		keepAlive.lastSessionValid = false
		keepAliveGeneration.Add(1)
	}
}

type acquireMode uint8

const (
	acquireAutomatic acquireMode = iota
	acquireManagedReset
	acquireManagedReuse
	acquireLegacy
)

func acquisitionFresh(mode acquireMode, lastValid bool, last, next utils.SessionKey) (bool, error) {
	switch mode {
	case acquireAutomatic:
		return !lastValid || !utils.IsContinuation(last, next), nil
	case acquireManagedReset:
		return true, nil
	case acquireManagedReuse, acquireLegacy:
		return false, nil
	default:
		return false, fmt.Errorf("unsupported keepalive acquisition mode: %d", mode)
	}
}

// keepAliveGet reuses the cached model when safe, otherwise resets or loads a
// fresh one. Runs under the request GIL, so no locking is needed here.
func keepAliveGet[T any](name string, param types.ModelParam, session utils.SessionKey, mode acquireMode) (any, bool, error) {
	// The SDK resolves bare names / aliases and picks the default precision
	// when none is given; pass the request string through verbatim.
	paths, err := geniex_sdk.ModelGetPaths(name)
	if err != nil {
		return nil, false, err
	}
	slog.Debug("KeepAliveGet", "name", name, "param", param, "model_path", paths.ModelPath)

	modelfile := paths.ModelPath

	if keepAlive.name == name && reflect.DeepEqual(keepAlive.param, param) {
		fresh, modeErr := acquisitionFresh(mode, keepAlive.lastSessionValid, keepAlive.lastSession, session)
		if modeErr != nil {
			return nil, false, modeErr
		}
		if mode != acquireAutomatic {
			keepAlive.lastSession = nil
			keepAlive.lastSessionValid = false
		}
		if fresh {
			if err := ResetKeepAlive(); err != nil {
				return nil, false, err
			}
			// A non-resettable model was destroyed to fail closed. Fall through
			// and reload it rather than returning the now-nil handle.
			if keepAlive.model == nil {
				return keepAliveGet[T](name, param, session, modeWithoutReset(mode))
			}
		}
		if mode == acquireAutomatic {
			keepAlive.lastSession = append(utils.SessionKey(nil), session...)
			keepAlive.lastSessionValid = true
		}
		return keepAlive.model, fresh, nil
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
		return nil, false, fmt.Errorf("unsupported model type: %s", reflect.TypeFor[T]())
	}
	if e != nil {
		return nil, false, e
	}
	keepAlive.name = name
	keepAlive.model = t
	keepAlive.param = param
	if mode == acquireAutomatic {
		keepAlive.lastSession = append(utils.SessionKey(nil), session...)
		keepAlive.lastSessionValid = true
	} else {
		keepAlive.lastSession = nil
		keepAlive.lastSessionValid = false
	}
	keepAliveGeneration.Add(1)

	return t, true, nil
}

func modeWithoutReset(mode acquireMode) acquireMode {
	if mode == acquireManagedReset {
		return acquireManagedReuse
	}
	return mode
}

// stop ends the sweep goroutine and frees the cached model — here rather than in
// the goroutine, so it lands before the SDK deinit that follows.
func (keepAlive *keepAliveService) stop() {
	close(keepAlive.stopCh)
	middleware.GILock.Lock()
	defer middleware.GILock.Unlock()
	keepAlive.destroy()
}
