// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package service

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"reflect"
	"sync"
	"time"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/internal/config"
	"github.com/qualcomm/GenieX/cli/internal/render"
	"github.com/qualcomm/GenieX/cli/internal/types"
)

var (
	computeWarningMu   sync.Mutex
	computeWarningSink io.Writer = os.Stdout
	computeWarned      = map[string]struct{}{}
)

func SetComputeWarningSinkForTest(w io.Writer) func() {
	computeWarningMu.Lock()
	prevSink := computeWarningSink
	computeWarningSink = w
	prevWarned := computeWarned
	computeWarned = map[string]struct{}{}
	computeWarningMu.Unlock()

	return func() {
		computeWarningMu.Lock()
		computeWarningSink = prevSink
		computeWarned = prevWarned
		computeWarningMu.Unlock()
	}
}

func emitComputeCoercionWarning(runtimeID, requestedCompute, warning string) {
	key := runtimeID + "\x00" + requestedCompute
	computeWarningMu.Lock()
	if _, seen := computeWarned[key]; seen {
		computeWarningMu.Unlock()
		return
	}
	computeWarned[key] = struct{}{}
	sink := computeWarningSink
	computeWarningMu.Unlock()
	fmt.Fprintln(sink, render.GetTheme().Warning.Sprintf("Warning: %s", warning))
}

// ResolveModelParam turns the (nctx, ngl, compute) knobs into the ModelParam
// the keep-alive cache keys on. The caller passes already-resolved values (the
// handler prefills unset request fields with the server-wide --nctx / --ngl /
// --compute defaults). NCtx / NGpuLayers are meaningful only for llama_cpp; for
// other plugins (e.g. qairt) NCtx is zeroed here and the SDK zeroes ngl so the
// plugin's param-guard is not tripped. Compute is resolved to a concrete
// DeviceID by the SDK (sdk/src/device.cpp); coercion warnings are logged.
// SpecParam bundles the speculative-decoding knobs sourced from a request; all
// zero-values mean "spec disabled". Only llama_cpp consumes these fields.
type SpecParam struct {
	Type       string
	DraftModel string
	NMax       int32
	NMin       int32
	PMin       float32
}

// resolveDraftModelPath maps a request's spec_draft_model value to an absolute
// GGUF path. An existing filesystem path is returned as-is; anything else is
// treated as a catalogue name (optionally suffixed with :precision) and looked
// up in the local cache. A missing cache entry returns an error - the caller
// is expected to `geniex pull` the draft model beforehand, matching how the
// target model is handled (server never auto-pulls; it consumes what's cached).
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

func ResolveModelParam(runtimeID, modelName string, reqNCtx, reqNgl int32, reqCompute string, spec SpecParam) (types.ModelParam, error) {
	// nctx / ngl / compute already carry the resolved value (explicit request
	// or the server default prefilled by the handler). Non-llama_cpp plugins
	// (e.g. qairt) reject non-zero nctx, so zero it for them; the SDK does the
	// same for ngl in geniex_resolve_device.
	nctx, ngl := reqNCtx, reqNgl
	if runtimeID != geniex_sdk.RuntimeLlamaCpp {
		nctx = 0
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
		emitComputeCoercionWarning(runtimeID, reqCompute, resolved.Warning)
	}

	mp := types.ModelParam{
		NCtx:       nctx,
		NGpuLayers: resolved.Ngl,
		DeviceID:   resolved.DeviceID,
	}
	if runtimeID == geniex_sdk.RuntimeLlamaCpp {
		mp.SpecType = spec.Type
		mp.SpecDraftModel = spec.DraftModel
		mp.SpecNMax = spec.NMax
		mp.SpecNMin = spec.NMin
		mp.SpecPMin = spec.PMin
	}
	return mp, nil
}

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

	// param already carries the resolved NCtx / NGpuLayers / DeviceID (see
	// resolveServeModelParam in the chat handler); the cache keys on it, so no
	// further resolution happens here.
	var t keepable
	var e error
	switch reflect.TypeFor[T]() {
	case reflect.TypeFor[geniex_sdk.LLM]():
		draftPath := ""
		if param.SpecType != "" && param.SpecDraftModel != "" {
			p, perr := resolveDraftModelPath(param.SpecDraftModel)
			if perr != nil {
				return nil, perr
			}
			draftPath = p
		}
		t, e = geniex_sdk.NewLLM(geniex_sdk.LlmCreateInput{
			ModelName: paths.ModelName,
			ModelPath: modelfile,
			DeviceID:  param.DeviceID,
			Config: geniex_sdk.ModelConfig{
				NCtx:           param.NCtx,
				NGpuLayers:     param.NGpuLayers,
				SpecType:       param.SpecType,
				SpecDraftModel: draftPath,
				SpecNMax:       param.SpecNMax,
				SpecNMin:       param.SpecNMin,
				SpecPMin:       param.SpecPMin,
			},
			RuntimeID: paths.RuntimeID,
		})
	case reflect.TypeFor[geniex_sdk.VLM]():
		t, e = geniex_sdk.NewVLM(geniex_sdk.VlmCreateInput{
			ModelName:  paths.ModelName,
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
