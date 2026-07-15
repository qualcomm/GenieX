// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package geniex_sdk

/*
#include <stdlib.h>
#include "geniex_model.h"

extern bool go_model_progress(geniex_FileProgress*, int32_t, void*);
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime/cgo"
	"strings"
	"unsafe"
)

// LCOV_EXCL_START

// PrecisionNA is the precision placeholder the SDK records for non-quantized
// (e.g. QAIRT) models.
const PrecisionNA = "N/A"

// ModelType mirrors geniex_ModelType.
type ModelType int32

const (
	ModelTypeLLM ModelType = C.GENIEX_MODEL_TYPE_LLM
	ModelTypeVLM ModelType = C.GENIEX_MODEL_TYPE_VLM
)

func (t ModelType) String() string {
	switch t {
	case ModelTypeLLM:
		return "llm"
	case ModelTypeVLM:
		return "vlm"
	default:
		return "unknown"
	}
}

// ParseModelType maps "llm"/"vlm" (case-insensitive) to a ModelType. The
// second return is false for any other value.
func ParseModelType(s string) (ModelType, bool) {
	switch strings.ToLower(s) {
	case "llm":
		return ModelTypeLLM, true
	case "vlm":
		return ModelTypeVLM, true
	default:
		return 0, false
	}
}

// SplitNamePrecision splits "name[:precision]" into (name, precision). A pasted
// HuggingFace URL prefix is stripped first so its scheme colon isn't mistaken
// for the precision separator. Name canonicalization, precision case-folding
// (GGUF quant labels are matched upper-cased), and hub routing all happen in
// the SDK across the FFI boundary — including Docker Hub, where the suffix is a
// case-sensitive registry tag the SDK deliberately leaves untouched. This only
// does the URL strip + ':' split so callers can pass the two to the FFI
// separately.
func SplitNamePrecision(arg string) (string, string) {
	arg = strings.TrimPrefix(arg, "https://huggingface.co/")
	arg = strings.TrimPrefix(arg, "http://huggingface.co/")
	name, precision, found := strings.Cut(arg, ":")
	if !found || precision == "" {
		return name, ""
	}
	return name, precision
}

// HubSource mirrors geniex_HubSource.
type HubSource int32

const (
	HubAuto        HubSource = C.GENIEX_HUB_AUTO
	HubHuggingFace HubSource = C.GENIEX_HUB_HUGGINGFACE
	HubModelScope  HubSource = C.GENIEX_HUB_MODELSCOPE
	HubAIHub       HubSource = C.GENIEX_HUB_AIHUB
	HubVolces      HubSource = C.GENIEX_HUB_VOLCES
	HubDocker      HubSource = C.GENIEX_HUB_DOCKER
	HubLocalFS     HubSource = C.GENIEX_HUB_LOCALFS
)

// ResolveHub reports the hub a pull/query will actually use for name given the
// requested hub. HubAuto resolves to HubDocker when name carries a Docker Hub
// prefix (docker.io/…); every other input is returned unchanged. The prefix
// table lives in the SDK, so callers must not re-derive it. No network I/O.
func ResolveHub(name string, hub HubSource) (HubSource, error) {
	cName := cStringIfSet(name)
	defer cFreeIfSet(unsafe.Pointer(cName))
	var out C.geniex_HubSource
	if res := C.geniex_model_resolve_hub(cName, C.geniex_HubSource(hub), &out); res != C.GENIEX_SUCCESS {
		return hub, modelError(res)
	}
	return HubSource(out), nil
}

// FileProgress mirrors geniex_FileProgress.
type FileProgress struct {
	FileName        string
	DownloadedBytes int64
	TotalBytes      int64 // -1 if unknown
}

// ModelProgressCallback is invoked periodically during a pull; returning
// false cancels the download.
type ModelProgressCallback func(files []FileProgress) bool

// ModelInit initializes the model manager. dataDir precedence:
// argument → GENIEX_DATADIR env → ~/.cache/geniex. Idempotent: a second
// call returns nil (the SDK reports ALREADY_INITIALIZED, treated as success).
func ModelInit(dataDir string) error {
	cDir := cStringIfSet(dataDir)
	defer cFreeIfSet(unsafe.Pointer(cDir))
	res := C.geniex_model_init(cDir)
	if res != C.GENIEX_SUCCESS && res != C.GENIEX_ERROR_COMMON_ALREADY_INITIALIZED {
		return modelError(res)
	}
	return nil
}

// ModelDeinit releases model-manager resources (a no-op in the SDK today).
func ModelDeinit() error {
	if res := C.geniex_model_deinit(); res != C.GENIEX_SUCCESS {
		return modelError(res)
	}
	return nil
}

// ModelPullInput mirrors geniex_ModelPullInput (minus the ABI gate, which
// ModelPull sets automatically).
type ModelPullInput struct {
	ModelName   string
	Precision   string
	Hub         HubSource
	LocalPath   string
	HfToken     string
	Chipset     string
	DisplayName string
	OnProgress  ModelProgressCallback
	// ModelType selects the model type to pull; nil means auto-detect.
	ModelType *ModelType
}

//export go_model_progress
func go_model_progress(files *C.geniex_FileProgress, count C.int32_t, userData unsafe.Pointer) C.bool {
	if userData == nil {
		return C.bool(true)
	}
	cb, ok := userDataToHandle(userData).Value().(ModelProgressCallback)
	if !ok || cb == nil {
		return C.bool(true)
	}
	entries := unsafe.Slice(files, int(count))
	items := make([]FileProgress, count)
	for i, e := range entries {
		items[i] = FileProgress{
			FileName:        C.GoString(e.file_name),
			DownloadedBytes: int64(e.downloaded_bytes),
			TotalBytes:      int64(e.total_bytes),
		}
	}
	return C.bool(cb(items))
}

// ModelPull downloads a model into the local cache (blocking, resumable).
func ModelPull(input ModelPullInput) error {
	cModelName := cStringIfSet(input.ModelName)
	cQuant := cStringIfSet(input.Precision)
	cLocalPath := cStringIfSet(input.LocalPath)
	cHfToken := cStringIfSet(input.HfToken)
	cChipset := cStringIfSet(input.Chipset)
	cDisplayName := cStringIfSet(input.DisplayName)
	defer func() {
		cFreeIfSet(unsafe.Pointer(cModelName))
		cFreeIfSet(unsafe.Pointer(cQuant))
		cFreeIfSet(unsafe.Pointer(cLocalPath))
		cFreeIfSet(unsafe.Pointer(cHfToken))
		cFreeIfSet(unsafe.Pointer(cChipset))
		cFreeIfSet(unsafe.Pointer(cDisplayName))
	}()

	in := C.geniex_ModelPullInput{
		struct_size:  C.uint32_t(C.sizeof_geniex_ModelPullInput),
		model_name:   cModelName,
		quant:        cQuant,
		hub:          C.geniex_HubSource(input.Hub),
		local_path:   cLocalPath,
		hf_token:     cHfToken,
		chipset:      cChipset,
		display_name: cDisplayName,
		model_type:   C.int32_t(C.GENIEX_MODEL_TYPE_AUTO),
	}
	if input.ModelType != nil {
		in.model_type = C.int32_t(*input.ModelType)
	}

	if input.OnProgress != nil {
		h := cgo.NewHandle(input.OnProgress)
		defer h.Delete()
		in.on_progress = C.geniex_download_progress_cb(C.go_model_progress)
		in.user_data = handleToUserData(h)
	}

	if res := C.geniex_model_pull(&in); res != C.GENIEX_SUCCESS {
		return modelError(res)
	}
	return nil
}

// ModelDetail mirrors geniex_ModelDetail.
type ModelDetail struct {
	Name       string
	ModelName  string
	RuntimeID  string
	ModelType  ModelType
	TotalSize  int64
	Precisions []string
}

// ModelListDetailed returns cached models with full metadata.
func ModelListDetailed() ([]ModelDetail, error) {
	var out C.geniex_ModelListDetailedOutput
	if res := C.geniex_model_list_detailed(&out); res != C.GENIEX_SUCCESS {
		return nil, modelError(res)
	}
	defer C.geniex_model_list_detailed_free(&out)

	if out.models == nil || out.count == 0 {
		return nil, nil
	}
	models := unsafe.Slice(out.models, int(out.count))
	result := make([]ModelDetail, out.count)
	for i, m := range models {
		result[i] = ModelDetail{
			Name:       C.GoString(m.name),
			ModelName:  C.GoString(m.model_name),
			RuntimeID:  C.GoString(m.plugin_id),
			ModelType:  ModelType(m.model_type),
			TotalSize:  int64(m.total_size),
			Precisions: cCharArrayToSlice(m.precisions, m.precision_count),
		}
	}
	return result, nil
}

// ChipsetInfo mirrors geniex_ChipsetInfo.
type ChipsetInfo struct {
	Name    string
	Aliases []string
}

// ModelListChipsets lists every chipset Qualcomm AI Hub supports, with aliases.
// Sourced from platform.json (cached 24h); the first call may hit the network.
func ModelListChipsets() ([]ChipsetInfo, error) {
	var out C.geniex_ChipsetList
	if res := C.geniex_model_list_chipsets(&out); res != C.GENIEX_SUCCESS {
		return nil, modelError(res)
	}
	defer C.geniex_model_list_chipsets_free(&out)

	if out.chipsets == nil || out.count == 0 {
		return nil, nil
	}
	chipsets := unsafe.Slice(out.chipsets, int(out.count))
	result := make([]ChipsetInfo, out.count)
	for i, c := range chipsets {
		result[i] = ChipsetInfo{
			Name:    C.GoString(c.name),
			Aliases: cCharArrayToSlice(c.aliases, c.alias_count),
		}
	}
	return result, nil
}

// ModelDetectChipset probes the current host for its chipset via a local
// detector (no network). Returns "" when the platform can't be probed.
func ModelDetectChipset() (string, error) {
	var out *C.char
	if res := C.geniex_model_detect_chipset(&out); res != C.GENIEX_SUCCESS {
		return "", modelError(res)
	}
	if out == nil {
		return "", nil
	}
	defer free(unsafe.Pointer(out))
	return C.GoString(out), nil
}

// ModelRemove deletes a cached model ("org/repo" or "org/repo:precision").
func ModelRemove(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	if res := C.geniex_model_remove(cName); res != C.GENIEX_SUCCESS {
		return modelError(res)
	}
	return nil
}

// ModelClean removes all cached models, returning the number removed.
func ModelClean() (int, error) {
	var count C.int32_t
	if res := C.geniex_model_clean(&count); res != C.GENIEX_SUCCESS {
		return 0, modelError(res)
	}
	return int(count), nil
}

// ModelGetType returns the model type of a cached model.
func ModelGetType(name string) (ModelType, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var t C.geniex_ModelType
	if res := C.geniex_model_get_type(cName, &t); res != C.GENIEX_SUCCESS {
		return 0, modelError(res)
	}
	return ModelType(t), nil
}

// ModelSetType overrides the stored model type of a cached model.
func ModelSetType(name string, t ModelType) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	if res := C.geniex_model_set_type(cName, C.geniex_ModelType(t)); res != C.GENIEX_SUCCESS {
		return modelError(res)
	}
	return nil
}

// ModelPaths mirrors geniex_ModelPaths. Optional fields are empty when unset.
type ModelPaths struct {
	ModelPath     string
	MmprojPath    string
	TokenizerPath string
	ModelDir      string
	ModelName     string
	RuntimeID     string
	ModelType     ModelType
	// QairtVersion is the QAIRT version the model's assets were pulled for
	// (from AI Hub metadata); empty for non-AI-Hub / llama.cpp models.
	QairtVersion string
}

// ModelGetPaths resolves "org/repo[:precision]" (or alias) to absolute on-disk paths.
func ModelGetPaths(name string) (*ModelPaths, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var out C.geniex_ModelPaths
	if res := C.geniex_model_get_paths(cName, &out); res != C.GENIEX_SUCCESS {
		return nil, modelError(res)
	}
	defer C.geniex_model_paths_free(&out)
	return &ModelPaths{
		ModelPath:     C.GoString(out.model_path),
		MmprojPath:    C.GoString(out.mmproj_path),
		TokenizerPath: C.GoString(out.tokenizer_path),
		ModelDir:      C.GoString(out.model_dir),
		ModelName:     C.GoString(out.model_name),
		RuntimeID:     C.GoString(out.plugin_id),
		ModelType:     ModelType(out.model_type),
		QairtVersion:  C.GoString(out.qairt_version),
	}, nil
}

// PrecisionCandidate mirrors geniex_QuantCandidate.
type PrecisionCandidate struct {
	Precision string
	Size      int64
	IsDefault bool
}

// ModelQueryResult mirrors geniex_ModelQueryOutput.
type ModelQueryResult struct {
	ModelName  string
	RuntimeID  string
	ModelType  ModelType
	Candidates []PrecisionCandidate
}

// ModelQuery resolves a model's remote candidate precisions without
// downloading. Reuses ModelPullInput's fields; OnProgress and Precision are
// ignored.
func ModelQuery(input ModelPullInput) (*ModelQueryResult, error) {
	cModelName := cStringIfSet(input.ModelName)
	cLocalPath := cStringIfSet(input.LocalPath)
	cHfToken := cStringIfSet(input.HfToken)
	cChipset := cStringIfSet(input.Chipset)
	cDisplayName := cStringIfSet(input.DisplayName)
	defer func() {
		cFreeIfSet(unsafe.Pointer(cModelName))
		cFreeIfSet(unsafe.Pointer(cLocalPath))
		cFreeIfSet(unsafe.Pointer(cHfToken))
		cFreeIfSet(unsafe.Pointer(cChipset))
		cFreeIfSet(unsafe.Pointer(cDisplayName))
	}()

	// Query reuses the pull input struct; precision / progress / model_type are
	// ignored by the SDK for a plan-only call.
	in := C.geniex_ModelPullInput{
		struct_size:  C.uint32_t(C.sizeof_geniex_ModelPullInput),
		model_name:   cModelName,
		hub:          C.geniex_HubSource(input.Hub),
		local_path:   cLocalPath,
		hf_token:     cHfToken,
		chipset:      cChipset,
		display_name: cDisplayName,
		model_type:   C.int32_t(C.GENIEX_MODEL_TYPE_AUTO),
	}

	var out C.geniex_ModelQueryOutput
	if res := C.geniex_model_query(&in, &out); res != C.GENIEX_SUCCESS {
		return nil, modelError(res)
	}
	defer C.geniex_model_query_free(&out)

	result := &ModelQueryResult{
		ModelName: C.GoString(out.model_name),
		RuntimeID: C.GoString(out.plugin_id),
		ModelType: ModelType(out.model_type),
	}
	if out.candidates != nil && out.candidate_count > 0 {
		cands := unsafe.Slice(out.candidates, int(out.candidate_count))
		result.Candidates = make([]PrecisionCandidate, out.candidate_count)
		for i, c := range cands {
			result.Candidates[i] = PrecisionCandidate{
				Precision: C.GoString(c.quant),
				Size:      int64(c.size),
				IsDefault: bool(c.is_default),
			}
		}
	}
	return result, nil
}

// ModelLastErrorMessage returns the library-owned (do not free) message for the
// last failing geniex_model_* call on this thread, or "" if none.
func ModelLastErrorMessage() string {
	return C.GoString(C.geniex_model_last_error_message())
}

// modelError wraps an FFI error code with the SDK's detailed last-error message
// (e.g. "quantization 'Q2_K' not found for model 'org/repo'") when available,
// while keeping the underlying SDKError matchable via errors.Is / errors.As.
func modelError(res C.int32_t) error {
	err := SDKError(res)
	if msg := ModelLastErrorMessage(); msg != "" {
		return fmt.Errorf("%w, %s", err, msg)
	}
	return err
}

// IsModelNotFound reports whether err is the SDK's "model not cached" code.
func IsModelNotFound(err error) bool {
	var sdkErr SDKError
	return errors.As(err, &sdkErr) && int32(sdkErr) == int32(C.GENIEX_ERROR_COMMON_FILE_NOT_FOUND)
}

// LCOV_EXCL_STOP
