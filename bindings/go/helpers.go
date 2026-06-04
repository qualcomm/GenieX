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

package geniex_sdk

/*
#include <stdlib.h>
#include <string.h>
#include "geniex.h"
*/
import "C"

import (
	"log/slog"
	"runtime/cgo"
	"unsafe"
)

// LCOV_EXCL_START

// free releases memory the SDK allocated (geniex_free), as opposed to
// freeC* helpers which release memory Go allocated via C.malloc / C.CString.
func free(ptr unsafe.Pointer) {
	C.geniex_free(ptr)
}

// cMalloc returns zero-initialized memory so partial field assignment on the
// returned struct doesn't leave stale bytes the C side reads as dangling
// pointers or bogus counts.
func cMalloc(size C.size_t) unsafe.Pointer {
	raw := C.malloc(size)
	if raw == nil {
		panic("C.malloc failed")
	}
	C.memset(raw, 0, size)
	return raw
}

func cStringIfSet(s string) *C.char {
	if s == "" {
		return nil
	}
	return C.CString(s)
}

// cCString / cGoString wrap C.CString and C.GoString so in-package tests can
// avoid `import "C"` (cmd/go forbids cgo in *_test.go files within the package).
func cCString(s string) *C.char  { return C.CString(s) }
func cGoString(p *C.char) string { return C.GoString(p) }

func cFreeIfSet(p unsafe.Pointer) {
	if p != nil {
		C.free(p)
	}
}

func sliceToCCharArray(slice []string) (**C.char, C.int32_t) {
	if len(slice) == 0 {
		return nil, 0
	}
	raw := cMalloc(C.size_t(len(slice)) * C.size_t(unsafe.Sizeof(uintptr(0))))
	cArray := unsafe.Slice((**C.char)(raw), len(slice))
	for i, s := range slice {
		cArray[i] = C.CString(s)
	}
	return (**C.char)(raw), C.int32_t(len(slice))
}

func cCharArrayToSlice(ptr **C.char, count C.int32_t) []string {
	if ptr == nil || count == 0 {
		return nil
	}
	arr := unsafe.Slice(ptr, int(count))
	out := make([]string, count)
	for i, cstr := range arr {
		out[i] = C.GoString(cstr)
	}
	return out
}

// freeCCharArray releases an array Go allocated; mlFreeCCharArray below is
// the SDK-allocated equivalent. Mixing them up corrupts the heap.
func freeCCharArray(ptr **C.char, count C.int32_t) {
	if ptr == nil || count == 0 {
		return
	}
	arr := unsafe.Slice(ptr, int(count))
	for _, p := range arr {
		C.free(unsafe.Pointer(p))
	}
	C.free(unsafe.Pointer(ptr))
}

func mlFreeCCharArray(ptr **C.char, count C.int32_t) {
	if ptr == nil || count == 0 {
		return
	}
	arr := unsafe.Slice(ptr, int(count))
	for _, p := range arr {
		free(unsafe.Pointer(p))
	}
	free(unsafe.Pointer(ptr))
}

// OnTokenCallback is dispatched per generated token; returning false stops
// generation. The Handle is round-tripped via the SDK's user_data so concurrent
// generations on different handles don't share state.
type OnTokenCallback func(token string) bool

// handleToUserData / userDataToHandle bridge cgo.Handle (a uintptr identifier,
// not a Go pointer) and C's void*. The trip through unsafe.Pointer is required
// by cgo's void* representation; isolating it here keeps `go vet` quiet at the
// many call sites.
func handleToUserData(h cgo.Handle) unsafe.Pointer {
	return *(*unsafe.Pointer)(unsafe.Pointer(&h))
}

func userDataToHandle(p unsafe.Pointer) cgo.Handle {
	return *(*cgo.Handle)(unsafe.Pointer(&p))
}

//export go_generate_stream_on_token
func go_generate_stream_on_token(token *C.char, userData unsafe.Pointer) C.bool {
	if userData == nil {
		return C.bool(true)
	}
	cb, ok := userDataToHandle(userData).Value().(OnTokenCallback)
	if !ok || cb == nil {
		slog.Error("on_token callback dispatch failed: invalid cgo.Handle or nil callback",
			"user_data", userData, "ok", ok, "cb_nil", cb == nil)
		return C.bool(true)
	}
	return C.bool(cb(C.GoString(token)))
}

// LCOV_EXCL_STOP
