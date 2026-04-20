# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bound C functions from libgeniex with argtypes/restype annotations."""

from ctypes import POINTER, byref, c_char_p, c_int32, c_void_p

from ._lib import load_library
from ._types import (
    ml_GetDeviceListInput,
    ml_GetDeviceListOutput,
    ml_GetPluginListOutput,
    ml_KvCacheLoadInput,
    ml_KvCacheLoadOutput,
    ml_KvCacheSaveInput,
    ml_KvCacheSaveOutput,
    ml_LlmApplyChatTemplateInput,
    ml_LlmApplyChatTemplateOutput,
    ml_LlmCreateInput,
    ml_LlmGenerateInput,
    ml_LlmGenerateOutput,
    ml_VlmApplyChatTemplateInput,
    ml_VlmApplyChatTemplateOutput,
    ml_VlmCreateInput,
    ml_VlmGenerateInput,
    ml_VlmGenerateOutput,
)

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class GeniexError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(f'GeniexError({code}): {message}')
        self.code = code


def _check(code: int) -> None:
    if code < 0:
        lib = load_library()
        msg_bytes = lib.ml_get_error_message(c_int32(code))
        msg = msg_bytes.decode() if msg_bytes else 'unknown error'
        raise GeniexError(code, msg)


# ---------------------------------------------------------------------------
# Function bindings
# ---------------------------------------------------------------------------


def _bind_all() -> None:
    lib = load_library()

    lib.ml_get_error_message.argtypes = [c_int32]
    lib.ml_get_error_message.restype = c_char_p

    lib.ml_init.argtypes = []
    lib.ml_init.restype = c_int32

    lib.ml_deinit.argtypes = []
    lib.ml_deinit.restype = c_int32

    lib.ml_free.argtypes = [c_void_p]
    lib.ml_free.restype = None

    lib.ml_version.argtypes = []
    lib.ml_version.restype = c_char_p

    lib.ml_get_plugin_list.argtypes = [POINTER(ml_GetPluginListOutput)]
    lib.ml_get_plugin_list.restype = c_int32

    lib.ml_get_device_list.argtypes = [POINTER(ml_GetDeviceListInput), POINTER(ml_GetDeviceListOutput)]
    lib.ml_get_device_list.restype = c_int32

    # LLM
    lib.ml_llm_create.argtypes = [POINTER(ml_LlmCreateInput), POINTER(c_void_p)]
    lib.ml_llm_create.restype = c_int32

    lib.ml_llm_destroy.argtypes = [c_void_p]
    lib.ml_llm_destroy.restype = c_int32

    lib.ml_llm_reset.argtypes = [c_void_p]
    lib.ml_llm_reset.restype = c_int32

    lib.ml_llm_generate.argtypes = [c_void_p, POINTER(ml_LlmGenerateInput), POINTER(ml_LlmGenerateOutput)]
    lib.ml_llm_generate.restype = c_int32

    lib.ml_llm_apply_chat_template.argtypes = [
        c_void_p,
        POINTER(ml_LlmApplyChatTemplateInput),
        POINTER(ml_LlmApplyChatTemplateOutput),
    ]
    lib.ml_llm_apply_chat_template.restype = c_int32

    lib.ml_llm_save_kv_cache.argtypes = [c_void_p, POINTER(ml_KvCacheSaveInput), POINTER(ml_KvCacheSaveOutput)]
    lib.ml_llm_save_kv_cache.restype = c_int32

    lib.ml_llm_load_kv_cache.argtypes = [c_void_p, POINTER(ml_KvCacheLoadInput), POINTER(ml_KvCacheLoadOutput)]
    lib.ml_llm_load_kv_cache.restype = c_int32

    # VLM
    lib.ml_vlm_create.argtypes = [POINTER(ml_VlmCreateInput), POINTER(c_void_p)]
    lib.ml_vlm_create.restype = c_int32

    lib.ml_vlm_destroy.argtypes = [c_void_p]
    lib.ml_vlm_destroy.restype = c_int32

    lib.ml_vlm_reset.argtypes = [c_void_p]
    lib.ml_vlm_reset.restype = c_int32

    lib.ml_vlm_generate.argtypes = [c_void_p, POINTER(ml_VlmGenerateInput), POINTER(ml_VlmGenerateOutput)]
    lib.ml_vlm_generate.restype = c_int32

    lib.ml_vlm_apply_chat_template.argtypes = [
        c_void_p,
        POINTER(ml_VlmApplyChatTemplateInput),
        POINTER(ml_VlmApplyChatTemplateOutput),
    ]
    lib.ml_vlm_apply_chat_template.restype = c_int32


_bound = False


def _ensure_bound() -> None:
    global _bound
    if not _bound:
        _bind_all()
        _bound = True


# ---------------------------------------------------------------------------
# Init lifecycle
# ---------------------------------------------------------------------------

_initialized = False


def init() -> None:
    global _initialized
    if _initialized:
        return
    _ensure_bound()
    lib = load_library()
    _check(lib.ml_init())
    _initialized = True


def deinit() -> None:
    global _initialized
    if not _initialized:
        return
    lib = load_library()
    lib.ml_deinit()
    _initialized = False


def ensure_init() -> None:
    if not _initialized:
        init()


# ---------------------------------------------------------------------------
# Helpers: convert Python string lists ↔ C char** arrays
# ---------------------------------------------------------------------------


def _encode(s: str | None) -> bytes | None:
    return s.encode() if s else None


def _str_list_to_c(strings: list[str]):
    """Return (array, count) of c_char_p array allocated from Python bytes."""
    from ctypes import cast, create_string_buffer

    count = len(strings)
    if count == 0:
        return None, 0
    arr_type = c_char_p * count
    arr = arr_type(*[s.encode() for s in strings])
    return arr, count


# ---------------------------------------------------------------------------
# Plugin / device helpers
# ---------------------------------------------------------------------------


def get_plugin_list() -> list[str]:
    _ensure_bound()
    lib = load_library()
    out = ml_GetPluginListOutput()
    _check(lib.ml_get_plugin_list(byref(out)))
    result = [out.plugin_ids[i].decode() for i in range(out.plugin_count)]
    if out.plugin_ids:
        lib.ml_free(out.plugin_ids)
    return result


def get_device_list(plugin_id: str) -> list[tuple[str, str]]:
    _ensure_bound()
    lib = load_library()
    inp = ml_GetDeviceListInput(plugin_id=plugin_id.encode())
    out = ml_GetDeviceListOutput()
    _check(lib.ml_get_device_list(byref(inp), byref(out)))
    result = [(out.device_ids[i].decode(), out.device_names[i].decode()) for i in range(out.device_count)]
    if out.device_ids:
        lib.ml_free(out.device_ids)
    if out.device_names:
        lib.ml_free(out.device_names)
    return result


def version() -> str:
    _ensure_bound()
    lib = load_library()
    return lib.ml_version().decode()
