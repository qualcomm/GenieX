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

import atexit
import logging
import os
from ctypes import CFUNCTYPE, POINTER, byref, c_char_p, c_int32, c_void_p

from ._lib import load_library
from ._types import (
    geniex_GetDeviceListInput,
    geniex_GetDeviceListOutput,
    geniex_GetPluginListOutput,
    geniex_KvCacheLoadInput,
    geniex_KvCacheLoadOutput,
    geniex_KvCacheSaveInput,
    geniex_KvCacheSaveOutput,
    geniex_LlmApplyChatTemplateInput,
    geniex_LlmApplyChatTemplateOutput,
    geniex_LlmCreateInput,
    geniex_LlmGenerateInput,
    geniex_LlmGenerateOutput,
    geniex_ModelListOutput,
    geniex_ModelPaths,
    geniex_ModelPullInput,
    geniex_ResolveDeviceInput,
    geniex_ResolveDeviceOutput,
    geniex_VlmApplyChatTemplateInput,
    geniex_VlmApplyChatTemplateOutput,
    geniex_VlmCapabilities,
    geniex_VlmCreateInput,
    geniex_VlmGenerateInput,
    geniex_VlmGenerateOutput,
)

# Mirrors geniex_LogLevel in sdk/include/geniex.h — values MUST match.
_LEVEL_TRACE = 0
_LEVEL_DEBUG = 1
_LEVEL_INFO = 2
_LEVEL_WARN = 3
_LEVEL_ERROR = 4
_LEVEL_NONE = 5  # sentinel above ERROR silences SDK output entirely

_LEVEL_STR_TO_PY = {
    'trace': logging.DEBUG,
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warn': logging.WARNING,
    'error': logging.ERROR,
    'none': logging.CRITICAL + 1,
}

geniex_log_callback = CFUNCTYPE(None, c_int32, c_char_p)

_logger = logging.getLogger('geniex')
_logger.addHandler(logging.NullHandler())

# Strong reference — ctypes callbacks must outlive the C side.
_log_cb_ref: 'geniex_log_callback | None' = None


def _sdk_log_bridge(level: int, msg_bytes):
    try:
        msg = msg_bytes.decode('utf-8', errors='replace') if msg_bytes else ''
    except Exception:
        msg = '<undecodable log message>'
    if level == _LEVEL_ERROR:
        _logger.error(msg)
    elif level == _LEVEL_WARN:
        _logger.warning(msg)
    elif level == _LEVEL_INFO:
        _logger.info(msg)
    else:
        _logger.debug(msg)


def set_log_level(level: str) -> None:
    """Set the ``geniex`` Python logger level.

    Accepts ``'trace' | 'debug' | 'info' | 'warn' | 'error' | 'none'``.
    Unknown values are ignored. Safe to call before or after :func:`init`.
    """
    level_norm = level.lower().strip() if level else ''
    if level_norm not in _LEVEL_STR_TO_PY:
        return
    _logger.setLevel(_LEVEL_STR_TO_PY[level_norm])


def install_log_callback() -> None:
    """Register the SDK → logging bridge and seed the logger from ``GENIEX_LOG``."""
    global _log_cb_ref
    if _log_cb_ref is not None:
        return

    lib = load_library()

    _log_cb_ref = geniex_log_callback(_sdk_log_bridge)
    lib.geniex_set_log(_log_cb_ref)

    requested = os.environ.get('GENIEX_LOG', '').strip().lower()
    if requested in {'off', '0', 'false'}:
        requested = 'none'
    if requested in _LEVEL_STR_TO_PY:
        _logger.setLevel(_LEVEL_STR_TO_PY[requested])


class GeniexError(Exception):
    """Raised when a geniex C call returns a negative error code."""

    def __init__(self, code: int, message: str):
        super().__init__(f'GeniexError({code}): {message}')
        self.code = code


# Subset of geniex_ErrorCode values that callers may want to check against.
# Keep aligned with sdk/include/geniex.h; expand on demand.
GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH = -200004


def _check(code: int) -> None:
    if code < 0:
        lib = load_library()
        msg_bytes = lib.geniex_get_error_message(c_int32(code))
        msg = msg_bytes.decode() if msg_bytes else 'unknown error'
        raise GeniexError(code, msg)


def _bind_all() -> None:
    lib = load_library()

    lib.geniex_get_error_message.argtypes = [c_int32]
    lib.geniex_get_error_message.restype = c_char_p

    lib.geniex_set_log.argtypes = [geniex_log_callback]
    lib.geniex_set_log.restype = c_int32

    lib.geniex_init.argtypes = []
    lib.geniex_init.restype = c_int32

    lib.geniex_deinit.argtypes = []
    lib.geniex_deinit.restype = c_int32

    lib.geniex_free.argtypes = [c_void_p]
    lib.geniex_free.restype = None

    lib.geniex_version.argtypes = []
    lib.geniex_version.restype = c_char_p

    lib.geniex_qairt_version.argtypes = []
    lib.geniex_qairt_version.restype = c_char_p

    lib.geniex_llama_cpp_version.argtypes = []
    lib.geniex_llama_cpp_version.restype = c_char_p

    lib.geniex_get_plugin_list.argtypes = [POINTER(geniex_GetPluginListOutput)]
    lib.geniex_get_plugin_list.restype = c_int32

    lib.geniex_get_device_list.argtypes = [POINTER(geniex_GetDeviceListInput), POINTER(geniex_GetDeviceListOutput)]
    lib.geniex_get_device_list.restype = c_int32

    lib.geniex_resolve_device.argtypes = [POINTER(geniex_ResolveDeviceInput), POINTER(geniex_ResolveDeviceOutput)]
    lib.geniex_resolve_device.restype = c_int32

    # LLM
    lib.geniex_llm_create.argtypes = [POINTER(geniex_LlmCreateInput), POINTER(c_void_p)]
    lib.geniex_llm_create.restype = c_int32

    lib.geniex_llm_destroy.argtypes = [c_void_p]
    lib.geniex_llm_destroy.restype = c_int32

    lib.geniex_llm_reset.argtypes = [c_void_p]
    lib.geniex_llm_reset.restype = c_int32

    lib.geniex_llm_generate.argtypes = [c_void_p, POINTER(geniex_LlmGenerateInput), POINTER(geniex_LlmGenerateOutput)]
    lib.geniex_llm_generate.restype = c_int32

    lib.geniex_llm_apply_chat_template.argtypes = [
        c_void_p,
        POINTER(geniex_LlmApplyChatTemplateInput),
        POINTER(geniex_LlmApplyChatTemplateOutput),
    ]
    lib.geniex_llm_apply_chat_template.restype = c_int32

    lib.geniex_llm_save_kv_cache.argtypes = [
        c_void_p,
        POINTER(geniex_KvCacheSaveInput),
        POINTER(geniex_KvCacheSaveOutput),
    ]
    lib.geniex_llm_save_kv_cache.restype = c_int32

    lib.geniex_llm_load_kv_cache.argtypes = [
        c_void_p,
        POINTER(geniex_KvCacheLoadInput),
        POINTER(geniex_KvCacheLoadOutput),
    ]
    lib.geniex_llm_load_kv_cache.restype = c_int32

    # VLM
    lib.geniex_vlm_create.argtypes = [POINTER(geniex_VlmCreateInput), POINTER(c_void_p)]
    lib.geniex_vlm_create.restype = c_int32

    lib.geniex_vlm_destroy.argtypes = [c_void_p]
    lib.geniex_vlm_destroy.restype = c_int32

    lib.geniex_vlm_reset.argtypes = [c_void_p]
    lib.geniex_vlm_reset.restype = c_int32

    lib.geniex_vlm_generate.argtypes = [c_void_p, POINTER(geniex_VlmGenerateInput), POINTER(geniex_VlmGenerateOutput)]
    lib.geniex_vlm_generate.restype = c_int32

    lib.geniex_vlm_apply_chat_template.argtypes = [
        c_void_p,
        POINTER(geniex_VlmApplyChatTemplateInput),
        POINTER(geniex_VlmApplyChatTemplateOutput),
    ]
    lib.geniex_vlm_apply_chat_template.restype = c_int32

    lib.geniex_vlm_get_capabilities.argtypes = [c_void_p, POINTER(geniex_VlmCapabilities)]
    lib.geniex_vlm_get_capabilities.restype = c_int32

    # Model manager
    lib.geniex_model_init.argtypes = [c_char_p]
    lib.geniex_model_init.restype = c_int32

    lib.geniex_model_deinit.argtypes = []
    lib.geniex_model_deinit.restype = c_int32

    lib.geniex_model_pull.argtypes = [POINTER(geniex_ModelPullInput)]
    lib.geniex_model_pull.restype = c_int32

    lib.geniex_model_list.argtypes = [POINTER(geniex_ModelListOutput)]
    lib.geniex_model_list.restype = c_int32

    lib.geniex_model_list_free.argtypes = [POINTER(geniex_ModelListOutput)]
    lib.geniex_model_list_free.restype = None

    lib.geniex_model_remove.argtypes = [c_char_p]
    lib.geniex_model_remove.restype = c_int32

    lib.geniex_model_clean.argtypes = [POINTER(c_int32)]
    lib.geniex_model_clean.restype = c_int32

    lib.geniex_model_get_paths.argtypes = [c_char_p, POINTER(geniex_ModelPaths)]
    lib.geniex_model_get_paths.restype = c_int32

    lib.geniex_model_paths_free.argtypes = [POINTER(geniex_ModelPaths)]
    lib.geniex_model_paths_free.restype = None

    lib.geniex_model_get_type.argtypes = [c_char_p, POINTER(c_int32)]
    lib.geniex_model_get_type.restype = c_int32

    lib.geniex_model_resolve_alias.argtypes = [c_char_p, POINTER(c_char_p)]
    lib.geniex_model_resolve_alias.restype = c_int32


_bound = False


def _ensure_bound() -> None:
    global _bound
    if not _bound:
        _bind_all()
        _bound = True
        install_log_callback()


_initialized = False


def init() -> None:
    """Initialise the geniex runtime. Idempotent."""
    global _initialized
    if _initialized:
        return
    _ensure_bound()
    lib = load_library()
    _check(lib.geniex_init())
    _initialized = True
    atexit.register(deinit)


def deinit() -> None:
    """Tear down the geniex runtime."""
    global _initialized
    if not _initialized:
        return
    lib = load_library()
    lib.geniex_deinit()
    _initialized = False


def ensure_init() -> None:
    if not _initialized:
        init()


def _encode(s: str | None) -> bytes | None:
    return s.encode() if s else None


def _str_list_to_c(strings: list[str]):
    count = len(strings)
    if count == 0:
        return None, 0
    arr_type = c_char_p * count
    arr = arr_type(*[s.encode() for s in strings])
    return arr, count


def get_plugin_list() -> list[str]:
    """Return the plugin ids registered with libgeniex."""
    _ensure_bound()
    lib = load_library()
    out = geniex_GetPluginListOutput()
    _check(lib.geniex_get_plugin_list(byref(out)))
    result = [out.plugin_ids[i].decode() for i in range(out.plugin_count)]
    if out.plugin_ids:
        lib.geniex_free(out.plugin_ids)
    return result


def get_device_list(plugin_id: str) -> list[tuple[str, str]]:
    """Return ``[(device_id, device_name), ...]`` for ``plugin_id``."""
    _ensure_bound()
    lib = load_library()
    inp = geniex_GetDeviceListInput(plugin_id=plugin_id.encode())
    out = geniex_GetDeviceListOutput()
    _check(lib.geniex_get_device_list(byref(inp), byref(out)))
    result = [(out.device_ids[i].decode(), out.device_names[i].decode()) for i in range(out.device_count)]
    if out.device_ids:
        lib.geniex_free(out.device_ids)
    if out.device_names:
        lib.geniex_free(out.device_names)
    return result


def resolve_device(
    plugin_id: str,
    model_name: str | None,
    mode: str | None,
    ngl_default: int,
) -> tuple[str | None, int, str | None]:
    """Raw SDK device-alias resolver. Prefer :func:`geniex.resolve_device_map`.

    Returns ``(device_id, ngl, warning)``; ``device_id`` / ``warning`` may
    be ``None``. Raises :class:`GeniexError` for unknown aliases.
    """
    _ensure_bound()
    lib = load_library()
    inp = geniex_ResolveDeviceInput(
        plugin_id=plugin_id.encode(),
        model_name=model_name.encode() if model_name else None,
        mode=mode.encode() if mode else None,
        ngl_default=int(ngl_default),
    )
    out = geniex_ResolveDeviceOutput()
    _check(lib.geniex_resolve_device(byref(inp), byref(out)))

    device_id: str | None = None
    warning: str | None = None
    if out.device_id:
        device_id = c_char_p(out.device_id).value.decode()
        lib.geniex_free(out.device_id)
    if out.warning:
        warning = c_char_p(out.warning).value.decode()
        lib.geniex_free(out.warning)
    return device_id, int(out.ngl), warning


def version() -> str:
    """Return the geniex SDK version string."""
    _ensure_bound()
    lib = load_library()
    return lib.geniex_version().decode()


def qairt_version() -> str:
    """Return the bundled QAIRT runtime version string."""
    _ensure_bound()
    lib = load_library()
    return lib.geniex_qairt_version().decode()


def llama_cpp_version() -> str:
    """Return the bundled llama.cpp build commit hash."""
    _ensure_bound()
    lib = load_library()
    return lib.geniex_llama_cpp_version().decode()
