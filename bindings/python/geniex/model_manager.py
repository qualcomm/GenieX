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

"""High-level Python API for the geniex model manager.

Wraps the `geniex_model_*` C FFI (implemented in Rust under
``sdk/model-manager``) so Python callers can download, list, and resolve
models without re-implementing hub logic per binding.
"""

from __future__ import annotations

from ctypes import byref, c_char_p, c_int32, sizeof
from dataclasses import dataclass
from typing import Callable

from ._ffi._api import GeniexError, _check, _ensure_bound
from ._ffi._lib import load_library
from ._ffi._types import (
    GENIEX_HUB_AIHUB,
    GENIEX_HUB_AUTO,
    GENIEX_HUB_HUGGINGFACE,
    GENIEX_HUB_LOCALFS,
    GENIEX_MODEL_TYPE_LLM,
    GENIEX_MODEL_TYPE_VLM,
    geniex_download_progress_cb,
    geniex_ModelListOutput,
    geniex_ModelPaths,
    geniex_ModelPullInput,
)

__all__ = [
    'ModelPaths',
    'FileProgress',
    'ProgressCallback',
    'init',
    'deinit',
    'pull',
    'list_models',
    'remove',
    'clean',
    'get_paths',
    'get_type',
    'resolve_alias',
]


@dataclass(frozen=True)
class FileProgress:
    """Per-file download progress."""

    file_name: str
    downloaded_bytes: int
    total_bytes: int  # -1 if unknown


@dataclass(frozen=True)
class ModelPaths:
    """Resolved absolute paths for a cached model."""

    model_path: str
    model_dir: str
    model_name: str
    plugin_id: str
    mmproj_path: str | None = None
    tokenizer_path: str | None = None
    device_id: str | None = None


ProgressCallback = Callable[[list[FileProgress]], bool]


_HUB_MAP = {
    'auto': GENIEX_HUB_AUTO,
    'hf': GENIEX_HUB_HUGGINGFACE,
    'huggingface': GENIEX_HUB_HUGGINGFACE,
    'aihub': GENIEX_HUB_AIHUB,
    'localfs': GENIEX_HUB_LOCALFS,
    'local': GENIEX_HUB_LOCALFS,
}


def _resolve_hub(hub: str | int) -> int:
    if isinstance(hub, int):
        return hub
    key = hub.lower()
    if key not in _HUB_MAP:
        raise ValueError(f'Unknown hub: {hub!r} (expected one of {list(_HUB_MAP)})')
    return _HUB_MAP[key]


_initialized = False


def init(data_dir: str | None = None) -> None:
    """Initialize the model manager.

    ``data_dir`` precedence: argument → ``GENIEX_DATADIR`` env → ``~/.cache/geniex``.
    Idempotent — calling twice is a no-op.
    """
    global _initialized
    if _initialized:
        return
    _ensure_bound()
    lib = load_library()
    path = data_dir.encode() if data_dir else None
    _check(lib.geniex_model_init(path))
    _initialized = True


def deinit() -> None:
    global _initialized
    if not _initialized:
        return
    lib = load_library()
    lib.geniex_model_deinit()
    _initialized = False


def _ensure_init() -> None:
    if not _initialized:
        init()


def _maybe_resolve_alias(model_name: str, quant: str | None) -> tuple[str, str | None]:
    """Expand a short alias to ``org/repo[:quant]`` if the input has no ``/``.

    Splits a trailing ``:quant`` off the input, then tries :func:`resolve_alias`
    for bare names (no ``/``). If the alias is unknown, the bare name is passed
    through unchanged — the SDK canonicalises bare names to ``aihub/<name>`` so
    they route to AI Hub without a registered alias. Canonical ``org/repo``
    inputs pass through unchanged.
    """
    name_part = model_name
    if ':' in model_name:
        name_part, parsed_quant = model_name.rsplit(':', 1)
        if quant is None and parsed_quant:
            quant = parsed_quant

    if '/' in name_part:
        return name_part, quant

    try:
        resolved = resolve_alias(name_part)
    except GeniexError:
        # Unknown alias: let the SDK treat it as a bare AI Hub model id.
        return name_part, quant

    if ':' in resolved:
        resolved_name, resolved_quant = resolved.rsplit(':', 1)
        if quant is None and resolved_quant:
            quant = resolved_quant
        return resolved_name, quant
    return resolved, quant


def pull(
    model_name: str,
    *,
    quant: str | None = None,
    hub: str | int = 'auto',
    local_path: str | None = None,
    hf_token: str | None = None,
    chipset: str | None = None,
    display_name: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Download a model (blocking). Resumes partial downloads.

    Args:
        model_name: ``org/repo``, ``org/repo:quant``, or a short alias
            (see :func:`resolve_alias`). Aliases are expanded locally
            before the call into the SDK.
        quant: Optional quantization hint (e.g. ``"Q4_K_M"``).
        hub: ``"auto"`` | ``"hf"`` | ``"aihub"`` | ``"localfs"``, or a raw integer enum.
        local_path: Required when ``hub == "localfs"`` — source directory.
        hf_token: HuggingFace bearer token. Falls back to ``GENIEX_HFTOKEN`` env.
        chipset: AI Hub target chipset (e.g. ``"qualcomm-snapdragon-x-elite"``).
            Only used when ``hub == "aihub"``. ``None`` asks the SDK to auto-detect,
            currently supported on Windows-on-Snapdragon hosts only.
        display_name: AI Hub ``display_name`` of the model. Required when
            ``hub == "aihub"``; ignored otherwise.
        on_progress: Callback ``(files) -> bool``; return False to cancel.
    """
    _ensure_init()
    lib = load_library()

    model_name, quant = _maybe_resolve_alias(model_name, quant)
    hub_val = _resolve_hub(hub)

    def _trampoline(files_ptr, count, _user_data):
        try:
            items = [
                FileProgress(
                    file_name=files_ptr[i].file_name.decode() if files_ptr[i].file_name else '',
                    downloaded_bytes=files_ptr[i].downloaded_bytes,
                    total_bytes=files_ptr[i].total_bytes,
                )
                for i in range(count)
            ]
            return bool(on_progress(items)) if on_progress else True
        except Exception:
            return False

    cb = geniex_download_progress_cb(_trampoline) if on_progress else geniex_download_progress_cb(0)

    # struct_size is the ABI version gate. Wrapping sizeof() here means
    # ctypes mirror drift between Python and the installed geniex.dll is
    # a rejected FFI call rather than a silent garbage read.
    inp = geniex_ModelPullInput(
        struct_size=sizeof(geniex_ModelPullInput),
        model_name=model_name.encode(),
        quant=quant.encode() if quant else None,
        hub=hub_val,
        local_path=local_path.encode() if local_path else None,
        hf_token=hf_token.encode() if hf_token else None,
        chipset=chipset.encode() if chipset else None,
        display_name=display_name.encode() if display_name else None,
        on_progress=cb,
        user_data=None,
    )
    _check(lib.geniex_model_pull(byref(inp)))


def list_models() -> list[str]:
    """List cached model names (``org/repo``)."""
    _ensure_init()
    lib = load_library()
    out = geniex_ModelListOutput()
    _check(lib.geniex_model_list(byref(out)))
    try:
        return [out.names[i].decode() for i in range(out.count)]
    finally:
        lib.geniex_model_list_free(byref(out))


def remove(model_name: str) -> None:
    """Delete a cached model from disk."""
    _ensure_init()
    lib = load_library()
    _check(lib.geniex_model_remove(model_name.encode()))


def clean() -> int:
    """Remove all cached models; returns the count removed."""
    _ensure_init()
    lib = load_library()
    n = c_int32(0)
    _check(lib.geniex_model_clean(byref(n)))
    return n.value


def get_paths(model_name: str) -> ModelPaths:
    """Resolve ``org/repo`` (optionally ``:quant``) or a short alias to absolute paths.

    Aliases (e.g. ``"qwen3"``) are expanded locally via :func:`resolve_alias`
    before the lookup; any ``:quant`` suffix on the input or on the resolved
    alias is preserved so cached quant-specific entries hit.
    """
    _ensure_init()
    lib = load_library()
    base, quant = _maybe_resolve_alias(model_name, None)
    key = f'{base}:{quant}' if quant else base
    out = geniex_ModelPaths()
    _check(lib.geniex_model_get_paths(key.encode(), byref(out)))
    try:
        return ModelPaths(
            model_path=out.model_path.decode() if out.model_path else '',
            model_dir=out.model_dir.decode() if out.model_dir else '',
            model_name=out.model_name.decode() if out.model_name else '',
            plugin_id=out.plugin_id.decode() if out.plugin_id else '',
            mmproj_path=out.mmproj_path.decode() if out.mmproj_path else None,
            tokenizer_path=out.tokenizer_path.decode() if out.tokenizer_path else None,
            device_id=out.device_id.decode() if out.device_id else None,
        )
    finally:
        lib.geniex_model_paths_free(byref(out))


def get_type(model_name: str) -> str:
    """Return ``"llm"`` or ``"vlm"``."""
    _ensure_init()
    lib = load_library()
    t = c_int32(0)
    _check(lib.geniex_model_get_type(model_name.encode(), byref(t)))
    if t.value == GENIEX_MODEL_TYPE_LLM:
        return 'llm'
    if t.value == GENIEX_MODEL_TYPE_VLM:
        return 'vlm'
    raise GeniexError(-1, f'Unknown model type: {t.value}')


def resolve_alias(alias: str) -> str:
    """Resolve a short alias (e.g. ``"qwen3"``) to a canonical ``org/repo``."""
    _ensure_init()
    lib = load_library()
    out = c_char_p()
    _check(lib.geniex_model_resolve_alias(alias.encode(), byref(out)))
    try:
        return out.value.decode() if out.value else ''
    finally:
        lib.geniex_free(out)


def ensure_cached(
    model_name_or_alias: str,
    *,
    quant: str | None = None,
    hub: str | int = 'auto',
    local_path: str | None = None,
    hf_token: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> ModelPaths:
    """Resolve alias, pull if missing, and return resolved paths.

    Accepts ``org/repo`` or ``org/repo:quant`` (colon syntax); an explicit
    ``quant`` keyword overrides anything parsed from the name.
    """
    _ensure_init()

    name_part = model_name_or_alias
    if ':' in model_name_or_alias:
        name_part, parsed_quant = model_name_or_alias.rsplit(':', 1)
        if quant is None and parsed_quant:
            quant = parsed_quant

    # Try alias resolution; if it fails, assume the input is already org/repo.
    try:
        full_name = resolve_alias(name_part)
    except GeniexError:
        full_name = name_part

    pull(
        full_name,
        quant=quant,
        hub=hub,
        local_path=local_path,
        hf_token=hf_token,
        on_progress=on_progress,
    )
    key = f'{full_name}:{quant}' if quant else full_name
    return get_paths(key)
