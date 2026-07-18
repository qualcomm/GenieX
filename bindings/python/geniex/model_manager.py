# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""High-level Python API for the geniex model manager."""

from __future__ import annotations

from ctypes import byref, c_char_p, c_int32, sizeof
from dataclasses import dataclass
from typing import Callable

from ._ffi._api import GenieXError, _check, _ensure_bound
from ._ffi._lib import load_library
from ._ffi._types import (
    GENIEX_HUB_AIHUB,
    GENIEX_HUB_AUTO,
    GENIEX_HUB_DOCKER,
    GENIEX_HUB_HUGGINGFACE,
    GENIEX_HUB_LOCALFS,
    GENIEX_MODEL_TYPE_LLM,
    GENIEX_MODEL_TYPE_VLM,
    geniex_ChipsetList,
    geniex_download_progress_cb,
    geniex_ModelListDetailedOutput,
    geniex_ModelPaths,
    geniex_ModelPullInput,
    geniex_ModelQueryOutput,
)

__all__ = [
    'ModelPaths',
    'ModelDetail',
    'PrecisionCandidate',
    'ModelQuery',
    'FileProgress',
    'ProgressCallback',
    'init',
    'deinit',
    'pull',
    'list_models',
    'list_detailed',
    'last_error_message',
    'query',
    'remove',
    'clean',
    'get_paths',
    'get_type',
    'set_type',
    'resolve_alias',
    'ensure_cached',
    'ChipsetInfo',
    'list_chipsets',
    'detect_chipset',
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
    runtime: str
    model_type: str  # "llm" or "vlm"
    mmproj_path: str | None = None
    tokenizer_path: str | None = None
    # QAIRT version the model's assets were pulled for (AI Hub metadata);
    # empty for non-AI-Hub / llama.cpp models.
    qairt_version: str = ''


@dataclass(frozen=True)
class ModelDetail:
    """Full metadata for one cached model."""

    name: str
    model_name: str
    runtime: str
    model_type: str  # "llm" or "vlm"
    total_size: int
    precisions: list[str]


@dataclass(frozen=True)
class ChipsetInfo:
    """One chipset Qualcomm AI Hub publishes assets for."""

    name: str
    aliases: list[str]


@dataclass(frozen=True)
class PrecisionCandidate:
    """One precision advertised by a hub for a model."""

    precision: str
    size: int
    is_default: bool


@dataclass(frozen=True)
class ModelQuery:
    """Result of a plan-only :func:`query`."""

    model_name: str
    runtime: str
    model_type: str  # "llm" or "vlm"
    candidates: list[PrecisionCandidate]


ProgressCallback = Callable[[list[FileProgress]], bool]

GENIEX_MODEL_TYPE_AUTO = -1


def _type_str(value: int) -> str:
    if value == GENIEX_MODEL_TYPE_LLM:
        return 'llm'
    if value == GENIEX_MODEL_TYPE_VLM:
        return 'vlm'
    raise ValueError(f'Unknown model type: {value}')


_HUB_MAP = {
    'auto': GENIEX_HUB_AUTO,
    'hf': GENIEX_HUB_HUGGINGFACE,
    'huggingface': GENIEX_HUB_HUGGINGFACE,
    'aihub': GENIEX_HUB_AIHUB,
    'docker': GENIEX_HUB_DOCKER,
    'dockerhub': GENIEX_HUB_DOCKER,
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
    """Initialise the model manager. Idempotent.

    ``data_dir`` precedence: argument → ``GENIEX_DATADIR`` env → ``~/.cache/geniex``.
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
    """Release model-manager resources."""
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
    # Split a trailing :quant, try resolve_alias() for bare names, leave org/repo alone.
    # Unknown bare names pass through — the SDK canonicalises them to aihub/<name>.
    name_part = model_name
    if ':' in model_name:
        name_part, parsed_quant = model_name.rsplit(':', 1)
        if quant is None and parsed_quant:
            quant = parsed_quant

    if '/' in name_part:
        return name_part, quant

    try:
        resolved = resolve_alias(name_part)
    except GenieXError:
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
    precision: str | None = None,
    hub: str | int = 'auto',
    local_path: str | None = None,
    hf_token: str | None = None,
    chipset: str | None = None,
    display_name: str | None = None,
    model_type: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Download a model into the local cache (blocking, resumable).

    Args:
        model_name: ``org/repo``, ``org/repo:precision``, or a short alias.
        precision: Optional precision hint (e.g. ``"Q4_K_M"``).
        hub: ``"auto" | "hf" | "aihub" | "docker" | "localfs"`` or a raw enum int.
        local_path: Required when ``hub == "localfs"``.
        hf_token: HuggingFace bearer token; falls back to ``GENIEX_HFTOKEN``.
        chipset: AI Hub target chipset; auto-detected on Windows-on-Snapdragon.
        display_name: AI Hub ``display_name``. Optional when the model name
            starts with ``qualcomm/``, ``qai-hub-models/``, or ``aihub/`` — the
            SDK derives it from the repo. Required only when the stored name
            cannot be mapped (rare).
        model_type: ``"llm" | "vlm" | None``. ``None`` auto-detects.
        on_progress: Callback ``(files) -> bool``; return ``False`` to cancel.
    """
    _ensure_init()
    lib = load_library()

    model_name, precision = _maybe_resolve_alias(model_name, precision)
    hub_val = _resolve_hub(hub)

    if model_type is None:
        model_type_val = GENIEX_MODEL_TYPE_AUTO
    elif model_type.lower() == 'llm':
        model_type_val = GENIEX_MODEL_TYPE_LLM
    elif model_type.lower() == 'vlm':
        model_type_val = GENIEX_MODEL_TYPE_VLM
    else:
        raise ValueError(f"Unknown model type: {model_type!r} (expected 'llm', 'vlm', or None)")

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

    # struct_size is the ABI version gate; the SDK rejects stale layouts.
    inp = geniex_ModelPullInput(
        struct_size=sizeof(geniex_ModelPullInput),
        model_name=model_name.encode(),
        quant=precision.encode() if precision else None,
        hub=hub_val,
        local_path=local_path.encode() if local_path else None,
        hf_token=hf_token.encode() if hf_token else None,
        chipset=chipset.encode() if chipset else None,
        display_name=display_name.encode() if display_name else None,
        on_progress=cb,
        user_data=None,
        model_type=model_type_val,
    )
    _check(lib.geniex_model_pull(byref(inp)))


def list_models() -> list[str]:
    """Return cached model names (``org/repo``)."""
    return [d.name for d in list_detailed()]


def last_error_message() -> str | None:
    """Return the detailed message for the last failing model-manager call.

    Thread-local and library-owned; valid only until the next failing call on
    this thread. Returns ``None`` if no error has been recorded.
    """
    _ensure_bound()
    lib = load_library()
    msg = lib.geniex_model_last_error_message()
    return msg.decode() if msg is not None else None


def list_detailed() -> list[ModelDetail]:
    """Return cached models with full metadata (size, plugin, type, precisions)."""
    _ensure_init()
    lib = load_library()
    out = geniex_ModelListDetailedOutput()
    _check(lib.geniex_model_list_detailed(byref(out)))
    try:
        models = []
        for i in range(out.count):
            d = out.models[i]
            models.append(
                ModelDetail(
                    name=d.name.decode() if d.name else '',
                    model_name=d.model_name.decode() if d.model_name else '',
                    runtime=d.plugin_id.decode() if d.plugin_id else '',
                    model_type=_type_str(d.model_type),
                    total_size=d.total_size,
                    precisions=[d.precisions[j].decode() for j in range(d.precision_count)],
                )
            )
        return models
    finally:
        lib.geniex_model_list_detailed_free(byref(out))


def query(
    model_name: str,
    *,
    hub: str | int = 'auto',
    local_path: str | None = None,
    hf_token: str | None = None,
    chipset: str | None = None,
    display_name: str | None = None,
) -> ModelQuery:
    """Resolve a model's remote candidate precisions without downloading."""
    _ensure_init()
    lib = load_library()

    name, _ = _maybe_resolve_alias(model_name, None)
    inp = geniex_ModelPullInput(
        struct_size=sizeof(geniex_ModelPullInput),
        model_name=name.encode(),
        quant=None,
        hub=_resolve_hub(hub),
        local_path=local_path.encode() if local_path else None,
        hf_token=hf_token.encode() if hf_token else None,
        chipset=chipset.encode() if chipset else None,
        display_name=display_name.encode() if display_name else None,
        on_progress=geniex_download_progress_cb(0),
        user_data=None,
        model_type=GENIEX_MODEL_TYPE_AUTO,
    )
    out = geniex_ModelQueryOutput()
    _check(lib.geniex_model_query(byref(inp), byref(out)))
    try:
        candidates = [
            PrecisionCandidate(
                precision=out.candidates[i].quant.decode() if out.candidates[i].quant else '',
                size=out.candidates[i].size,
                is_default=bool(out.candidates[i].is_default),
            )
            for i in range(out.candidate_count)
        ]
        return ModelQuery(
            model_name=out.model_name.decode() if out.model_name else '',
            runtime=out.plugin_id.decode() if out.plugin_id else '',
            model_type=_type_str(out.model_type),
            candidates=candidates,
        )
    finally:
        lib.geniex_model_query_free(byref(out))


def remove(model_name: str) -> None:
    """Delete ``model_name`` from the local cache."""
    _ensure_init()
    lib = load_library()
    _check(lib.geniex_model_remove(model_name.encode()))


def clean() -> int:
    """Remove all cached models. Returns the number removed."""
    _ensure_init()
    lib = load_library()
    n = c_int32(0)
    _check(lib.geniex_model_clean(byref(n)))
    return n.value


def get_paths(model_name: str) -> ModelPaths:
    """Resolve ``org/repo[:precision]`` (or alias) to absolute on-disk paths."""
    _ensure_init()
    lib = load_library()
    base, precision = _maybe_resolve_alias(model_name, None)
    key = f'{base}:{precision}' if precision else base
    out = geniex_ModelPaths()
    _check(lib.geniex_model_get_paths(key.encode(), byref(out)))
    try:
        return ModelPaths(
            model_path=out.model_path.decode() if out.model_path else '',
            model_dir=out.model_dir.decode() if out.model_dir else '',
            model_name=out.model_name.decode() if out.model_name else '',
            runtime=out.plugin_id.decode() if out.plugin_id else '',
            model_type=_type_str(out.model_type),
            mmproj_path=out.mmproj_path.decode() if out.mmproj_path else None,
            tokenizer_path=out.tokenizer_path.decode() if out.tokenizer_path else None,
            qairt_version=out.qairt_version.decode() if out.qairt_version else '',
        )
    finally:
        lib.geniex_model_paths_free(byref(out))


def get_type(model_name: str) -> str:
    """Return ``"llm"`` or ``"vlm"`` for a cached model."""
    _ensure_init()
    lib = load_library()
    t = c_int32(0)
    _check(lib.geniex_model_get_type(model_name.encode(), byref(t)))
    return _type_str(t.value)


def set_type(model_name: str, model_type: str) -> None:
    """Override the stored model type (``"llm"`` or ``"vlm"``) of a cached model."""
    _ensure_init()
    lib = load_library()
    key = model_type.lower()
    if key == 'llm':
        value = GENIEX_MODEL_TYPE_LLM
    elif key == 'vlm':
        value = GENIEX_MODEL_TYPE_VLM
    else:
        raise ValueError(f"Unknown model type: {model_type!r} (expected 'llm' or 'vlm')")
    _check(lib.geniex_model_set_type(model_name.encode(), value))


def resolve_alias(alias: str) -> str:
    """Expand a short alias (e.g. ``"qwen3"``) to ``"org/repo"``."""
    _ensure_init()
    lib = load_library()
    out = c_char_p()
    _check(lib.geniex_model_resolve_alias(alias.encode(), byref(out)))
    try:
        return out.value.decode() if out.value else ''
    finally:
        lib.geniex_free(out)


def resolve_effective_hub(model_name: str, hub: str | int = 'auto') -> int:
    """Return the hub enum a pull/query will actually use for ``model_name``.

    ``"auto"`` resolves to Docker Hub when the name carries a Docker Hub prefix
    (``docker.io/…``); every other hub is returned unchanged. The prefix table
    lives in the SDK, so callers must not re-derive it. No network I/O.
    """
    _ensure_init()
    lib = load_library()
    out = c_int32()
    _check(lib.geniex_model_resolve_hub(model_name.encode(), _resolve_hub(hub), byref(out)))
    return out.value


def list_chipsets() -> list[ChipsetInfo]:
    """List every chipset Qualcomm AI Hub supports, with aliases.

    Sourced from ``platform.json`` (cached 24h); the first call may hit the network.
    """
    _ensure_init()
    lib = load_library()
    out = geniex_ChipsetList()
    _check(lib.geniex_model_list_chipsets(byref(out)))
    try:
        chipsets = []
        for i in range(out.count):
            c = out.chipsets[i]
            chipsets.append(
                ChipsetInfo(
                    name=c.name.decode() if c.name else '',
                    aliases=[c.aliases[j].decode() for j in range(c.alias_count)],
                )
            )
        return chipsets
    finally:
        lib.geniex_model_list_chipsets_free(byref(out))


def detect_chipset() -> str | None:
    """Detect the current host's chipset via a local probe (no network).

    Returns ``None`` when the platform cannot be probed.
    """
    _ensure_init()
    lib = load_library()
    out = c_char_p()
    _check(lib.geniex_model_detect_chipset(byref(out)))
    if not out.value:
        return None
    try:
        return out.value.decode()
    finally:
        lib.geniex_free(out)


def ensure_cached(
    model_name_or_alias: str,
    *,
    precision: str | None = None,
    hub: str | int = 'auto',
    local_path: str | None = None,
    hf_token: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> ModelPaths:
    """Resolve alias, :func:`pull` if missing, and return :class:`ModelPaths`."""
    _ensure_init()

    name_part = model_name_or_alias
    if ':' in model_name_or_alias:
        name_part, parsed_precision = model_name_or_alias.rsplit(':', 1)
        if precision is None and parsed_precision:
            precision = parsed_precision

    try:
        full_name = resolve_alias(name_part)
    except GenieXError:
        full_name = name_part

    # No precision + remote source: resolve the hub default before pulling so
    # only one variant is downloaded instead of all of them.
    if precision is None and local_path is None:
        try:
            result = query(full_name, hub=hub, hf_token=hf_token)
            default = next((c.precision for c in result.candidates if c.is_default), None)
            if default:
                precision = default
        except GenieXError:
            pass  # offline or unsupported hub; let pull decide

    pull(
        full_name,
        precision=precision,
        hub=hub,
        local_path=local_path,
        hf_token=hf_token,
        on_progress=on_progress,
    )
    key = f'{full_name}:{precision}' if precision else full_name
    return get_paths(key)
