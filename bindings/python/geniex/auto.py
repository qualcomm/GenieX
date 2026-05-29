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

"""AutoModelForCausalLM and AutoModelForVision2Seq factory classes."""

from __future__ import annotations

import json
import logging
import os
from ctypes import byref, c_void_p

from . import _progress
from . import model_manager as _mm
from ._ffi._api import GeniexError, _check, ensure_init, get_plugin_list, load_library, resolve_device
from ._ffi._types import geniex_LlmCreateInput, geniex_ModelConfig, geniex_VlmCreateInput
from .model_manager import ProgressCallback
from .modeling import GeniexLLM, GeniexVLM

_logger = logging.getLogger('geniex')

PLUGIN_LLAMA_CPP = 'llama_cpp'
PLUGIN_QAIRT = 'qairt'

_KNOWN_ALIASES = {'cpu', 'gpu', 'npu', 'hybrid'}

# Stable owner for each alias, independent of plugin enumeration order.
# cpu/gpu/hybrid are llama_cpp-only concepts; npu is qairt's NPU-only default.
_ALIAS_OWNERS = {
    'cpu': PLUGIN_LLAMA_CPP,
    'gpu': PLUGIN_LLAMA_CPP,
    'hybrid': PLUGIN_LLAMA_CPP,
    'npu': PLUGIN_QAIRT,
}


def _apply_plugin_hint(device_map: str, plugin_id: str | None) -> str:
    """Bind a bare alias to the manifest's plugin so ``device_map='npu'`` on a
    cached llama_cpp model resolves to ``llama_cpp:npu`` instead of being
    captured by the static alias-owner table.

    No-op when the manifest doesn't carry a plugin (e.g. user passed a raw
    local path) — alias resolution then falls back to ``_ALIAS_OWNERS``.
    """
    if not plugin_id:
        return device_map
    if not device_map or device_map == 'auto':
        return plugin_id
    if device_map.lower() in _KNOWN_ALIASES:
        return f'{plugin_id}:{device_map.lower()}'
    return device_map


def resolve_device_map(
    device_map: str,
    model_name: str | None = None,
) -> tuple[str | None, str | None, int | None]:
    """Resolve a ``device_map`` string to ``(plugin_id, device_id, ngl_override)``.

    Accepted forms: ``"auto"`` / ``""`` (first plugin + SDK default),
    ``"cpu" | "gpu" | "npu" | "hybrid"`` (alias against the plugin that owns
    it — cpu/gpu/hybrid → llama_cpp, npu → qairt), ``"<plugin_id>"``, or
    ``"<plugin_id>:<device_id>"``.

    ``ngl_override`` is ``None`` unless the alias forces a specific
    ``n_gpu_layers`` (``cpu`` → 0, ``hybrid`` → 999).
    """
    if not device_map or device_map == 'auto':
        plugins = get_plugin_list()
        if not plugins:
            return None, None, None
        return _call_sdk(plugins[0], model_name, None)

    key = device_map.lower()

    if key in _KNOWN_ALIASES:
        plugins = get_plugin_list()
        owner = _ALIAS_OWNERS[key]
        plugin_id = owner if owner in plugins else (plugins[0] if plugins else PLUGIN_LLAMA_CPP)
        return _call_sdk(plugin_id, model_name, key)

    if ':' in device_map:
        plugin_id, device_id = device_map.split(':', 1)
        if device_id.lower() in _KNOWN_ALIASES:
            return _call_sdk(plugin_id, model_name, device_id.lower())
        if plugin_id == PLUGIN_QAIRT and device_id.upper() != 'NPU':
            _logger.warning(
                'qairt plugin only supports NPU inference; ignoring device_map=%r and running on NPU',
                device_map,
            )
            return plugin_id, 'NPU', None
        return plugin_id, device_id, None

    return _call_sdk(device_map, model_name, None)


def _call_sdk(
    plugin_id: str,
    model_name: str | None,
    alias: str | None,
) -> tuple[str, str | None, int | None]:
    # ngl_default=-1 is a sentinel so we can distinguish "SDK forced a value"
    # from "alias passed through" and surface the latter as None.
    device_id, ngl, warning = resolve_device(plugin_id, model_name, alias, -1)
    if warning:
        _logger.warning('%s', warning)
    ngl_override: int | None = None if ngl == -1 else ngl
    return plugin_id, device_id, ngl_override


def _resolve_local_anchor(path: str) -> str:
    # The C++ side derives the model dir via parent_path(), so we return a
    # file inside the directory rather than the directory itself.
    if os.path.isdir(path):
        anchor = os.path.join(path, 'tokenizer.json')
        if not os.path.isfile(anchor):
            entries = sorted(e for e in os.listdir(path) if os.path.isfile(os.path.join(path, e)))
            if not entries:
                raise FileNotFoundError(f'No files found in model directory: {path}')
            anchor = os.path.join(path, entries[0])
        return anchor
    return path


def _resolve_model_sources(
    model_name_or_path: str,
    quant: str | None,
    hf_token: str | None,
    progress: ProgressCallback | bool | None,
    model_name: str | None = None,
) -> tuple[str, str | None, str | None, _mm.ModelPaths | None]:
    if os.path.exists(model_name_or_path):
        # Optional: when the caller passes both a local path and `model_name=`,
        # register the bundle in the cache via the localfs hub so subsequent
        # `_mm.get_paths(model_name)` / `_mm.get_type(model_name)` lookups work
        # (e.g. VLM auto-detect). Not required for QAIRT to *load* — the plugin
        # reads metadata.json directly from the bundle.
        if model_name:
            # Reject mismatched model_name early — the QAIRT plugin would
            # otherwise fail at load with a generic "Invalid model format".
            meta = _read_bundle_metadata(model_name_or_path)
            bundle_id = (meta or {}).get('model_id')
            if bundle_id and bundle_id != model_name:
                raise ValueError(
                    f"model_name '{model_name}' does not match bundle "
                    f"'model_id={bundle_id}' in {model_name_or_path}"
                )
            local_dir = model_name_or_path if os.path.isdir(model_name_or_path) else os.path.dirname(model_name_or_path)
            _mm.pull(model_name, hub='localfs', local_path=os.path.abspath(local_dir))
            paths = _mm.get_paths(model_name)
            return paths.model_path, paths.mmproj_path, paths.tokenizer_path, paths
        return _resolve_local_anchor(model_name_or_path), None, None, None

    # Try the local cache first so AiHub/LocalFs models pulled previously
    # don't force the caller to respecify hub='aihub'.
    key = f'{model_name_or_path}:{quant}' if quant else model_name_or_path
    try:
        cached = _mm.get_paths(key)
        return cached.model_path, cached.mmproj_path, cached.tokenizer_path, cached
    except (GeniexError, FileNotFoundError, OSError):
        pass

    printer = _progress.resolve(progress)
    try:
        try:
            paths = _mm.ensure_cached(
                model_name_or_path,
                quant=quant,
                hub='auto',
                hf_token=hf_token,
                on_progress=printer,
            )
        except GeniexError as e:
            translated = _translate_quant_error(e, model_name_or_path, quant)
            if translated is not None:
                raise translated from e
            raise
    finally:
        _progress.finish(printer)
    return paths.model_path, paths.mmproj_path, paths.tokenizer_path, paths


# Model-manager FFI returns -100000 (Unknown error) when manifest inference
# rejects a requested quant — the Rust side already builds a richer
# "available: [...]" message, but it's not surfaced through the C ABI yet.
# Translate that catch-all into a focused ValueError when the caller
# explicitly asked for a specific quant, so they can spot the typo without
# inspecting model-manager logs. The detailed list is tracked under #737.
GENIEX_ERROR_COMMON_UNKNOWN = -100000


def _translate_quant_error(err: GeniexError, model_name_or_path: str, quant: str | None) -> ValueError | None:
    if quant is None or err.code != GENIEX_ERROR_COMMON_UNKNOWN:
        return None
    return ValueError(f'Could not resolve quant {quant!r} for {model_name_or_path!r}.')


def _reject_gguf_on_qairt(model_path: str, plugin_id: str | None, device_map: str) -> None:
    # QAIRT only consumes its own .bin shards + metadata.json bundles. A .gguf
    # routed to QAIRT would otherwise fail deep in the plugin with a generic
    # "Invalid model format" error.
    if plugin_id == PLUGIN_QAIRT and model_path.lower().endswith('.gguf'):
        raise ValueError(
            f".gguf models are not supported by device_map='{device_map}' "
            f"(QAIRT/NPU). Use device_map='auto' or 'hybrid' instead."
        )


# AutoModelFor*.from_pretrained factory defaults; not user overrides, so coerce silently.
_QAIRT_SILENT_NGL = 999
_QAIRT_SILENT_NCTX = 0


def _build_model_config(plugin_id: str | None, n_ctx: int, n_gpu_layers: int, **kwargs) -> geniex_ModelConfig:
    if plugin_id == PLUGIN_QAIRT:
        if n_gpu_layers not in (0, _QAIRT_SILENT_NGL):
            _logger.warning('qairt plugin does not consume n_gpu_layers=%d; forcing 0', n_gpu_layers)
        n_gpu_layers = 0
        if n_ctx != _QAIRT_SILENT_NCTX:
            _logger.warning('qairt plugin does not consume n_ctx=%d; forcing 0', n_ctx)
            n_ctx = 0
    cfg = geniex_ModelConfig(n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
    _int_fields = {'n_threads', 'n_threads_batch', 'n_batch', 'n_ubatch', 'n_seq_max', 'max_tokens'}
    _bool_fields = {'enable_thinking', 'verbose'}
    _str_fields = {'chat_template_path', 'chat_template_content', 'system_prompt'}
    for k, v in kwargs.items():
        if k in _int_fields:
            setattr(cfg, k, int(v))
        elif k in _bool_fields:
            setattr(cfg, k, bool(v))
        elif k in _str_fields and v is not None:
            setattr(cfg, k, v.encode())
    return cfg


def _read_bundle_metadata(model_path: str) -> dict | None:
    # Read metadata.json from a QAIRT bundle directory (or the dir holding a
    # passed-in file). Returns None when the file is missing/unreadable so
    # callers fall through to other signals.
    bundle_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
    meta_path = os.path.join(bundle_dir, 'metadata.json')
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _qairt_bundle_is_vlm(model_path: str) -> bool | None:
    meta = _read_bundle_metadata(model_path)
    if meta is None:
        return None
    genie = meta.get('genie') or {}
    return bool(genie.get('supports_vision'))


def _detect_supports_thinking(model_path: str | None, tokenizer_path: str | None) -> bool | None:
    # Inspect the model's own Jinja chat template (tokenizer_config.json's
    # `chat_template` field). A template that branches on `enable_thinking`
    # or emits literal `<think>` tags is the authoritative signal that the
    # model has a thinking mode. Returns None when the file isn't on disk
    # (e.g. GGUF bundles embed the template inside the file) so callers can
    # fall back to a default.
    candidates: list[str] = []
    if tokenizer_path:
        # tokenizer_path points at tokenizer.json; its sibling is the config.
        candidates.append(os.path.join(os.path.dirname(tokenizer_path), 'tokenizer_config.json'))
    if model_path:
        bundle_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        candidates.append(os.path.join(bundle_dir, 'tokenizer_config.json'))
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding='utf-8') as f:
                tc = json.load(f)
        except (OSError, ValueError):
            continue
        template = tc.get('chat_template')
        if not isinstance(template, str):
            return None
        return ('enable_thinking' in template) or ('<think>' in template)
    return None


def _is_vlm(mmproj_path: str | None, cache_key: str, model_path: str | None = None) -> bool:
    # mmproj_path is the llama_cpp signal (multimodal projector file).
    # QAIRT VLMs bundle the vision encoder into the QNN binary and have no
    # mmproj. For raw local QAIRT bundles we read metadata.json directly so
    # `from_pretrained("/path/to/qairt_vlm_bundle")` routes to VLM without
    # requiring a prior `geniex pull`. Cached models fall through to the
    # manifest's ModelType.
    if mmproj_path is not None:
        return True
    if model_path:
        from_meta = _qairt_bundle_is_vlm(model_path)
        if from_meta is not None:
            return from_meta
    try:
        return _mm.get_type(cache_key) == 'vlm'
    except Exception:  # noqa: BLE001 — uncached / unknown → treat as LLM
        return False


def _create_vlm_handle(
    resolved_name: str,
    model_path: str,
    mmproj_path: str | None,
    tokenizer_path: str | None,
    plugin_id: str | None,
    device_id: str | None,
    config: geniex_ModelConfig,
    license_id: str | None,
    license_key: str | None,
    meta: dict | None = None,
) -> GeniexVLM:
    inp = geniex_VlmCreateInput(
        model_name=resolved_name.encode(),
        model_path=model_path.encode(),
        config=config,
    )
    if mmproj_path:
        inp.mmproj_path = mmproj_path.encode()
    if tokenizer_path:
        inp.tokenizer_path = tokenizer_path.encode()
    if plugin_id:
        inp.plugin_id = plugin_id.encode()
    if device_id:
        inp.device_id = device_id.encode()
    if license_id:
        inp.license_id = license_id.encode()
    if license_key:
        inp.license_key = license_key.encode()

    handle = c_void_p()
    lib = load_library()
    _check(lib.geniex_vlm_create(byref(inp), byref(handle)))
    return GeniexVLM(handle, meta=meta)


class AutoModelForCausalLM:
    """Factory for causal language models (text-only and multimodal)."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        model_name: str | None = None,
        quant: str | None = None,
        device_map: str = 'auto',
        n_ctx: int = 0,
        n_gpu_layers: int = 999,
        mmproj_path: str | None = None,
        tokenizer_path: str | None = None,
        license_id: str | None = None,
        license_key: str | None = None,
        hf_token: str | None = None,
        progress: ProgressCallback | bool | None = None,
        **kwargs,
    ) -> GeniexLLM | GeniexVLM:
        """Load a causal LM or VLM by HF repo id, alias, or local path.

        ``model_name`` is **optional**. The QAIRT plugin no longer needs it —
        it dispatches by reading ``metadata.json`` from the bundle. Pass it
        only if you want to register a local-path bundle in the geniex cache
        (so it shows up in ``geniex list`` and survives across runs), or to
        provide a hint to ``geniex_resolve_device`` for the gpt-oss
        llama_cpp device-default override.

        When the model is detected as multimodal (e.g. phi4_multimodal,
        qwen3.5-vl, gemma4), a :class:`GeniexVLM` is returned instead.
        """
        ensure_init()
        model_path, _mmproj, _tok, paths = _resolve_model_sources(
            model_name_or_path,
            quant,
            hf_token,
            progress,
            model_name,
        )
        resolved_name = model_name or (paths.model_name if paths and paths.model_name else model_name_or_path)
        effective_device_map = _apply_plugin_hint(device_map, paths.plugin_id if paths else None)
        plugin_id, device_id, ngl_override = resolve_device_map(effective_device_map, resolved_name)
        _reject_gguf_on_qairt(model_path, plugin_id, device_map)
        if ngl_override is not None:
            n_gpu_layers = ngl_override
        config = _build_model_config(plugin_id, n_ctx, n_gpu_layers, **kwargs)
        resolved_tok_path = tokenizer_path or _tok
        meta = {
            'backend': plugin_id,
            'device': device_id,
            'quant': quant,
            'model_path': model_path,
            'supports_thinking': _detect_supports_thinking(model_path, resolved_tok_path),
        }

        resolved_mmproj = mmproj_path or _mmproj
        if _is_vlm(resolved_mmproj, model_name or model_name_or_path, model_path):
            return _create_vlm_handle(
                resolved_name,
                model_path,
                resolved_mmproj,
                tokenizer_path or _tok,
                plugin_id,
                device_id,
                config,
                license_id,
                license_key,
                meta=meta,
            )

        inp = geniex_LlmCreateInput(
            model_name=resolved_name.encode(),
            model_path=model_path.encode(),
            config=config,
        )
        if resolved_tok_path:
            inp.tokenizer_path = resolved_tok_path.encode()
        if plugin_id:
            inp.plugin_id = plugin_id.encode()
        if device_id:
            inp.device_id = device_id.encode()
        if license_id:
            inp.license_id = license_id.encode()
        if license_key:
            inp.license_key = license_key.encode()

        handle = c_void_p()
        lib = load_library()
        _check(lib.geniex_llm_create(byref(inp), byref(handle)))
        return GeniexLLM(handle, meta=meta)


class AutoModelForVision2Seq:
    """Factory for vision-language / multimodal models."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        model_name: str | None = None,
        quant: str | None = None,
        device_map: str = 'auto',
        n_ctx: int = 0,
        n_gpu_layers: int = 999,
        mmproj_path: str | None = None,
        tokenizer_path: str | None = None,
        license_id: str | None = None,
        license_key: str | None = None,
        hf_token: str | None = None,
        progress: ProgressCallback | bool | None = None,
        **kwargs,
    ) -> GeniexVLM:
        """Load a VLM by HF repo id, alias, or local path.

        See :class:`AutoModelForCausalLM.from_pretrained` for shared parameters.
        ``model_name`` is optional — the QAIRT plugin reads metadata.json from
        the bundle directly. ``mmproj_path`` is an optional override for the
        multimodal projector file.
        """
        ensure_init()
        model_path, _mmproj, _tok, paths = _resolve_model_sources(
            model_name_or_path,
            quant,
            hf_token,
            progress,
            model_name,
        )
        resolved_name = model_name or (paths.model_name if paths and paths.model_name else model_name_or_path)
        effective_device_map = _apply_plugin_hint(device_map, paths.plugin_id if paths else None)
        plugin_id, device_id, ngl_override = resolve_device_map(effective_device_map, resolved_name)
        _reject_gguf_on_qairt(model_path, plugin_id, device_map)
        if ngl_override is not None:
            n_gpu_layers = ngl_override
        config = _build_model_config(plugin_id, n_ctx, n_gpu_layers, **kwargs)
        resolved_tok_path = tokenizer_path or _tok
        meta = {
            'backend': plugin_id,
            'device': device_id,
            'quant': quant,
            'model_path': model_path,
            'supports_thinking': _detect_supports_thinking(model_path, resolved_tok_path),
        }

        return _create_vlm_handle(
            resolved_name,
            model_path,
            mmproj_path or _mmproj,
            resolved_tok_path,
            plugin_id,
            device_id,
            config,
            license_id,
            license_key,
            meta=meta,
        )
