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

import logging
import os
import sys
from ctypes import byref, c_void_p

from . import model_manager as _mm
from ._ffi._api import _check, ensure_init, get_plugin_list, load_library, resolve_device
from ._ffi._types import geniex_LlmCreateInput, geniex_ModelConfig, geniex_VlmCreateInput
from .modeling import GeniexLLM, GeniexVLM

_logger = logging.getLogger('geniex')

PLUGIN_LLAMA_CPP = 'llama_cpp'
PLUGIN_QAIRT = 'qairt'

_KNOWN_ALIASES = {'cpu', 'gpu', 'npu', 'hybrid'}


def _resolve_device(
    device_map: str,
    model_name: str | None = None,
) -> tuple[str | None, str | None, int | None]:
    """Return ``(plugin_id, device_id, ngl_override)`` from ``device_map``.

    This is the pybind-side entry point. Plugin selection and the
    ``<plugin>:<device>`` parsing stay in Python; the actual alias →
    concrete (device_id, ngl) mapping is delegated to the SDK's
    :func:`geniex._ffi._api.resolve_device`.

    Accepted forms:
      - ``"auto"`` / empty — pick the first plugin, then ask the SDK
        for that plugin's default alias.
      - ``"<plugin_id>"`` — use this plugin with its default alias.
      - ``"<plugin_id>:<device_id>"`` — fully specified. If
        ``<device_id>`` is a friendly alias, the SDK translates it;
        otherwise it passes through unchanged.
      - ``"cpu"`` / ``"gpu"`` / ``"npu"`` / ``"hybrid"`` — friendly alias
        against the first plugin.

    ``ngl_override`` is non-None when the alias forced a specific
    ``n_gpu_layers`` (``cpu`` → 0, ``hybrid`` → 999). Callers should
    only overwrite their own default when this is non-None.
    """
    # Empty / "auto" → first plugin + SDK default alias.
    if not device_map or device_map == 'auto':
        plugins = get_plugin_list()
        if not plugins:
            return None, None, None
        plugin_id = plugins[0]
        return _call_sdk(plugin_id, model_name, None)

    key = device_map.lower()

    # Bare friendly alias — pick the first plugin and let the SDK translate.
    if key in _KNOWN_ALIASES:
        plugins = get_plugin_list()
        plugin_id = plugins[0] if plugins else PLUGIN_LLAMA_CPP
        return _call_sdk(plugin_id, model_name, key)

    if ':' in device_map:
        plugin_id, device_id = device_map.split(':', 1)
        if device_id.lower() in _KNOWN_ALIASES:
            return _call_sdk(plugin_id, model_name, device_id.lower())
        # Concrete device id. qairt only exposes its NPU — coerce.
        if plugin_id == PLUGIN_QAIRT and device_id.upper() != 'NPU':
            print(
                f'warning: qairt plugin only supports NPU inference; '
                f'ignoring device_map={device_map!r} and running on NPU',
                file=sys.stderr,
            )
            return plugin_id, 'NPU', None
        return plugin_id, device_id, None

    # Bare plugin id (e.g. "llama_cpp", "qairt") — SDK picks the default.
    return _call_sdk(device_map, model_name, None)


def _call_sdk(
    plugin_id: str,
    model_name: str | None,
    alias: str | None,
) -> tuple[str, str | None, int | None]:
    """Call ``geniex_resolve_device`` and map its result into the
    ``(plugin_id, device_id, ngl_override)`` triple this module returns.

    ``ngl_default=-1`` is a sentinel so we can detect "SDK didn't force
    a value" and surface it as ``None`` to the caller.
    """
    device_id, ngl, warning = resolve_device(plugin_id, model_name, alias, -1)
    if warning:
        print(f'warning: {warning}', file=sys.stderr)
    # SDK returned the sentinel unchanged → no override. Any other value
    # (0 for cpu, 999 for hybrid, or caller-provided) means "do override".
    ngl_override: int | None = None if ngl == -1 else ngl
    return plugin_id, device_id, ngl_override


def _resolve_local_anchor(path: str) -> str:
    """Return an anchor file for a local directory/file path.

    The C++ side derives the directory via ``parent_path()``, so we need to
    point at a file inside the directory rather than the directory itself.
    """
    if os.path.isdir(path):
        anchor = os.path.join(path, 'tokenizer.json')
        if not os.path.isfile(anchor):
            entries = [e for e in os.listdir(path) if os.path.isfile(os.path.join(path, e))]
            if not entries:
                raise FileNotFoundError(f'No files found in model directory: {path}')
            anchor = os.path.join(path, entries[0])
        return anchor
    return path


def _resolve_model_sources(
    model_name_or_path: str,
    quant: str | None,
    hf_token: str | None,
) -> tuple[str, str | None, str | None, _mm.ModelPaths | None]:
    """Return ``(model_path, mmproj_path, tokenizer_path, paths)``.

    Local paths are anchored directly; everything else goes through the
    model manager (``geniex_model_*`` FFI) which handles alias resolution,
    download, and path resolution.

    AiHub/LocalFs models cached by a previous ``pull`` hit the fast path:
    ``get_paths`` succeeds without talking to any hub, so notebook-style
    ``from_pretrained("qualcomm/Qwen3-4B")`` works without forcing users
    to re-specify ``hub='aihub'``. Only when the cache is empty do we
    fall through to ``ensure_cached`` (HF/auto).
    """
    if os.path.exists(model_name_or_path):
        return _resolve_local_anchor(model_name_or_path), None, None, None

    key = f'{model_name_or_path}:{quant}' if quant else model_name_or_path
    try:
        cached = _mm.get_paths(key)
        return cached.model_path, cached.mmproj_path, cached.tokenizer_path, cached
    except Exception:  # noqa: BLE001 — any failure = cache miss, fall through
        pass

    paths = _mm.ensure_cached(
        model_name_or_path,
        quant=quant,
        hub='auto',
        hf_token=hf_token,
    )
    return paths.model_path, paths.mmproj_path, paths.tokenizer_path, paths


def _build_model_config(plugin_id: str | None, n_ctx: int, n_gpu_layers: int, **kwargs) -> geniex_ModelConfig:
    if plugin_id == PLUGIN_QAIRT:
        if n_gpu_layers != 0:
            _logger.debug('qairt plugin does not consume n_gpu_layers; forcing 0')
            n_gpu_layers = 0
        if n_ctx != 0:
            _logger.debug('qairt plugin does not consume n_ctx; forcing 0')
            n_ctx = 0
    cfg = geniex_ModelConfig(
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
    )
    # Pass through any recognised extra kwargs
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


class AutoModelForCausalLM:
    """Factory for text-only causal language models."""

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
        tokenizer_path: str | None = None,
        license_id: str | None = None,
        license_key: str | None = None,
        hf_token: str | None = None,
        **kwargs,
    ) -> GeniexLLM:
        """Load a causal LM and return a GeniexLLM instance.

        Args:
            model_name_or_path: HuggingFace repo id or local path.
            model_name: Override the registry model name (e.g. 'granite4' for QAIRT).
                        Defaults to model_name_or_path when not set.
            quant: Quantization variant (e.g. 'Q4_K_M').  Used to filter files
                when downloading from HuggingFace Hub.
            device_map: ``'auto'`` | ``'cpu'`` | ``'gpu'`` | ``'npu'`` |
                ``'hybrid'`` | ``'<plugin_id>'`` |
                ``'<plugin_id>:<device_id>'``. The default (``'auto'``)
                maps to ``hybrid`` for ``llama_cpp`` (fast per-tensor
                HTP+CPU scheduling) and to ``npu`` for ``qairt``.
                ``hybrid`` leaves ``device_id`` empty with
                ``n_gpu_layers=999``; ``npu`` pins to ``HTP0``. Run
                ``geniex-py devices`` (or
                :func:`geniex._ffi.get_device_list`) to list concrete
                device ids available on this machine.
            n_ctx: Context length (0 = model default). Forced to 0 when the
                resolved plugin is ``qairt`` (the QAIRT runtime does not
                consume ``n_ctx``).
            n_gpu_layers: Layers to offload to GPU/NPU (999 = offload all).
                Forced to 0 when ``device_map`` resolves to CPU, to 999
                when it resolves to ``hybrid``, and to 0 when the resolved
                plugin is ``qairt`` (the QAIRT runtime does not consume
                ``n_gpu_layers``).
            tokenizer_path: Optional override for tokenizer file path.
            license_id: NPU licence ID.
            license_key: NPU licence key.
        """
        ensure_init()
        model_path, _mmproj, _tok, paths = _resolve_model_sources(model_name_or_path, quant, hf_token)
        # QAIRT uses `model_name` as a registry key (e.g. `qwen3_4b`),
        # not the org/repo string. Fall back to the manifest's ModelName
        # whenever the caller didn't override and the cached model is QAIRT.
        resolved_name = model_name or (
            paths.model_name if paths and paths.plugin_id == PLUGIN_QAIRT else model_name_or_path
        )
        plugin_id, device_id, ngl_override = _resolve_device(device_map, resolved_name)
        if ngl_override is not None:
            n_gpu_layers = ngl_override
        config = _build_model_config(plugin_id, n_ctx, n_gpu_layers, **kwargs)

        inp = geniex_LlmCreateInput(
            model_name=resolved_name.encode(),
            model_path=model_path.encode(),
            config=config,
        )
        resolved_tokenizer = tokenizer_path or _tok
        if resolved_tokenizer:
            inp.tokenizer_path = resolved_tokenizer.encode()
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
        return GeniexLLM(handle)


class AutoModelForVision2Seq:
    """Factory for vision-language / multimodal models."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        quant: str | None = None,
        device_map: str = 'auto',
        n_ctx: int = 0,
        n_gpu_layers: int = 999,
        mmproj_path: str | None = None,
        tokenizer_path: str | None = None,
        license_id: str | None = None,
        license_key: str | None = None,
        hf_token: str | None = None,
        **kwargs,
    ) -> GeniexVLM:
        """Load a VLM and return a GeniexVLM instance.

        Args:
            model_name_or_path: HuggingFace repo id or local path.
            quant: Quantization variant.
            device_map: ``'auto'`` | ``'cpu'`` | ``'gpu'`` | ``'npu'`` |
                ``'hybrid'`` | ``'<plugin_id>'`` |
                ``'<plugin_id>:<device_id>'``. Default (``'auto'``) maps
                to ``hybrid`` for ``llama_cpp`` and ``npu`` for ``qairt``.
                See :class:`AutoModelForCausalLM.from_pretrained` for
                full semantics.
            n_ctx: Context length (0 = model default). Forced to 0 when the
                resolved plugin is ``qairt`` (the QAIRT runtime does not
                consume ``n_ctx``).
            n_gpu_layers: Layers to offload to GPU/NPU (999 = offload all).
                Forced to 0 when ``device_map`` resolves to CPU, to 999
                when it resolves to ``hybrid``, and to 0 when the resolved
                plugin is ``qairt`` (the QAIRT runtime does not consume
                ``n_gpu_layers``).
            mmproj_path: Path to the multimodal projector file.
            tokenizer_path: Optional override for tokenizer file path.
            license_id: NPU licence ID.
            license_key: NPU licence key.
        """
        ensure_init()
        model_path, _mmproj, _tok, paths = _resolve_model_sources(model_name_or_path, quant, hf_token)
        resolved_name = paths.model_name if paths and paths.plugin_id == PLUGIN_QAIRT else model_name_or_path
        plugin_id, device_id, ngl_override = _resolve_device(device_map, resolved_name)
        if ngl_override is not None:
            n_gpu_layers = ngl_override
        config = _build_model_config(plugin_id, n_ctx, n_gpu_layers, **kwargs)

        inp = geniex_VlmCreateInput(
            model_name=resolved_name.encode(),
            model_path=model_path.encode(),
            config=config,
        )
        resolved_mmproj = mmproj_path or _mmproj
        if resolved_mmproj:
            inp.mmproj_path = resolved_mmproj.encode()
        resolved_tokenizer = tokenizer_path or _tok
        if resolved_tokenizer:
            inp.tokenizer_path = resolved_tokenizer.encode()
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
        return GeniexVLM(handle)
