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

import os
from ctypes import byref, c_void_p

from . import model_manager as _mm
from ._ffi._api import _check, ensure_init, get_plugin_list, load_library
from ._ffi._types import geniex_LlmCreateInput, geniex_ModelConfig, geniex_VlmCreateInput
from .modeling import GeniexLLM, GeniexVLM

# llama_cpp friendly-name mapping.
#
# Key insight: llama.cpp's *best* NPU/GPU path is when ``device_id`` is left
# unset and ``n_gpu_layers`` is high. Then llama.cpp assigns each tensor to
# whichever backend supports it (HTP for computable ops, CPU for fallbacks),
# which is its built-in hybrid execution. Passing ``device_id="HTP0"`` forces
# ``mpar.devices = {HTP0}`` and disables that hybrid assignment — slower.
#
# So:
#   cpu → empty device id + n_gpu_layers=0     (pure CPU)
#   gpu → "GPUOpenCL"      + n_gpu_layers=None (Adreno; GPU must be explicit)
#   npu → empty device id  + n_gpu_layers=None (hybrid HTP+CPU, fast path)
#
# If you need to pin to a specific HTP device, pass the literal id as
# ``device_map="llama_cpp:HTP0"`` / ``"llama_cpp:HTP0,HTP1,HTP2,HTP3"``.
_LLAMA_CPP_DEVICE_ALIASES: dict[str, tuple[str, int | None]] = {
    'cpu': ('', 0),
    'gpu': ('GPUOpenCL', None),
    'npu': ('', None),
}


def _resolve_device(device_map: str) -> tuple[str | None, str | None, int | None]:
    """Return ``(plugin_id, device_id, ngl_override)`` from a ``device_map`` string.

    Accepted forms:
      - ``"auto"`` — pick the first plugin and let it choose its own default device.
      - ``"<plugin_id>"`` — use this plugin, let it pick the default device.
      - ``"<plugin_id>:<device_id>"`` — fully specified.
      - ``"cpu"`` / ``"gpu"`` / ``"npu"`` — friendly alias for
        ``llama_cpp:<translated>``; see :data:`_LLAMA_CPP_DEVICE_ALIASES`.
      - ``"llama_cpp:cpu"`` / ``"llama_cpp:gpu"`` / ``"llama_cpp:npu"`` — same
        translation applied to the device portion.

    ``ngl_override`` is non-None when the alias implies a specific
    ``n_gpu_layers`` value (``cpu`` → 0). Concrete ggml device ids and
    non-friendly plugin names are passed through unchanged.

    Device ids are plugin-specific (e.g. for ``llama_cpp`` they come from
    ``ggml_backend_dev_name()`` and vary by build / host hardware). Call
    :func:`geniex._ffi.get_device_list` to enumerate them at runtime.
    """
    if device_map == 'auto' or not device_map:
        plugins = get_plugin_list()
        if not plugins:
            return None, None, None
        # Plugin only — leave device_id unset so the plugin chooses its own
        # default layout (hybrid HTP+CPU for llama_cpp on Snapdragon).
        return plugins[0], None, None

    # Bare friendly name → llama_cpp:<alias>
    if device_map.lower() in _LLAMA_CPP_DEVICE_ALIASES:
        device_id, ngl = _LLAMA_CPP_DEVICE_ALIASES[device_map.lower()]
        return 'llama_cpp', device_id or None, ngl

    if ':' in device_map:
        plugin_id, device_id = device_map.split(':', 1)
        if plugin_id == 'llama_cpp' and device_id.lower() in _LLAMA_CPP_DEVICE_ALIASES:
            translated, ngl = _LLAMA_CPP_DEVICE_ALIASES[device_id.lower()]
            return plugin_id, translated or None, ngl
        return plugin_id, device_id, None
    return device_map, None, None


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
) -> tuple[str, str | None, str | None]:
    """Return ``(model_path, mmproj_path, tokenizer_path)``.

    Local paths are anchored directly; everything else goes through the
    model manager (``geniex_model_*`` FFI) which handles alias resolution,
    download, and path resolution.
    """
    if os.path.exists(model_name_or_path):
        return _resolve_local_anchor(model_name_or_path), None, None

    paths = _mm.ensure_cached(
        model_name_or_path,
        quant=quant,
        hub='auto',
        hf_token=hf_token,
    )
    return paths.model_path, paths.mmproj_path, paths.tokenizer_path


def _build_model_config(n_ctx: int, n_gpu_layers: int, **kwargs) -> geniex_ModelConfig:
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
            device_map: 'auto' | 'cpu' | 'gpu' | 'npu' | '<plugin_id>' |
                '<plugin_id>:<device_id>'. The friendly aliases
                (``cpu|gpu|npu``) target the ``llama_cpp`` plugin; ``npu``
                in particular leaves ``device_id`` unset so llama.cpp's
                hybrid HTP+CPU scheduler runs (see
                :data:`_LLAMA_CPP_DEVICE_ALIASES`). Run ``geniex-py devices``
                (or :func:`geniex._ffi.get_device_list`) to enumerate
                concrete device ids available on this machine.
            n_ctx: Context length (0 = model default).
            n_gpu_layers: Layers to offload to GPU/NPU (999 = offload all).
                Forced to 0 when ``device_map`` resolves to CPU.
            tokenizer_path: Optional override for tokenizer file path.
            license_id: NPU licence ID.
            license_key: NPU licence key.
        """
        ensure_init()
        plugin_id, device_id, ngl_override = _resolve_device(device_map)
        if ngl_override is not None:
            n_gpu_layers = ngl_override
        model_path, _mmproj, _tok = _resolve_model_sources(model_name_or_path, quant, hf_token)
        config = _build_model_config(n_ctx, n_gpu_layers, **kwargs)

        inp = geniex_LlmCreateInput(
            model_name=(model_name or model_name_or_path).encode(),
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
            device_map: 'auto' | 'cpu' | 'gpu' | 'npu' | '<plugin_id>' |
                '<plugin_id>:<device_id>'. The friendly aliases
                (``cpu|gpu|npu``) target the ``llama_cpp`` plugin; ``npu``
                in particular leaves ``device_id`` unset so llama.cpp's
                hybrid HTP+CPU scheduler runs (see
                :data:`_LLAMA_CPP_DEVICE_ALIASES`). Run ``geniex-py devices``
                (or :func:`geniex._ffi.get_device_list`) to enumerate
                concrete device ids available on this machine.
            n_ctx: Context length (0 = model default).
            n_gpu_layers: Layers to offload to GPU/NPU (999 = offload all).
                Forced to 0 when ``device_map`` resolves to CPU.
            mmproj_path: Path to the multimodal projector file.
            tokenizer_path: Optional override for tokenizer file path.
            license_id: NPU licence ID.
            license_key: NPU licence key.
        """
        ensure_init()
        plugin_id, device_id, ngl_override = _resolve_device(device_map)
        if ngl_override is not None:
            n_gpu_layers = ngl_override
        model_path, _mmproj, _tok = _resolve_model_sources(model_name_or_path, quant, hf_token)
        config = _build_model_config(n_ctx, n_gpu_layers, **kwargs)

        inp = geniex_VlmCreateInput(
            model_name=model_name_or_path.encode(),
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
