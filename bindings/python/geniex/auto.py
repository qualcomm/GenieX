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

from .geniex_sdk._api import (
    _check,
    ensure_init,
    get_device_list,
    get_plugin_list,
    load_library,
)
from .geniex_sdk._types import (
    ml_LlmCreateInput,
    ml_ModelConfig,
    ml_VlmCreateInput,
)
from .modeling import GeniexLLM, GeniexVLM

_CPU_PLUGIN = 'llama_cpp'


def _resolve_device(device_map: str) -> tuple[str | None, str | None]:
    """Return (plugin_id, device_id) from a device_map string."""
    if device_map == 'cpu':
        return _CPU_PLUGIN, None
    if ':' in device_map and device_map != 'auto':
        parts = device_map.split(':', 1)
        return parts[0], parts[1]
    # 'auto' — pick first available plugin + device
    plugins = get_plugin_list()
    if not plugins:
        return None, None
    plugin_id = plugins[0]
    devices = get_device_list(plugin_id)
    device_id = devices[0][0] if devices else None
    return plugin_id, device_id


def _resolve_model_path(model_name_or_path: str, quant: str | None) -> str:
    """Resolve a local path or HuggingFace repo id to a local model path.

    When given a directory (e.g. a QAIRT model folder), returns a file inside
    it so the C++ side can derive the directory via parent_path().
    """
    if os.path.isdir(model_name_or_path):
        # Prefer tokenizer.json as the anchor file; fall back to the first file.
        anchor = os.path.join(model_name_or_path, 'tokenizer.json')
        if not os.path.isfile(anchor):
            entries = [e for e in os.listdir(model_name_or_path)
                       if os.path.isfile(os.path.join(model_name_or_path, e))]
            if not entries:
                raise FileNotFoundError(f'No files found in model directory: {model_name_or_path}')
            anchor = os.path.join(model_name_or_path, entries[0])
        return anchor
    if os.path.exists(model_name_or_path):
        return model_name_or_path

    # Download from HuggingFace Hub using huggingface_hub
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            'huggingface_hub is required to download models. '
            'Install it with: pip install huggingface_hub'
        ) from exc

    kwargs: dict = {'repo_id': model_name_or_path}
    if quant:
        # Filter to the specific quantization file(s)
        kwargs['allow_patterns'] = [f'*{quant}*']
    return snapshot_download(**kwargs)


def _build_model_config(n_ctx: int, n_gpu_layers: int, **kwargs) -> ml_ModelConfig:
    cfg = ml_ModelConfig(
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


# ---------------------------------------------------------------------------
# AutoModelForCausalLM
# ---------------------------------------------------------------------------


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
        n_gpu_layers: int = -1,
        tokenizer_path: str | None = None,
        license_id: str | None = None,
        license_key: str | None = None,
        **kwargs,
    ) -> GeniexLLM:
        """Load a causal LM and return a GeniexLLM instance.

        Args:
            model_name_or_path: HuggingFace repo id or local path.
            model_name: Override the registry model name (e.g. 'granite4' for QAIRT).
                        Defaults to model_name_or_path when not set.
            quant: Quantization variant (e.g. 'Q4_K_M').  Used to filter files
                when downloading from HuggingFace Hub.
            device_map: 'auto' | 'cpu' | '<plugin_id>:<device_id>'.
            n_ctx: Context length (0 = model default).
            n_gpu_layers: Layers to offload to GPU (-1 = all).
            tokenizer_path: Optional override for tokenizer file path.
            license_id: NPU licence ID.
            license_key: NPU licence key.
        """
        ensure_init()
        plugin_id, device_id = _resolve_device(device_map)
        model_path = _resolve_model_path(model_name_or_path, quant)
        config = _build_model_config(n_ctx, n_gpu_layers, **kwargs)

        inp = ml_LlmCreateInput(
            model_name=(model_name or model_name_or_path).encode(),
            model_path=model_path.encode(),
            config=config,
        )
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
        _check(lib.ml_llm_create(byref(inp), byref(handle)))
        return GeniexLLM(handle)


# ---------------------------------------------------------------------------
# AutoModelForVision2Seq
# ---------------------------------------------------------------------------


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
        n_gpu_layers: int = -1,
        mmproj_path: str | None = None,
        tokenizer_path: str | None = None,
        license_id: str | None = None,
        license_key: str | None = None,
        **kwargs,
    ) -> GeniexVLM:
        """Load a VLM and return a GeniexVLM instance.

        Args:
            model_name_or_path: HuggingFace repo id or local path.
            quant: Quantization variant.
            device_map: 'auto' | 'cpu' | '<plugin_id>:<device_id>'.
            n_ctx: Context length (0 = model default).
            n_gpu_layers: Layers to offload to GPU (-1 = all).
            mmproj_path: Path to the multimodal projector file.
            tokenizer_path: Optional override for tokenizer file path.
            license_id: NPU licence ID.
            license_key: NPU licence key.
        """
        ensure_init()
        plugin_id, device_id = _resolve_device(device_map)
        model_path = _resolve_model_path(model_name_or_path, quant)
        config = _build_model_config(n_ctx, n_gpu_layers, **kwargs)

        inp = ml_VlmCreateInput(
            model_name=model_name_or_path.encode(),
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
        _check(lib.ml_vlm_create(byref(inp), byref(handle)))
        return GeniexVLM(handle)
