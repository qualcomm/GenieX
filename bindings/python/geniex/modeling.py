# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from ctypes import POINTER, byref, c_char_p, c_void_p, cast, pointer, string_at

from ._ffi._api import GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH, _check, _str_list_to_c, load_library
from ._ffi._types import (
    geniex_GenerationConfig,
    geniex_KvCacheLoadInput,
    geniex_KvCacheLoadOutput,
    geniex_KvCacheSaveInput,
    geniex_KvCacheSaveOutput,
    geniex_LlmApplyChatTemplateInput,
    geniex_LlmApplyChatTemplateOutput,
    geniex_LlmChatMessage,
    geniex_LlmGenerateInput,
    geniex_LlmGenerateOutput,
    geniex_SamplerConfig,
    geniex_token_callback,
    geniex_VlmApplyChatTemplateInput,
    geniex_VlmApplyChatTemplateOutput,
    geniex_VlmCapabilities,
    geniex_VlmChatMessage,
    geniex_VlmContent,
    geniex_VlmGenerateInput,
    geniex_VlmGenerateOutput,
)
from .generation.output import GenerateOutput, ProfileData
from .generation.streamer import TextIteratorStreamer
from .tokenizer import ModelTokenizer


def _enc(s: str | None) -> bytes | None:
    return s.encode() if s else None


def _decode_utf8(p) -> str:
    # Lenient UTF-8 decode for C-string outputs. The SDK can hand back a buffer
    # whose tail is a partial multibyte sequence — e.g. when generation stops
    # at max_new_tokens mid-character — and a strict decode would raise.
    return string_at(p).decode('utf-8', errors='replace') if p else ''


def _build_sampler(
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    seed: int,
    grammar: str | None,
    json_mode: bool,
) -> geniex_SamplerConfig:
    return geniex_SamplerConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        seed=seed,
        grammar_string=_enc(grammar),
        enable_json=json_mode,
    )


def _build_gen_config(
    max_new_tokens: int,
    stop: list[str],
    sampler: geniex_SamplerConfig,
    images: list[str],
    audios: list[str],
    sliding_window: bool = False,
    sliding_window_n_keep: int = 0,
) -> tuple[geniex_GenerationConfig, object, object, object]:
    # Callers must keep the returned arrays alive until after the C call
    # returns — ctypes does not retain a reference through cast().
    stop_arr, stop_count = _str_list_to_c(stop)
    img_arr, img_count = _str_list_to_c(images)
    aud_arr, aud_count = _str_list_to_c(audios)

    cfg = geniex_GenerationConfig(
        max_tokens=max_new_tokens,
        stop_count=stop_count,
        sampler_config=pointer(sampler),
        image_count=img_count,
        audio_count=aud_count,
        sliding_window=sliding_window,
        sliding_window_n_keep=sliding_window_n_keep,
    )
    if stop_arr is not None:
        cfg.stop = cast(stop_arr, POINTER(c_char_p))
    if img_arr is not None:
        cfg.image_paths = cast(img_arr, POINTER(c_char_p))
    if aud_arr is not None:
        cfg.audio_paths = cast(aud_arr, POINTER(c_char_p))

    return cfg, stop_arr, img_arr, aud_arr


def _apply_meta(profile: ProfileData, meta: dict | None) -> ProfileData:
    if meta:
        profile.backend = meta.get('backend')
        profile.device = meta.get('device')
        profile.quant = meta.get('quant')
        profile.model_path = meta.get('model_path')
    return profile


class GenieXLLM:
    """LLM handle returned by :meth:`AutoModelForCausalLM.from_pretrained`."""

    def __init__(self, handle: c_void_p, meta: dict | None = None) -> None:
        self._handle = handle
        self._meta = meta
        self.tokenizer = ModelTokenizer(self)

    def __repr__(self) -> str:
        lines = []
        if self._meta:
            if self._meta.get('model_name'):
                lines.append(f"  model='{self._meta['model_name']}'")
            if self._meta.get('backend'):
                lines.append(f"  backend='{self._meta['backend']}'")
            if self._meta.get('device'):
                lines.append(f"  device='{self._meta['device']}'")
            if self._meta.get('quant'):
                lines.append(f"  quant='{self._meta['quant']}'")
        if lines:
            inner = ',\n'.join(lines)
            return f'GenieXLLM(\n{inner},\n)'
        return 'GenieXLLM()'

    @property
    def supports_thinking(self) -> bool:
        """Whether the loaded model has a thinking mode (parsed once at load
        time from the model's own chat template). Drives the default value of
        ``apply_chat_template(enable_thinking=...)``. Falls back to ``True``
        when the bundle ships no on-disk ``tokenizer_config.json`` (e.g. GGUF
        models embed the template inside the file)."""
        if self._meta is None:
            return True
        detected = self._meta.get('supports_thinking')
        return True if detected is None else bool(detected)

    def _apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool,
        enable_thinking: bool,
        tools: str | None,
    ) -> str:
        lib = load_library()
        count = len(messages)
        MsgArray = geniex_LlmChatMessage * count
        c_msgs = MsgArray(
            *[
                geniex_LlmChatMessage(
                    role=msg['role'].encode(),
                    content=(msg['content'] if isinstance(msg['content'], str) else '').encode(),
                )
                for msg in messages
            ]
        )
        inp = geniex_LlmApplyChatTemplateInput(
            messages=c_msgs,
            message_count=count,
            tools=_enc(tools),
            enable_thinking=enable_thinking,
            add_generation_prompt=add_generation_prompt,
        )
        out = geniex_LlmApplyChatTemplateOutput()
        _check(lib.geniex_llm_apply_chat_template(self._handle, byref(inp), byref(out)))
        result = _decode_utf8(out.formatted_text)
        if out.formatted_text:
            lib.geniex_free(out.formatted_text)
        return result

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        # 0 = defer to bundle/plugin default. Pass non-zero to override.
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        seed: int = 0,
        stop: list[str] | None = None,
        grammar: str | None = None,
        json_mode: bool = False,
        stream: bool = False,
        # Opt-in ring-buffer context eviction (qairt only).
        sliding_window: bool = False,
        sliding_window_n_keep: int = 0,
        **_kwargs,
    ) -> GenerateOutput | TextIteratorStreamer:
        """Generate text from ``prompt``.

        Returns a :class:`GenerateOutput` when ``stream=False`` (default),
        or a :class:`TextIteratorStreamer` that yields token chunks and
        exposes ``.output`` once the generation thread finishes.
        """
        stop = stop or []
        sampler = _build_sampler(
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            seed,
            grammar,
            json_mode,
        )
        cfg, _sa, _ia, _aa = _build_gen_config(
            max_new_tokens, stop, sampler, [], [], sliding_window, sliding_window_n_keep
        )

        if stream:
            return self._generate_stream(prompt, cfg, sampler, _sa, _ia, _aa)
        return self._generate_blocking(prompt, cfg, sampler, _sa, _ia, _aa)

    def _generate_blocking(self, prompt: str, cfg, sampler, *_keep) -> GenerateOutput:
        lib = load_library()

        @geniex_token_callback
        def _noop(token, _ud):
            return True

        inp = geniex_LlmGenerateInput(
            prompt_utf8=prompt.encode(),
            config=pointer(cfg),
            on_token=_noop,
            user_data=None,
        )
        out = geniex_LlmGenerateOutput()
        rc = lib.geniex_llm_generate(self._handle, byref(inp), byref(out))
        full = _decode_utf8(out.full_text)
        profile = _apply_meta(ProfileData.from_c(out.profile_data), self._meta)
        if out.full_text:
            lib.geniex_free(out.full_text)
        if rc == GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH:
            # Absorb the C error: surface truncation as a normal return with a
            # synthesized discriminator on the profile. The C plugin reports
            # 'length' for both max_tokens and context-length truncation; we
            # promote the latter to its own value so callers can distinguish.
            profile.stop_reason = 'context_length'
            return GenerateOutput.from_raw(full, profile)
        _check(rc)
        return GenerateOutput.from_raw(full, profile)

    def _generate_stream(self, prompt: str, cfg, sampler, *_keep) -> TextIteratorStreamer:
        streamer = TextIteratorStreamer()
        cb = streamer._make_callback()

        def _run() -> GenerateOutput:
            lib = load_library()
            inp = geniex_LlmGenerateInput(
                prompt_utf8=prompt.encode(),
                config=pointer(cfg),
                on_token=cb,
                user_data=None,
            )
            out = geniex_LlmGenerateOutput()
            rc = lib.geniex_llm_generate(self._handle, byref(inp), byref(out))
            full = _decode_utf8(out.full_text)
            profile = _apply_meta(ProfileData.from_c(out.profile_data), self._meta)
            if out.full_text:
                lib.geniex_free(out.full_text)
            if rc == GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH:
                profile.stop_reason = 'context_length'
                return GenerateOutput.from_raw(full, profile)
            _check(rc)
            return GenerateOutput.from_raw(full, profile)

        streamer.start(_run)
        return streamer

    def reset(self) -> None:
        """Clear KV cache and reset sampler state."""
        lib = load_library()
        _check(lib.geniex_llm_reset(self._handle))

    def save_kv_cache(self, path: str) -> None:
        """Save the current KV cache to ``path``."""
        lib = load_library()
        inp = geniex_KvCacheSaveInput(path=path.encode())
        out = geniex_KvCacheSaveOutput()
        _check(lib.geniex_llm_save_kv_cache(self._handle, byref(inp), byref(out)))

    def load_kv_cache(self, path: str) -> None:
        """Restore a previously saved KV cache from ``path``."""
        lib = load_library()
        inp = geniex_KvCacheLoadInput(path=path.encode())
        out = geniex_KvCacheLoadOutput()
        _check(lib.geniex_llm_load_kv_cache(self._handle, byref(inp), byref(out)))

    def close(self) -> None:
        """Release the C handle. Idempotent."""
        if self._handle:
            lib = load_library()
            lib.geniex_llm_destroy(self._handle)
            self._handle = None  # type: ignore[assignment]

    def __enter__(self) -> 'GenieXLLM':
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _messages_have_modality(messages: list[dict], modality: str) -> bool:
    # True if the last user message contains a content block with the given
    # ``type`` (e.g. 'image' / 'audio'). Only the current turn is checked:
    # prior-turn media is already in the KV cache and must not be re-supplied.
    for msg in reversed(messages):
        if msg.get('role') != 'user':
            continue
        content = msg.get('content')
        if isinstance(content, list):
            return any(isinstance(item, dict) and item.get('type') == modality for item in content)
        return False
    return False


def _build_vlm_messages(messages: list[dict]):
    count = len(messages)
    MsgArray = geniex_VlmChatMessage * count
    c_msgs_list = []
    _content_refs: list = []  # keep content arrays alive

    for msg in messages:
        role = msg['role'].encode()
        content = msg.get('content', '')

        if isinstance(content, str):
            ContentArray = geniex_VlmContent * 1
            contents = ContentArray(geniex_VlmContent(type=b'text', text=content.encode()))
            _content_refs.append(contents)
            c_msgs_list.append(geniex_VlmChatMessage(role=role, contents=contents, content_count=1))
        elif isinstance(content, list):
            n = len(content)
            ContentArray = geniex_VlmContent * n
            items = []
            for item in content:
                ctype = item.get('type', 'text').encode()
                text_val = (item.get('text') or item.get('image') or item.get('audio') or '').encode()
                items.append(geniex_VlmContent(type=ctype, text=text_val))
            contents = ContentArray(*items)
            _content_refs.append(contents)
            c_msgs_list.append(geniex_VlmChatMessage(role=role, contents=contents, content_count=n))
        else:
            c_msgs_list.append(geniex_VlmChatMessage(role=role, content_count=0))

    arr = MsgArray(*c_msgs_list)
    return arr, count, _content_refs


class GenieXVLM:
    """VLM handle returned by :meth:`AutoModelForVision2Seq.from_pretrained`."""

    def __init__(self, handle: c_void_p, meta: dict | None = None) -> None:
        self._handle = handle
        self._meta = meta
        # Track whether the most recent apply_chat_template saw any
        # multimodal content blocks — used by generate() to detect callers
        # who built a messages payload with image/audio refs but forgot to
        # pass images=[...] / audios=[...].
        self._last_template_has_image = False
        self._last_template_has_audio = False
        self.tokenizer = ModelTokenizer(self)

    def __repr__(self) -> str:
        lines = []
        if self._meta:
            if self._meta.get('model_name'):
                lines.append(f"  model='{self._meta['model_name']}'")
            if self._meta.get('backend'):
                lines.append(f"  backend='{self._meta['backend']}'")
            if self._meta.get('device'):
                lines.append(f"  device='{self._meta['device']}'")
            if self._meta.get('quant'):
                lines.append(f"  quant='{self._meta['quant']}'")
        if lines:
            inner = ',\n'.join(lines)
            return f'GenieXVLM(\n{inner},\n)'
        return 'GenieXVLM()'

    @property
    def supports_thinking(self) -> bool:
        """See :attr:`GeniexLLM.supports_thinking`."""
        if self._meta is None:
            return True
        detected = self._meta.get('supports_thinking')
        return True if detected is None else bool(detected)

    def _apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool,
        enable_thinking: bool,
        tools: str | None,
    ) -> str:
        lib = load_library()
        self._last_template_has_image = _messages_have_modality(messages, 'image')
        self._last_template_has_audio = _messages_have_modality(messages, 'audio')
        c_msgs, count, _refs = _build_vlm_messages(messages)
        inp = geniex_VlmApplyChatTemplateInput(
            messages=c_msgs,
            message_count=count,
            tools=_enc(tools),
            enable_thinking=enable_thinking,
        )
        out = geniex_VlmApplyChatTemplateOutput()
        _check(lib.geniex_vlm_apply_chat_template(self._handle, byref(inp), byref(out)))
        result = _decode_utf8(out.formatted_text)
        if out.formatted_text:
            lib.geniex_free(out.formatted_text)
        return result

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        # 0 = defer to bundle/plugin default. Pass non-zero to override.
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        seed: int = 0,
        stop: list[str] | None = None,
        grammar: str | None = None,
        json_mode: bool = False,
        images: list[str] | None = None,
        audios: list[str] | None = None,
        stream: bool = False,
        **_kwargs,
    ) -> GenerateOutput | TextIteratorStreamer:
        """Generate text from ``prompt`` with optional ``images`` / ``audios`` file paths.

        Returns a :class:`GenerateOutput` when ``stream=False`` (default),
        or a :class:`TextIteratorStreamer` when ``stream=True``.
        """
        stop = stop or []
        images = images or []
        audios = audios or []
        if not images and self._last_template_has_image:
            raise ValueError(
                'messages reference image content but generate(images=[...]) '
                'is empty. Pass image paths via images=[...].'
            )
        if not audios and self._last_template_has_audio:
            raise ValueError(
                'messages reference audio content but generate(audios=[...]) '
                'is empty. Pass audio paths via audios=[...].'
            )
        for path in images:
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Image file not found: {path}')
        for path in audios:
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Audio file not found: {path}')
        sampler = _build_sampler(
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            seed,
            grammar,
            json_mode,
        )
        cfg, _sa, _ia, _aa = _build_gen_config(max_new_tokens, stop, sampler, images, audios)

        if stream:
            return self._generate_stream(prompt, cfg, sampler, _sa, _ia, _aa)
        return self._generate_blocking(prompt, cfg, sampler, _sa, _ia, _aa)

    def _generate_blocking(self, prompt: str, cfg, sampler, *_keep) -> GenerateOutput:
        lib = load_library()

        @geniex_token_callback
        def _noop(token, _ud):
            return True

        inp = geniex_VlmGenerateInput(
            prompt_utf8=prompt.encode(),
            config=pointer(cfg),
            on_token=_noop,
            user_data=None,
        )
        out = geniex_VlmGenerateOutput()
        rc = lib.geniex_vlm_generate(self._handle, byref(inp), byref(out))
        full = _decode_utf8(out.full_text)
        profile = _apply_meta(ProfileData.from_c(out.profile_data), self._meta)
        if out.full_text:
            lib.geniex_free(out.full_text)
        if rc == GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH:
            profile.stop_reason = 'context_length'
            return GenerateOutput.from_raw(full, profile)
        _check(rc)
        return GenerateOutput.from_raw(full, profile)

    def _generate_stream(self, prompt: str, cfg, sampler, *_keep) -> TextIteratorStreamer:
        streamer = TextIteratorStreamer()
        cb = streamer._make_callback()

        def _run() -> GenerateOutput:
            lib = load_library()
            inp = geniex_VlmGenerateInput(
                prompt_utf8=prompt.encode(),
                config=pointer(cfg),
                on_token=cb,
                user_data=None,
            )
            out = geniex_VlmGenerateOutput()
            rc = lib.geniex_vlm_generate(self._handle, byref(inp), byref(out))
            full = _decode_utf8(out.full_text)
            profile = _apply_meta(ProfileData.from_c(out.profile_data), self._meta)
            if out.full_text:
                lib.geniex_free(out.full_text)
            if rc == GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH:
                profile.stop_reason = 'context_length'
                return GenerateOutput.from_raw(full, profile)
            _check(rc)
            return GenerateOutput.from_raw(full, profile)

        streamer.start(_run)
        return streamer

    def capabilities(self) -> dict[str, bool]:
        """Return which input modalities (``vision`` / ``audio``) the loaded mmproj supports.

        Plugins without modality probes (e.g. QAIRT) return both as ``False``.
        """
        lib = load_library()
        out = geniex_VlmCapabilities()
        _check(lib.geniex_vlm_get_capabilities(self._handle, byref(out)))
        return {'vision': bool(out.supports_vision), 'audio': bool(out.supports_audio)}

    def reset(self) -> None:
        """Clear KV cache and reset sampler state."""
        lib = load_library()
        _check(lib.geniex_vlm_reset(self._handle))

    def close(self) -> None:
        """Release the C handle. Idempotent."""
        if self._handle:
            lib = load_library()
            lib.geniex_vlm_destroy(self._handle)
            self._handle = None  # type: ignore[assignment]

    def __enter__(self) -> 'GenieXVLM':
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
