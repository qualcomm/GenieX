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

from __future__ import annotations

from ctypes import POINTER, byref, c_void_p, pointer, string_at
from typing import Iterator

from .generation.output import GenerateOutput, ProfileData
from .generation.streamer import TextIteratorStreamer
from .geniex_sdk._api import (
    GeniexError,
    _check,
    _str_list_to_c,
    load_library,
)
from .geniex_sdk._types import (
    ml_GenerationConfig,
    ml_KvCacheLoadInput,
    ml_KvCacheLoadOutput,
    ml_KvCacheSaveInput,
    ml_KvCacheSaveOutput,
    ml_LlmApplyChatTemplateInput,
    ml_LlmApplyChatTemplateOutput,
    ml_LlmChatMessage,
    ml_LlmGenerateInput,
    ml_LlmGenerateOutput,
    ml_SamplerConfig,
    ml_VlmApplyChatTemplateInput,
    ml_VlmApplyChatTemplateOutput,
    ml_VlmChatMessage,
    ml_VlmContent,
    ml_VlmGenerateInput,
    ml_VlmGenerateOutput,
    ml_token_callback,
)
from .tokenizer import ModelTokenizer


def _enc(s: str | None) -> bytes | None:
    return s.encode() if s else None


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
) -> ml_SamplerConfig:
    return ml_SamplerConfig(
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
    sampler: ml_SamplerConfig,
    images: list[str],
    audios: list[str],
) -> tuple[ml_GenerationConfig, object, object, object, object, object]:
    """Returns (config, *arrays) — callers must keep arrays alive for the call."""
    stop_arr, stop_count = _str_list_to_c(stop)
    img_arr, img_count = _str_list_to_c(images)
    aud_arr, aud_count = _str_list_to_c(audios)

    cfg = ml_GenerationConfig(
        max_tokens=max_new_tokens,
        stop_count=stop_count,
        sampler_config=pointer(sampler),
        image_count=img_count,
        audio_count=aud_count,
    )
    if stop_arr is not None:
        from ctypes import cast, c_char_p, POINTER

        cfg.stop = cast(stop_arr, POINTER(c_char_p))
    if img_arr is not None:
        from ctypes import cast, c_char_p, POINTER

        cfg.image_paths = cast(img_arr, POINTER(c_char_p))
    if aud_arr is not None:
        from ctypes import cast, c_char_p, POINTER

        cfg.audio_paths = cast(aud_arr, POINTER(c_char_p))

    return cfg, stop_arr, img_arr, aud_arr


# ---------------------------------------------------------------------------
# GeniexLLM
# ---------------------------------------------------------------------------


class GeniexLLM:
    """Internal LLM wrapper around a C ml_LLM* handle."""

    def __init__(self, handle: c_void_p) -> None:
        self._handle = handle
        self.tokenizer = ModelTokenizer(self)

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    def _apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool,
        enable_thinking: bool,
        tools: str | None,
    ) -> str:
        lib = load_library()
        count = len(messages)
        MsgArray = ml_LlmChatMessage * count
        c_msgs = MsgArray(*[
            ml_LlmChatMessage(
                role=msg['role'].encode(),
                content=(msg['content'] if isinstance(msg['content'], str) else '').encode(),
            )
            for msg in messages
        ])
        inp = ml_LlmApplyChatTemplateInput(
            messages=c_msgs,
            message_count=count,
            tools=_enc(tools),
            enable_thinking=enable_thinking,
            add_generation_prompt=add_generation_prompt,
        )
        out = ml_LlmApplyChatTemplateOutput()
        _check(lib.ml_llm_apply_chat_template(self._handle, byref(inp), byref(out)))
        result = string_at(out.formatted_text).decode() if out.formatted_text else ''
        if out.formatted_text:
            lib.ml_free(out.formatted_text)
        return result

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        repetition_penalty: float = 1.1,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        seed: int = -1,
        stop: list[str] | None = None,
        grammar: str | None = None,
        json_mode: bool = False,
        stream: bool = False,
        **_kwargs,
    ) -> GenerateOutput | TextIteratorStreamer:
        stop = stop or []
        sampler = _build_sampler(
            temperature, top_p, top_k, min_p,
            repetition_penalty, presence_penalty, frequency_penalty,
            seed, grammar, json_mode,
        )
        cfg, _sa, _ia, _aa = _build_gen_config(max_new_tokens, stop, sampler, [], [])

        if stream:
            return self._generate_stream(prompt, cfg, sampler, _sa, _ia, _aa)
        return self._generate_blocking(prompt, cfg, sampler, _sa, _ia, _aa)

    def _generate_blocking(self, prompt: str, cfg, sampler, *_keep) -> GenerateOutput:
        lib = load_library()

        # no-op streaming callback that just collects tokens silently
        @ml_token_callback
        def _noop(token, _ud):
            return True

        inp = ml_LlmGenerateInput(
            prompt_utf8=prompt.encode(),
            config=pointer(cfg),
            on_token=_noop,
            user_data=None,
        )
        out = ml_LlmGenerateOutput()
        _check(lib.ml_llm_generate(self._handle, byref(inp), byref(out)))
        full = string_at(out.full_text).decode() if out.full_text else ''
        profile = ProfileData.from_c(out.profile_data)
        if out.full_text:
            lib.ml_free(out.full_text)
        return GenerateOutput.from_raw(full, profile)

    def _generate_stream(self, prompt: str, cfg, sampler, *_keep) -> TextIteratorStreamer:
        streamer = TextIteratorStreamer()
        cb = streamer._make_callback()

        def _run() -> GenerateOutput:
            lib = load_library()
            inp = ml_LlmGenerateInput(
                prompt_utf8=prompt.encode(),
                config=pointer(cfg),
                on_token=cb,
                user_data=None,
            )
            out = ml_LlmGenerateOutput()
            _check(lib.ml_llm_generate(self._handle, byref(inp), byref(out)))
            full = string_at(out.full_text).decode() if out.full_text else ''
            profile = ProfileData.from_c(out.profile_data)
            if out.full_text:
                lib.ml_free(out.full_text)
            return GenerateOutput.from_raw(full, profile)

        streamer.start(_run)
        return streamer

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        lib = load_library()
        _check(lib.ml_llm_reset(self._handle))

    def save_kv_cache(self, path: str) -> None:
        lib = load_library()
        inp = ml_KvCacheSaveInput(path=path.encode())
        out = ml_KvCacheSaveOutput()
        _check(lib.ml_llm_save_kv_cache(self._handle, byref(inp), byref(out)))

    def load_kv_cache(self, path: str) -> None:
        lib = load_library()
        inp = ml_KvCacheLoadInput(path=path.encode())
        out = ml_KvCacheLoadOutput()
        _check(lib.ml_llm_load_kv_cache(self._handle, byref(inp), byref(out)))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._handle:
            lib = load_library()
            lib.ml_llm_destroy(self._handle)
            self._handle = None  # type: ignore[assignment]

    def __enter__(self) -> 'GeniexLLM':
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# GeniexVLM
# ---------------------------------------------------------------------------


def _build_vlm_messages(messages: list[dict]):
    """Convert list[dict] to (ml_VlmChatMessage array, count)."""
    count = len(messages)
    MsgArray = ml_VlmChatMessage * count
    c_msgs_list = []
    # keep content arrays alive
    _content_refs: list = []

    for msg in messages:
        role = msg['role'].encode()
        content = msg.get('content', '')

        if isinstance(content, str):
            ContentArray = ml_VlmContent * 1
            contents = ContentArray(ml_VlmContent(type=b'text', text=content.encode()))
            _content_refs.append(contents)
            c_msgs_list.append(ml_VlmChatMessage(role=role, contents=contents, content_count=1))
        elif isinstance(content, list):
            n = len(content)
            ContentArray = ml_VlmContent * n
            items = []
            for item in content:
                ctype = item.get('type', 'text').encode()
                # image field may be a path; text field is text
                text_val = (item.get('text') or item.get('image') or item.get('audio') or '').encode()
                items.append(ml_VlmContent(type=ctype, text=text_val))
            contents = ContentArray(*items)
            _content_refs.append(contents)
            c_msgs_list.append(ml_VlmChatMessage(role=role, contents=contents, content_count=n))
        else:
            c_msgs_list.append(ml_VlmChatMessage(role=role, content_count=0))

    arr = MsgArray(*c_msgs_list)
    return arr, count, _content_refs


class GeniexVLM:
    """Internal VLM wrapper around a C ml_VLM* handle."""

    def __init__(self, handle: c_void_p) -> None:
        self._handle = handle
        self.tokenizer = ModelTokenizer(self)

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    def _apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool,
        enable_thinking: bool,
        tools: str | None,
    ) -> str:
        lib = load_library()
        c_msgs, count, _refs = _build_vlm_messages(messages)
        inp = ml_VlmApplyChatTemplateInput(
            messages=c_msgs,
            message_count=count,
            tools=_enc(tools),
            enable_thinking=enable_thinking,
            # VLM template input has no add_generation_prompt field in C API
        )
        out = ml_VlmApplyChatTemplateOutput()
        _check(lib.ml_vlm_apply_chat_template(self._handle, byref(inp), byref(out)))
        result = string_at(out.formatted_text).decode() if out.formatted_text else ''
        if out.formatted_text:
            lib.ml_free(out.formatted_text)
        return result

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        repetition_penalty: float = 1.1,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        seed: int = -1,
        stop: list[str] | None = None,
        grammar: str | None = None,
        json_mode: bool = False,
        images: list[str] | None = None,
        audios: list[str] | None = None,
        stream: bool = False,
        **_kwargs,
    ) -> GenerateOutput | TextIteratorStreamer:
        stop = stop or []
        images = images or []
        audios = audios or []
        sampler = _build_sampler(
            temperature, top_p, top_k, min_p,
            repetition_penalty, presence_penalty, frequency_penalty,
            seed, grammar, json_mode,
        )
        cfg, _sa, _ia, _aa = _build_gen_config(max_new_tokens, stop, sampler, images, audios)

        if stream:
            return self._generate_stream(prompt, cfg, sampler, _sa, _ia, _aa)
        return self._generate_blocking(prompt, cfg, sampler, _sa, _ia, _aa)

    def _generate_blocking(self, prompt: str, cfg, sampler, *_keep) -> GenerateOutput:
        lib = load_library()

        @ml_token_callback
        def _noop(token, _ud):
            return True

        inp = ml_VlmGenerateInput(
            prompt_utf8=prompt.encode(),
            config=pointer(cfg),
            on_token=_noop,
            user_data=None,
        )
        out = ml_VlmGenerateOutput()
        _check(lib.ml_vlm_generate(self._handle, byref(inp), byref(out)))
        full = string_at(out.full_text).decode() if out.full_text else ''
        profile = ProfileData.from_c(out.profile_data)
        if out.full_text:
            lib.ml_free(out.full_text)
        return GenerateOutput.from_raw(full, profile)

    def _generate_stream(self, prompt: str, cfg, sampler, *_keep) -> TextIteratorStreamer:
        streamer = TextIteratorStreamer()
        cb = streamer._make_callback()

        def _run() -> GenerateOutput:
            lib = load_library()
            inp = ml_VlmGenerateInput(
                prompt_utf8=prompt.encode(),
                config=pointer(cfg),
                on_token=cb,
                user_data=None,
            )
            out = ml_VlmGenerateOutput()
            _check(lib.ml_vlm_generate(self._handle, byref(inp), byref(out)))
            full = string_at(out.full_text).decode() if out.full_text else ''
            profile = ProfileData.from_c(out.profile_data)
            if out.full_text:
                lib.ml_free(out.full_text)
            return GenerateOutput.from_raw(full, profile)

        streamer.start(_run)
        return streamer

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        lib = load_library()
        _check(lib.ml_vlm_reset(self._handle))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._handle:
            lib = load_library()
            lib.ml_vlm_destroy(self._handle)
            self._handle = None  # type: ignore[assignment]

    def __enter__(self) -> 'GeniexVLM':
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
