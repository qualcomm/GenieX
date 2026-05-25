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

import re
from dataclasses import dataclass, field

from .._ffi._types import geniex_ProfileData

_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)


def _format_us(us: int) -> str:
    if us <= 0:
        return '0 µs'
    if us < 1_000:
        return f'{us} µs'
    if us < 1_000_000:
        return f'{us / 1_000:.1f} ms'
    return f'{us / 1_000_000:.2f} s'


@dataclass(repr=False)
class ProfileData:
    ttft: int = 0
    prompt_time: int = 0
    decode_time: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_speed: float = 0.0
    decode_speed: float = 0.0
    stop_reason: str | None = None
    backend: str | None = None
    device: str | None = None
    quant: str | None = None
    model_path: str | None = None

    @classmethod
    def from_c(cls, c: geniex_ProfileData) -> 'ProfileData':
        stop = c.stop_reason.decode() if c.stop_reason else None
        return cls(
            ttft=c.ttft,
            prompt_time=c.prompt_time,
            decode_time=c.decode_time,
            prompt_tokens=c.prompt_tokens,
            generated_tokens=c.generated_tokens,
            prefill_speed=c.prefill_speed,
            decode_speed=c.decoding_speed,
            stop_reason=stop,
        )

    def __repr__(self) -> str:
        return (
            f'ProfileData('
            f'ttft={_format_us(self.ttft)}, '
            f'prompt_time={_format_us(self.prompt_time)}, '
            f'decode_time={_format_us(self.decode_time)}, '
            f'prompt_tokens={self.prompt_tokens} tok, '
            f'generated_tokens={self.generated_tokens} tok, '
            f'prefill_speed={self.prefill_speed:.1f} tok/s, '
            f'decode_speed={self.decode_speed:.1f} tok/s, '
            f'stop_reason={self.stop_reason}, '
            f'backend={self.backend}, '
            f'device={self.device}, '
            f'quant={self.quant}, '
            f'model_path={self.model_path})'
        )


@dataclass
class GenerateOutput:
    text: str = ''
    thinking: str | None = None
    profile: ProfileData = field(default_factory=ProfileData)

    @classmethod
    def from_raw(cls, full_text: str, profile: ProfileData) -> 'GenerateOutput':
        thinking: str | None = None
        text = full_text
        m = _THINK_RE.search(full_text)
        if m:
            thinking = m.group(1).strip()
            text = _THINK_RE.sub('', full_text).strip()
        return cls(text=text, thinking=thinking, profile=profile)
