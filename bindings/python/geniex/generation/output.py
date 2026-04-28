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


@dataclass
class ProfileData:
    ttft: int = 0
    prompt_time: int = 0
    decode_time: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_speed: float = 0.0
    decode_speed: float = 0.0
    stop_reason: str | None = None

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
