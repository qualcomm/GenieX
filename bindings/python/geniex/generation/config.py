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

from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    """Transformers-compatible generation parameters.

    Numeric sampler fields default to 0 (= defer to bundle/plugin default).
    """

    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int = 0
    stop: list[str] = field(default_factory=list)
    grammar: str | None = None
    json_mode: bool = False
    images: list[str] = field(default_factory=list)
    audios: list[str] = field(default_factory=list)
    stream: bool = False
