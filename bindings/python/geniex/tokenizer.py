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

import json
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .modeling import GeniexLLM, GeniexVLM


class ModelTokenizer:
    """Transformers-compatible facade for ``apply_chat_template`` on a loaded model."""

    def __init__(self, model: 'GeniexLLM | GeniexVLM') -> None:
        self._model = model

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
        tools: list[dict] | str | None = None,
    ) -> str:
        """Format chat ``messages`` using the loaded model's chat template.

        ``tokenize=True`` is rejected — the C runtime handles tokenisation
        internally, so callers should pass the returned string straight to
        :meth:`GeniexLLM.generate`. ``tools`` accepts a list of dicts or a
        pre-serialised JSON string.

        ``enable_thinking`` semantics:

        * ``None`` (default) — auto-resolve. For thinking-capable models we
          enable thinking; for non-thinking models we still pass ``True`` so
          the underlying ChatML template skips the empty
          ``<think>\\n\\n</think>\\n\\n`` suppression block (that block is
          only meaningful for *thinking* models being asked to skip a turn,
          and on non-thinking instruct models it derails generation).
        * ``True`` — same as auto-resolve.
        * ``False`` — explicitly ask a thinking-capable model to skip its
          thinking turn. Forced to ``True`` (with a warning) on non-thinking
          models, where the suppression block is OOD.

        See :attr:`GeniexLLM.supports_thinking` for how the capability is
        detected.
        """
        if tokenize:
            raise ValueError(
                'tokenize=True is not supported by geniex — the C runtime decodes tokens internally. '
                'Use tokenize=False and pass the returned string directly to model.generate().'
            )

        tools_str: str | None = None
        if tools is not None:
            tools_str = tools if isinstance(tools, str) else json.dumps(tools)

        supports_thinking = self._model.supports_thinking
        if enable_thinking is None:
            enable_thinking = True
        elif enable_thinking is False and not supports_thinking:
            warnings.warn(
                'enable_thinking=False on a non-thinking model injects an empty '
                '<think></think> block that can derail generation. Forcing True.',
                stacklevel=2,
            )
            enable_thinking = True

        return self._model._apply_chat_template(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tools=tools_str,
        )
