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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .modeling import GeniexLLM, GeniexVLM


class ModelTokenizer:
    """Facade that provides a transformers-compatible apply_chat_template() interface.

    Does not perform standalone tokenization — the underlying C runtime requires
    a loaded model handle to apply the chat template.
    """

    def __init__(self, model: 'GeniexLLM | GeniexVLM') -> None:
        self._model = model

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        tools: list[dict] | str | None = None,
    ) -> str:
        """Format a list of chat messages using the model's built-in chat template.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            tokenize: Must be False — standalone tokenization is not supported.
            add_generation_prompt: Whether to append the generation prompt suffix.
            enable_thinking: Enable thinking mode (Qwen reasoning models).
            tools: Tool definitions as a list of dicts or a pre-serialised JSON string.

        Returns:
            Formatted prompt string ready to pass to model.generate().
        """
        if tokenize:
            raise ValueError(
                'tokenize=True is not supported by geniex — the C runtime decodes tokens internally. '
                'Use tokenize=False and pass the returned string directly to model.generate().'
            )

        tools_str: str | None = None
        if tools is not None:
            tools_str = tools if isinstance(tools, str) else json.dumps(tools)

        return self._model._apply_chat_template(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tools=tools_str,
        )
