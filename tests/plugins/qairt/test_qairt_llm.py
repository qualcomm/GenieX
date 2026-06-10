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

"""qairt LLM matrix: npu (only supported backend for qairt)."""

from __future__ import annotations

import geniex
import pytest

from _models import QAIRT_LLM_MODEL
from _quality_data import (
    LLM_QUALITY_MAX_NEW_TOKENS,
    LLM_QUALITY_PROMPTS,
    LLM_QUALITY_SEED,
    LLM_QUALITY_TEMPERATURE,
)

pytestmark = pytest.mark.llm


@pytest.mark.parametrize('device_map', ['npu'])
def test_generate_blocking(qairt_llm_paths, device_map):
    with geniex.AutoModelForCausalLM.from_pretrained(
        QAIRT_LLM_MODEL,
        device_map=device_map,
    ) as llm:
        assert isinstance(llm, geniex.GenieXLLM)
        out = llm.generate('Say hi.', max_new_tokens=16, temperature=0.0, seed=42)
        assert isinstance(out, geniex.GenerateOutput)
        assert out.text
        assert out.profile.generated_tokens > 0


@pytest.mark.parametrize('device_map', ['npu'])
@pytest.mark.parametrize(('prompt', 'expected'), LLM_QUALITY_PROMPTS)
def test_quality_keywords(qairt_llm_paths, device_map, prompt, expected):
    # Same prompts / sampler as the llama_cpp matrix so QAIRT NPU output is
    # comparable cross-plugin. Upstream test-llama.cpp's scorecard has no
    # QAIRT path; this is geniex's own quality regression on the NPU.
    with geniex.AutoModelForCausalLM.from_pretrained(
        QAIRT_LLM_MODEL,
        device_map=device_map,
    ) as llm:
        out = llm.generate(
            prompt,
            max_new_tokens=LLM_QUALITY_MAX_NEW_TOKENS,
            temperature=LLM_QUALITY_TEMPERATURE,
            seed=LLM_QUALITY_SEED,
        )
        assert isinstance(out, geniex.GenerateOutput)
        assert out.text, f'empty completion for prompt={prompt!r}'
        # See test_llama_cpp_llm.test_quality_keywords for why this is hoisted.
        matched = expected.lower() in out.text.lower()
        assert matched, (
            f'prompt={prompt!r} expected_substring={expected!r} ' f'device_map={device_map!r} got={out.text!r}'
        )
