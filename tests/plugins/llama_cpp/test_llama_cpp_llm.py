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

"""llama_cpp LLM matrix: cpu, npu, hybrid."""

from __future__ import annotations

import geniex
import pytest

from _models import LLAMA_CPP_LLM_MODEL, LLAMA_CPP_LLM_QUANT
from _quality_data import (
    LLM_QUALITY_MAX_NEW_TOKENS,
    LLM_QUALITY_PROMPTS,
    LLM_QUALITY_SEED,
    LLM_QUALITY_TEMPERATURE,
)

pytestmark = pytest.mark.llm


@pytest.mark.parametrize('device_map', ['cpu', 'npu', 'hybrid'])
def test_generate_blocking(llama_cpp_llm_paths, device_map):
    with geniex.AutoModelForCausalLM.from_pretrained(
        LLAMA_CPP_LLM_MODEL,
        quant=LLAMA_CPP_LLM_QUANT,
        device_map=device_map,
    ) as llm:
        assert isinstance(llm, geniex.GenieXLLM)
        out = llm.generate('Say hi.', max_new_tokens=8, temperature=0.0, seed=42)
        assert isinstance(out, geniex.GenerateOutput)
        assert out.text
        assert out.profile.generated_tokens > 0


@pytest.mark.parametrize('device_map', ['cpu'])
def test_generate_stream(llama_cpp_llm_paths, device_map):
    with geniex.AutoModelForCausalLM.from_pretrained(
        LLAMA_CPP_LLM_MODEL,
        quant=LLAMA_CPP_LLM_QUANT,
        device_map=device_map,
    ) as llm:
        streamer = llm.generate('Say hi.', max_new_tokens=8, temperature=0.0, seed=42, stream=True)
        assert isinstance(streamer, geniex.TextIteratorStreamer)
        chunks = list(streamer)
        assert chunks
        assert streamer.output is not None
        assert streamer.output.text


@pytest.mark.parametrize('device_map', ['cpu', 'npu', 'hybrid'])
@pytest.mark.parametrize(('prompt', 'expected'), LLM_QUALITY_PROMPTS)
def test_quality_keywords(llama_cpp_llm_paths, device_map, prompt, expected):
    # Mirrors run_scorecard_posix.py:_section_quality_checks (test-llama.cpp):
    # same prompts, n_predict=256, seed=1, plugin-default sampler (NOT greedy
    # — see _quality_data.py module docstring), case-insensitive substring
    # match. A regression on either side is directly comparable.
    with geniex.AutoModelForCausalLM.from_pretrained(
        LLAMA_CPP_LLM_MODEL,
        quant=LLAMA_CPP_LLM_QUANT,
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
        assert expected.lower() in out.text.lower(), (
            f'prompt={prompt!r} expected_substring={expected!r} ' f'device_map={device_map!r} got={out.text!r}'
        )
