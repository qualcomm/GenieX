# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""llama_cpp LLM matrix: cpu, npu, gpu."""

from __future__ import annotations

import geniex
import pytest

from _models import LLAMA_CPP_LLM_MODEL, LLAMA_CPP_LLM_PRECISION
from _quality_data import (
    LLM_QUALITY_MAX_NEW_TOKENS,
    LLM_QUALITY_PROMPTS,
    LLM_QUALITY_SEED,
    LLM_QUALITY_TEMPERATURE,
)

pytestmark = pytest.mark.llm


@pytest.mark.parametrize('device_map', ['cpu', 'npu', 'gpu'])
def test_generate_blocking(llama_cpp_llm_paths, device_map):
    with geniex.AutoModelForCausalLM.from_pretrained(
        LLAMA_CPP_LLM_MODEL,
        precision=LLAMA_CPP_LLM_PRECISION,
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
        precision=LLAMA_CPP_LLM_PRECISION,
        device_map=device_map,
    ) as llm:
        streamer = llm.generate('Say hi.', max_new_tokens=8, temperature=0.0, seed=42, stream=True)
        assert isinstance(streamer, geniex.TextIteratorStreamer)
        chunks = list(streamer)
        assert chunks
        assert streamer.output is not None
        assert streamer.output.text


@pytest.mark.parametrize('device_map', ['cpu', 'npu', 'gpu'])
@pytest.mark.parametrize(('prompt', 'expected'), LLM_QUALITY_PROMPTS)
def test_quality_keywords(llama_cpp_llm_paths, device_map, prompt, expected):
    # Mirrors run_scorecard_posix.py:_section_quality_checks (test-llama.cpp):
    # same prompts, n_predict=256, seed=1, plugin-default sampler, chat
    # template applied. Upstream's `llama-completion` runs without `-no-cnv`,
    # so COMMON_CONVERSATION_MODE_AUTO wraps the prompt in the model's chat
    # template; feeding the raw string instead lets Qwen3-style models continue
    # in completion mode and the keyword only appears by sampler luck.
    with geniex.AutoModelForCausalLM.from_pretrained(
        LLAMA_CPP_LLM_MODEL,
        precision=LLAMA_CPP_LLM_PRECISION,
        device_map=device_map,
    ) as llm:
        formatted = llm.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        out = llm.generate(
            formatted,
            max_new_tokens=LLM_QUALITY_MAX_NEW_TOKENS,
            temperature=LLM_QUALITY_TEMPERATURE,
            seed=LLM_QUALITY_SEED,
        )
        assert isinstance(out, geniex.GenerateOutput)
        assert out.text, f'empty completion for prompt={prompt!r}'
        # Hoist the comparison into a local bool so pytest's assertion
        # introspection has nothing to walk — otherwise out.text gets
        # echoed 4–5x per failure (lower() chain + GenerateOutput repr).
        matched = expected.lower() in out.text.lower()
        assert matched, (
            f'prompt={prompt!r} expected_substring={expected!r} ' f'device_map={device_map!r} got={out.text!r}'
        )
