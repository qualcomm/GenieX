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

"""End-to-end LLM test against a small real GGUF on the llama_cpp plugin."""

from __future__ import annotations

import pytest

import geniex

from .conftest import LLAMA_CPP_MODEL, LLAMA_CPP_QUANT, _device_tests_enabled, _is_snapdragon_host


@pytest.fixture(scope='module')
def llm(llama_cpp_paths):
    model = geniex.AutoModelForCausalLM.from_pretrained(
        LLAMA_CPP_MODEL,
        quant=LLAMA_CPP_QUANT,
    )
    yield model
    model.close()


def test_instance_types(llm):
    assert isinstance(llm, geniex.GeniexLLM)
    assert llm.tokenizer is not None


def test_apply_chat_template_returns_string(llm):
    text = llm.tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'hi'}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(text, str) and text


def test_generate_blocking_produces_output(llm):
    out = llm.generate('Say hi.', max_new_tokens=16, temperature=0.0, seed=42)
    assert isinstance(out, geniex.GenerateOutput)
    assert out.text
    assert out.profile.generated_tokens > 0


def test_generate_stream_yields_chunks(llm):
    streamer = llm.generate('Say hi.', max_new_tokens=16, temperature=0.0, seed=42, stream=True)
    assert isinstance(streamer, geniex.TextIteratorStreamer)
    chunks = list(streamer)
    assert chunks, 'streamer yielded no chunks'
    assert streamer.output is not None
    assert streamer.output.text


def test_reset_allows_second_generation(llm):
    llm.reset()
    out = llm.generate('Say hi.', max_new_tokens=8, temperature=0.0, seed=1)
    assert out.text


def test_context_manager_closes_handle(llama_cpp_paths):
    with geniex.AutoModelForCausalLM.from_pretrained(
        LLAMA_CPP_MODEL,
        quant=LLAMA_CPP_QUANT,
    ) as m:
        assert isinstance(m, geniex.GeniexLLM)


def test_streamer_cancel_stops_generation(llm):
    llm.reset()
    streamer = llm.generate(
        'Write a very long essay about snails.',
        max_new_tokens=512,
        temperature=0.0,
        seed=2,
        stream=True,
    )
    chunks = []
    for i, chunk in enumerate(streamer):
        chunks.append(chunk)
        if i >= 3:
            streamer.cancel()
    # Generation must end even though max_new_tokens was not reached.
    assert streamer.output is not None
    assert len(chunks) < 512


@pytest.mark.parametrize('device_map', ['cpu', 'gpu', 'npu', 'hybrid'])
def test_generate_on_device(llama_cpp_paths, device_map):
    if not _device_tests_enabled() or not _is_snapdragon_host():
        pytest.skip('per-device tests require GENIEX_DEVICE_TEST=1 on a Snapdragon host')
    with geniex.AutoModelForCausalLM.from_pretrained(
        LLAMA_CPP_MODEL,
        quant=LLAMA_CPP_QUANT,
        device_map=device_map,
    ) as llm:
        out = llm.generate('Say hi.', max_new_tokens=8, temperature=0.0, seed=42)
        assert out.text
        assert out.profile.generated_tokens > 0
