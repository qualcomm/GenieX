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

"""llama_cpp LLM matrix: cpu, gpu, npu, hybrid."""

from __future__ import annotations

import geniex
import pytest

from _models import LLAMA_CPP_LLM_MODEL, LLAMA_CPP_LLM_QUANT

pytestmark = pytest.mark.llm


@pytest.mark.parametrize('device_map', ['cpu', 'gpu', 'npu', 'hybrid'])
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
