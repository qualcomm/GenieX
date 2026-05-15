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

"""AutoModelForCausalLM VLM auto-detection tests.

Verifies that AutoModelForCausalLM returns GeniexVLM for multimodal
models and GeniexLLM for text-only models. Skipped when models are
not cached.
"""

from __future__ import annotations

import pytest

import geniex
from geniex import model_manager as _mm

VLM_MODEL = 'ggml-org/SmolVLM-500M-Instruct-GGUF'
LLM_MODEL = 'bartowski/Qwen_Qwen3-0.6B-GGUF'
LLM_QUANT = 'Q4_0'


@pytest.fixture(scope='module')
def vlm_cached(geniex_session):
    try:
        return _mm.get_paths(VLM_MODEL)
    except geniex.GeniexError as e:
        pytest.skip(f'VLM model {VLM_MODEL} not cached ({e}); run `geniex-py pull` first')


@pytest.fixture(scope='module')
def llm_cached(geniex_session):
    try:
        return _mm.get_paths(f'{LLM_MODEL}:{LLM_QUANT}')
    except geniex.GeniexError as e:
        pytest.skip(f'LLM model {LLM_MODEL} not cached ({e})')


def test_auto_model_returns_vlm_for_multimodal(vlm_cached):
    model = geniex.AutoModelForCausalLM.from_pretrained(VLM_MODEL)
    try:
        assert isinstance(model, geniex.GeniexVLM)
    finally:
        model.close()


def test_auto_model_returns_llm_for_text_only(llm_cached):
    model = geniex.AutoModelForCausalLM.from_pretrained(LLM_MODEL, quant=LLM_QUANT)
    try:
        assert isinstance(model, geniex.GeniexLLM)
    finally:
        model.close()
