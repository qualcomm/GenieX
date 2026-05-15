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
models and GeniexLLM for text-only models, across both llama_cpp and
qairt plugins. Skipped when models are not cached.
"""

from __future__ import annotations

import pytest

import geniex
from geniex import model_manager as _mm

from .conftest import QAIRT_MODEL, QAIRT_VLM_MODEL

LLAMA_CPP_VLM_MODEL = 'ggml-org/SmolVLM-500M-Instruct-GGUF'
LLAMA_CPP_LLM_MODEL = 'bartowski/Qwen_Qwen3-0.6B-GGUF'
LLAMA_CPP_LLM_QUANT = 'Q4_0'


@pytest.fixture(scope='module')
def llama_cpp_vlm_cached(geniex_session):
    try:
        return _mm.get_paths(LLAMA_CPP_VLM_MODEL)
    except geniex.GeniexError as e:
        pytest.skip(f'VLM {LLAMA_CPP_VLM_MODEL} not cached ({e}); run `geniex-py pull` first')


@pytest.fixture(scope='module')
def llama_cpp_llm_cached(geniex_session):
    try:
        return _mm.get_paths(f'{LLAMA_CPP_LLM_MODEL}:{LLAMA_CPP_LLM_QUANT}')
    except geniex.GeniexError as e:
        pytest.skip(f'LLM {LLAMA_CPP_LLM_MODEL} not cached ({e})')


def test_auto_model_returns_vlm_for_llama_cpp_multimodal(llama_cpp_vlm_cached):
    model = geniex.AutoModelForCausalLM.from_pretrained(LLAMA_CPP_VLM_MODEL)
    try:
        assert isinstance(model, geniex.GeniexVLM)
    finally:
        model.close()


def test_auto_model_returns_llm_for_llama_cpp_text_only(llama_cpp_llm_cached):
    model = geniex.AutoModelForCausalLM.from_pretrained(LLAMA_CPP_LLM_MODEL, quant=LLAMA_CPP_LLM_QUANT)
    try:
        assert isinstance(model, geniex.GeniexLLM)
    finally:
        model.close()


def test_auto_model_returns_vlm_for_qairt_multimodal(qairt_vlm_paths):
    model = geniex.AutoModelForCausalLM.from_pretrained(QAIRT_VLM_MODEL)
    try:
        assert isinstance(model, geniex.GeniexVLM)
    finally:
        model.close()


def test_auto_model_returns_llm_for_qairt_text_only(qairt_paths):
    model = geniex.AutoModelForCausalLM.from_pretrained(QAIRT_MODEL)
    try:
        assert isinstance(model, geniex.GeniexLLM)
    finally:
        model.close()
