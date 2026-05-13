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

"""QAIRT VLM end-to-end test. Skips automatically when not on Snapdragon or
when ``GENIEX_DEVICE_TEST=1`` is not set."""

from __future__ import annotations

from pathlib import Path

import pytest

import geniex
from geniex import model_manager as _mm

from .conftest import QAIRT_VLM_MODEL

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEST_IMAGE = _REPO_ROOT / 'cli' / 'server' / 'docs' / 'ui' / 'favicon-32x32.png'


@pytest.fixture(scope='module')
def test_image():
    if not _TEST_IMAGE.is_file():
        pytest.skip(f'test image missing: {_TEST_IMAGE}')
    return str(_TEST_IMAGE)


@pytest.fixture(scope='module')
def qairt_vlm(qairt_vlm_paths):
    model = geniex.AutoModelForVision2Seq.from_pretrained(QAIRT_VLM_MODEL)
    yield model
    model.close()


def test_qairt_vlm_loads(qairt_vlm):
    assert isinstance(qairt_vlm, geniex.GeniexVLM)


def test_qairt_vlm_get_type_is_vlm(qairt_vlm_paths):
    assert _mm.get_type(QAIRT_VLM_MODEL) == 'vlm'


def test_qairt_vlm_generate_with_image(qairt_vlm, test_image):
    prompt = qairt_vlm.tokenizer.apply_chat_template(
        [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': test_image},
                    {'type': 'text', 'text': 'Describe this image briefly.'},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    out = qairt_vlm.generate(
        prompt,
        max_new_tokens=16,
        temperature=0.0,
        seed=42,
        images=[test_image],
    )
    assert isinstance(out, geniex.GenerateOutput)
    assert out.text
    assert out.profile.generated_tokens > 0
