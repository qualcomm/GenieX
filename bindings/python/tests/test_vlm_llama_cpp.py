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

"""VLM end-to-end test against a small cached GGUF VLM (SmolVLM-500M).

Skipped automatically when the model is not already in the user cache —
the pull is too slow and too dependent on network conditions for a
regular test run. Populate the cache via ``geniex-py pull
ggml-org/SmolVLM-500M-Instruct-GGUF`` once.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import geniex
from geniex import model_manager as _mm

VLM_MODEL = 'ggml-org/SmolVLM-500M-Instruct-GGUF'

# Any small PNG in the repo works; we pick one that is checked in so the
# test doesn't depend on external downloads.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEST_IMAGE = _REPO_ROOT / 'cli' / 'server' / 'docs' / 'ui' / 'favicon-32x32.png'


@pytest.fixture(scope='module')
def vlm_paths(geniex_session):
    try:
        return _mm.get_paths(VLM_MODEL)
    except geniex.GeniexError as e:
        pytest.skip(f'VLM model {VLM_MODEL} not cached ({e}); run `geniex-py pull` first')


@pytest.fixture(scope='module')
def test_image():
    if not _TEST_IMAGE.is_file():
        pytest.skip(f'test image missing: {_TEST_IMAGE}')
    return str(_TEST_IMAGE)


@pytest.fixture(scope='module')
def vlm(vlm_paths):
    model = geniex.AutoModelForVision2Seq.from_pretrained(VLM_MODEL)
    yield model
    model.close()


def test_vlm_loads_as_geniex_vlm(vlm):
    assert isinstance(vlm, geniex.GeniexVLM)


def test_vlm_get_type_returns_vlm(vlm_paths):
    assert _mm.get_type(VLM_MODEL) == 'vlm'


def _vlm_prompt(vlm, image_path: str, text: str) -> str:
    # VLM templates expect typed content parts so the tokenizer emits the
    # right marker tokens for each modality.
    return vlm.tokenizer.apply_chat_template(
        [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image_path},
                    {'type': 'text', 'text': text},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def test_vlm_generate_with_image(vlm, test_image):
    prompt = _vlm_prompt(vlm, test_image, 'Describe this image briefly.')
    out = vlm.generate(
        prompt,
        max_new_tokens=16,
        temperature=0.0,
        seed=42,
        images=[test_image],
    )
    assert isinstance(out, geniex.GenerateOutput)
    assert out.text
    assert out.profile.generated_tokens > 0


def test_vlm_stream_with_image(vlm, test_image):
    prompt = _vlm_prompt(vlm, test_image, 'Describe this image.')
    streamer = vlm.generate(
        prompt,
        max_new_tokens=16,
        temperature=0.0,
        seed=42,
        images=[test_image],
        stream=True,
    )
    assert isinstance(streamer, geniex.TextIteratorStreamer)
    # Drain the streamer without asserting content — tiny VLM models can
    # hit EOS on the first token for a given seed/image, so we only check
    # that the stream terminates cleanly and publishes a GenerateOutput.
    list(streamer)
    assert streamer.output is not None
