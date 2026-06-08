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

"""llama_cpp VLM matrix: cpu, npu, hybrid."""

from __future__ import annotations

import geniex
import pytest

from _models import LLAMA_CPP_VLM_MODEL

pytestmark = pytest.mark.vlm


def _vlm_prompt(vlm: geniex.GenieXVLM, image_path: str, text: str) -> str:
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


@pytest.mark.parametrize('device_map', ['cpu', 'npu', 'hybrid'])
def test_generate_with_image(llama_cpp_vlm_paths, test_image, device_map):
    with geniex.AutoModelForVision2Seq.from_pretrained(
        LLAMA_CPP_VLM_MODEL,
        device_map=device_map,
    ) as vlm:
        assert isinstance(vlm, geniex.GenieXVLM)
        prompt = _vlm_prompt(vlm, test_image, 'Describe this image.')
        out = vlm.generate(
            prompt,
            max_new_tokens=8,
            temperature=0.0,
            seed=42,
            images=[test_image],
        )
        assert isinstance(out, geniex.GenerateOutput)
        # Tiny VLMs can hit EOS on the first token, so don't assert text is
        # non-empty; the profile object proves a generation step ran.
        assert out.profile is not None


def test_multi_turn_without_reset(llama_cpp_vlm_paths, test_image):
    # The caller re-templates the whole conversation each turn and never calls
    # reset(); the second turn must still decode against a token-accurate n_past.
    # With the old character-offset tracking the second turn diverged.
    with geniex.AutoModelForVision2Seq.from_pretrained(
        LLAMA_CPP_VLM_MODEL,
        device_map='cpu',
    ) as vlm:
        history = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': test_image},
                    {'type': 'text', 'text': 'Describe this image.'},
                ],
            }
        ]
        prompt1 = vlm.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        out1 = vlm.generate(prompt1, max_new_tokens=8, temperature=0.0, seed=42, images=[test_image])
        assert out1.profile.prompt_tokens > 0

        history.append({'role': 'assistant', 'content': out1.text or '...'})
        history.append({'role': 'user', 'content': [{'type': 'text', 'text': 'What color is it?'}]})
        prompt2 = vlm.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        # Old char-offset tracking sliced prompt2 past the image marker while a
        # bitmap was still supplied, so mtmd_tokenize failed and generate() raised.
        out2 = vlm.generate(prompt2, max_new_tokens=8, temperature=0.0, seed=42, images=[test_image])
        assert isinstance(out2, geniex.GenerateOutput)
        assert out2.profile.prompt_tokens > 0
