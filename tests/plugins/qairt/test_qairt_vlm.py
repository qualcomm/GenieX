# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""qairt VLM matrix: npu (only supported backend for qairt)."""

from __future__ import annotations

import geniex
import pytest

from _models import QAIRT_VLM_MODEL
from _quality_data import (
    VLM_QUALITY_KEYWORDS,
    VLM_QUALITY_MAX_NEW_TOKENS,
    VLM_QUALITY_PROMPT,
    VLM_QUALITY_SEED,
    VLM_QUALITY_TEMPERATURE,
)

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


@pytest.mark.parametrize('device_map', ['npu'])
def test_generate_with_image(qairt_vlm_paths, test_image, device_map):
    with geniex.AutoModelForVision2Seq.from_pretrained(
        QAIRT_VLM_MODEL,
        device_map=device_map,
    ) as vlm:
        assert isinstance(vlm, geniex.GenieXVLM)
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


@pytest.mark.parametrize('device_map', ['npu'])
def test_quality_keywords(qairt_vlm_paths, quality_image, device_map):
    # Same keyword check as test_llama_cpp_vlm.test_quality_keywords; QAIRT VLM
    # is a much larger model so any miss here is a real regression.
    with geniex.AutoModelForVision2Seq.from_pretrained(
        QAIRT_VLM_MODEL,
        device_map=device_map,
    ) as vlm:
        prompt = _vlm_prompt(vlm, quality_image, VLM_QUALITY_PROMPT)
        out = vlm.generate(
            prompt,
            max_new_tokens=VLM_QUALITY_MAX_NEW_TOKENS,
            temperature=VLM_QUALITY_TEMPERATURE,
            seed=VLM_QUALITY_SEED,
            images=[quality_image],
        )
        assert isinstance(out, geniex.GenerateOutput)
        assert out.text, f'empty caption for device_map={device_map!r}'
        text = out.text.lower()
        # See test_llama_cpp_llm.test_quality_keywords for why this is hoisted.
        matched = any(kw in text for kw in VLM_QUALITY_KEYWORDS)
        assert matched, (
            f'caption did not match any expected keyword '
            f'device_map={device_map!r} keywords={VLM_QUALITY_KEYWORDS} '
            f'got={out.text!r}'
        )
