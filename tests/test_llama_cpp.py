# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""llama_cpp plugin: LLM + VLM + precision + MTP."""

from __future__ import annotations

from pathlib import Path

import geniex
import pytest

from _models import (
    LLAMA_CPP_LLM_MODEL,
    LLAMA_CPP_LLM_PRECISION,
    LLAMA_CPP_MTP_TARGET_MODEL,
    LLAMA_CPP_VLM_MODEL,
    LLAMA_CPP_VLM_PRECISION,
)
from _quality_data import (
    LLM_QUALITY_MAX_NEW_TOKENS,
    LLM_QUALITY_PROMPTS,
    LLM_QUALITY_SEED,
    LLM_QUALITY_TEMPERATURE,
    VLM_QUALITY_KEYWORDS,
    VLM_QUALITY_MAX_NEW_TOKENS,
    VLM_QUALITY_PROMPT,
    VLM_QUALITY_SEED,
    VLM_QUALITY_TEMPERATURE,
)

_LLM_BACKENDS = ['cpu', 'gpu', 'npu']
_VLM_BACKENDS = ['cpu', 'gpu', 'npu']


def test_model_manager_pull(llama_cpp_llm_paths, llama_cpp_vlm_paths, llama_cpp_mtp_paths):
    for name, paths in [
        ('llm', llama_cpp_llm_paths),
        ('vlm', llama_cpp_vlm_paths),
        ('mtp-target', llama_cpp_mtp_paths['target']),
        ('mtp-draft', llama_cpp_mtp_paths['draft']),
    ]:
        assert Path(paths.model_path).is_file(), f'{name}: model_path missing: {paths.model_path}'


def _run_multi_turn(llm) -> list[str]:
    history: list[dict] = [{'role': 'system', 'content': 'Answer in one short sentence.'}]
    replies: list[str] = []
    for user in [
        'My name is Alice and I am 30 years old. Just acknowledge.',
        'What is my name?',
    ]:
        history.append({'role': 'user', 'content': user})
        prompt = llm.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        out = llm.generate(prompt, max_new_tokens=64, temperature=0.0, seed=42)
        assert out.text, f'empty completion at turn {len(replies) + 1}'
        assert out.profile.generated_tokens > 0
        replies.append(out.text)
        history.append({'role': 'assistant', 'content': out.text})
    return replies


@pytest.mark.llm
@pytest.mark.parametrize('device_map', _LLM_BACKENDS)
def test_llm_multi_turn(llama_cpp_llm_paths, device_map):
    with geniex.AutoModelForCausalLM.from_pretrained(
        LLAMA_CPP_LLM_MODEL,
        precision=LLAMA_CPP_LLM_PRECISION,
        device_map=device_map,
    ) as llm:
        replies = _run_multi_turn(llm)
    assert (
        'alice' in replies[-1].lower()
    ), f'device_map={device_map!r} expected reply to recall "Alice", got={replies[-1]!r}'


@pytest.mark.vlm
@pytest.mark.parametrize('device_map', _VLM_BACKENDS)
def test_vlm_multi_turn(llama_cpp_vlm_paths, test_image, device_map):
    with geniex.AutoModelForVision2Seq.from_pretrained(
        LLAMA_CPP_VLM_MODEL,
        precision=LLAMA_CPP_VLM_PRECISION,
        device_map=device_map,
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
        out1 = vlm.generate(prompt1, max_new_tokens=16, temperature=0.0, seed=42, images=[test_image])
        assert out1.profile.prompt_tokens > 0

        history.append({'role': 'assistant', 'content': out1.text or '...'})
        history.append({'role': 'user', 'content': [{'type': 'text', 'text': 'What color is it?'}]})
        prompt2 = vlm.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        # Second turn without a bitmap — old char-offset tracking sliced past the
        # image marker while an image was still supplied and mtmd_tokenize failed.
        out2 = vlm.generate(prompt2, max_new_tokens=16, temperature=0.0, seed=42, images=[])
        assert isinstance(out2, geniex.GenerateOutput)
        assert out2.profile.prompt_tokens > 0


def _vlm_prompt(vlm, image_path: str, text: str) -> str:
    return vlm.tokenizer.apply_chat_template(
        [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': text},
                    {'type': 'image', 'image': image_path},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


@pytest.mark.llm
@pytest.mark.parametrize('device_map', _LLM_BACKENDS)
@pytest.mark.parametrize(('prompt', 'expected'), LLM_QUALITY_PROMPTS)
def test_llm_quality_keywords(llama_cpp_llm_paths, device_map, prompt, expected):
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
        assert out.text, f'empty completion for prompt={prompt!r}'
        # Hoist to a bool so pytest's assertion introspection doesn't echo
        # out.text 4-5x per failure.
        matched = expected.lower() in out.text.lower()
        assert matched, (
            f'prompt={prompt!r} expected_substring={expected!r} ' f'device_map={device_map!r} got={out.text!r}'
        )


@pytest.mark.vlm
@pytest.mark.parametrize('device_map', _VLM_BACKENDS)
def test_vlm_quality_keywords(llama_cpp_vlm_paths, quality_image, device_map):
    with geniex.AutoModelForVision2Seq.from_pretrained(
        LLAMA_CPP_VLM_MODEL,
        precision=LLAMA_CPP_VLM_PRECISION,
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
        assert out.text, f'empty caption for device_map={device_map!r}'
        matched = any(kw in out.text.lower() for kw in VLM_QUALITY_KEYWORDS)
        assert matched, (
            f'caption did not match any expected keyword '
            f'device_map={device_map!r} keywords={VLM_QUALITY_KEYWORDS} '
            f'got={out.text!r}'
        )


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['npu'])
def test_mtp_multi_turn(llama_cpp_mtp_paths, device_map):
    # Local paths route through model_name= so the model-manager sees the
    # catalogue id; positional-arg would push the path into resolved_name and
    # trip the org/repo validator.
    with geniex.AutoModelForCausalLM.from_pretrained(
        llama_cpp_mtp_paths['target'].model_path,
        model_name=LLAMA_CPP_MTP_TARGET_MODEL,
        device_map=f'llama_cpp:{device_map}',
        spec_type='draft-mtp',
        spec_draft_model=llama_cpp_mtp_paths['draft'].model_path,
        spec_n_max=3,
    ) as llm:
        replies = _run_multi_turn(llm)
    assert (
        'alice' in replies[-1].lower()
    ), f'device_map={device_map!r} expected reply to recall "Alice", got={replies[-1]!r}'
