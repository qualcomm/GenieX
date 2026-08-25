# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""qairt plugin: LLM + VLM + precision. MTP is llama_cpp-only."""

from __future__ import annotations

import json
from pathlib import Path

import geniex
import pytest

from _models import QAIRT_LLM_MODEL, QAIRT_VLM_MODEL
from _quality_data import (
    LLM_QUALITY_MAX_NEW_TOKENS,
    LLM_QUALITY_PROMPTS,
    LLM_QUALITY_SEED,
    LLM_QUALITY_TEMPERATURE,
    PARITY_INPUT_IDS,
    PARITY_QAIRT_KL_MAX,
    VLM_QUALITY_KEYWORDS,
    VLM_QUALITY_MAX_NEW_TOKENS,
    VLM_QUALITY_PROMPT,
    VLM_QUALITY_SEED,
    VLM_QUALITY_TEMPERATURE,
    parity_kl_divergence,
    parity_top1_agreement,
)


def test_model_manager_pull(qairt_llm_paths, qairt_vlm_paths):
    for name, paths in [('llm', qairt_llm_paths), ('vlm', qairt_vlm_paths)]:
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
@pytest.mark.parametrize('device_map', ['npu'])
def test_llm_multi_turn(qairt_llm_paths, device_map):
    with geniex.AutoModelForCausalLM.from_pretrained(
        QAIRT_LLM_MODEL,
        device_map=device_map,
    ) as llm:
        replies = _run_multi_turn(llm)
    assert (
        'alice' in replies[-1].lower()
    ), f'device_map={device_map!r} expected reply to recall "Alice", got={replies[-1]!r}'


def _vlm_prompt(vlm, image_path: str, text: str) -> str:
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


@pytest.mark.vlm
def test_vlm_multi_turn(qairt_vlm_paths, test_image):
    # QAIRT re-encodes on every generate, so the follow-up turn must re-pass
    # the image; llama_cpp drops it on turn 2 instead.
    with geniex.AutoModelForVision2Seq.from_pretrained(
        QAIRT_VLM_MODEL,
        device_map='npu',
    ) as vlm:
        history = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': test_image},
                    {'type': 'text', 'text': 'Describe this image briefly.'},
                ],
            }
        ]
        prompt1 = vlm.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        out1 = vlm.generate(prompt1, max_new_tokens=16, temperature=0.0, seed=42, images=[test_image])
        assert out1.profile.prompt_tokens > 0

        history.append({'role': 'assistant', 'content': out1.text or '...'})
        history.append(
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': test_image},
                    {'type': 'text', 'text': 'What color is it?'},
                ],
            }
        )
        prompt2 = vlm.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        out2 = vlm.generate(prompt2, max_new_tokens=16, temperature=0.0, seed=42, images=[test_image])
        assert isinstance(out2, geniex.GenerateOutput)
        assert out2.profile.prompt_tokens > 0


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['npu'])
@pytest.mark.parametrize(('prompt', 'expected'), LLM_QUALITY_PROMPTS)
def test_llm_quality_keywords(qairt_llm_paths, device_map, prompt, expected):
    with geniex.AutoModelForCausalLM.from_pretrained(
        QAIRT_LLM_MODEL,
        device_map=device_map,
    ) as llm:
        formatted = llm.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        out = llm.generate(
            formatted,
            max_new_tokens=LLM_QUALITY_MAX_NEW_TOKENS,
            temperature=LLM_QUALITY_TEMPERATURE,
            seed=LLM_QUALITY_SEED,
        )
        assert out.text, f'empty completion for prompt={prompt!r}'
        # A think trace that eats the whole budget leaves a truncated answer
        # stub, which reads as a missing keyword — report it as truncation.
        assert (
            out.profile.stop_reason != 'length'
        ), f'completion truncated at {LLM_QUALITY_MAX_NEW_TOKENS} tokens: prompt={prompt!r} got={out.text!r}'
        matched = expected.lower() in out.text.lower()
        assert matched, (
            f'prompt={prompt!r} expected_substring={expected!r} ' f'device_map={device_map!r} got={out.text!r}'
        )


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['npu'])
def test_llm_logits_self_consistency(qairt_llm_paths, device_map):
    def _forward() -> tuple[list[list[tuple[int, float]]], list[float]]:
        with geniex.AutoModelForCausalLM.from_pretrained(
            QAIRT_LLM_MODEL,
            device_map=device_map,
        ) as llm:
            top1_rows = llm.forward_logits(PARITY_INPUT_IDS, all_positions=True, top_n=1)
            last_row = llm.forward_logits(PARITY_INPUT_IDS, all_positions=False, top_n=0)[0]
        return top1_rows, last_row

    ref_top1, ref_last = _forward()
    cand_top1, cand_last = _forward()
    agree = parity_top1_agreement(ref_top1, cand_top1)
    kl = parity_kl_divergence(ref_last, cand_last)
    assert agree == 1.0, f'device_map={device_map!r} top1={agree:.3f} != 1.0'
    assert kl <= PARITY_QAIRT_KL_MAX, f'device_map={device_map!r} KL={kl:.6f} > {PARITY_QAIRT_KL_MAX}'


@pytest.mark.vlm
@pytest.mark.parametrize('device_map', ['npu'])
def test_vlm_quality_keywords(qairt_vlm_paths, quality_image, device_map):
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
        assert out.text, f'empty caption for device_map={device_map!r}'
        matched = any(kw in out.text.lower() for kw in VLM_QUALITY_KEYWORDS)
        assert matched, (
            f'caption did not match any expected keyword '
            f'device_map={device_map!r} keywords={VLM_QUALITY_KEYWORDS} '
            f'got={out.text!r}'
        )


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['npu'])
def test_chat_template_roles_and_sentinels(qairt_llm_paths, device_map):
    with geniex.AutoModelForCausalLM.from_pretrained(
        QAIRT_LLM_MODEL,
        device_map=device_map,
    ) as llm:
        prompt = llm.tokenizer.apply_chat_template(
            [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'hi'},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    assert '<|im_start|>system' in prompt
    assert '<|im_start|>user' in prompt
    assert prompt.rstrip().endswith('<|im_start|>assistant')
    assert prompt.index('<|im_start|>system') < prompt.index('<|im_start|>user')


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['npu'])
def test_chat_template_enable_thinking(qairt_llm_paths, device_map):
    with geniex.AutoModelForCausalLM.from_pretrained(
        QAIRT_LLM_MODEL,
        device_map=device_map,
    ) as llm:
        msgs = [{'role': 'user', 'content': 'hi'}]
        with_think = llm.tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=True)
        without_think = llm.tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        default_think = llm.tokenizer.apply_chat_template(msgs, tokenize=False)
    assert (
        default_think == with_think
    ), 'default enable_thinking should auto-resolve to True on a thinking-capable model'
    assert with_think != without_think, f'enable_thinking flag did not reach the template: {with_think!r}'


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['npu'])
def test_chat_template_tools_list_and_json_string_equivalent(qairt_llm_paths, device_map):
    tool = {
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get current weather.',
            'parameters': {
                'type': 'object',
                'properties': {'city': {'type': 'string'}},
                'required': ['city'],
            },
        },
    }
    msgs = [{'role': 'user', 'content': "what's the weather in Paris?"}]
    with geniex.AutoModelForCausalLM.from_pretrained(
        QAIRT_LLM_MODEL,
        device_map=device_map,
    ) as llm:
        from_list = llm.tokenizer.apply_chat_template(msgs, tokenize=False, tools=[tool])
        from_str = llm.tokenizer.apply_chat_template(msgs, tokenize=False, tools=json.dumps([tool]))
    assert from_list == from_str, 'tools=list[dict] and tools=json.dumps(list[dict]) should render identically'
    assert 'get_weather' in from_list


# NOTE: no `test_chat_template_content_load_override` mirror here — the QAIRT plugin
# silently retains its baked ChatML template when `chat_template_content=` is passed
# to `from_pretrained`, while llama_cpp honours the override. Tracked as a plugin
# asymmetry to close separately; adding an xfail here would only clutter the suite.
