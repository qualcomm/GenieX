# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""qairt plugin: LLM + VLM + precision. MTP is llama_cpp-only."""

from __future__ import annotations

import json
from pathlib import Path

import geniex
import pytest
from geniex._ffi._api import GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED

from _models import matrix, primary, pull_cells
from _quality_data import (
    DETERMINISM_MAX_NEW_TOKENS,
    DETERMINISM_PROMPT,
    GREEDY_TEMPERATURE,
    LLM_QUALITY_MAX_NEW_TOKENS,
    LLM_QUALITY_PROMPTS,
    LLM_QUALITY_SEED,
    PARITY_INPUT_IDS,
    PARITY_QAIRT_KL_MAX,
    VLM_QUALITY_KEYWORDS,
    VLM_QUALITY_MAX_NEW_TOKENS,
    VLM_QUALITY_PROMPT,
    VLM_QUALITY_SEED,
    parity_kl_divergence,
    parity_top1_agreement,
)

_LLM = primary('qairt_llm')
_LLM_MATRIX = matrix('qairt_llm')
_VLM_MATRIX = matrix('qairt_vlm')


@pytest.mark.parametrize('model', pull_cells('qairt_llm', 'qairt_vlm'))
def test_model_manager_pull(cached, model):
    paths = cached(model)
    assert Path(paths.model_path).is_file(), f'{model.id}: model_path missing: {paths.model_path}'


def _run_multi_turn(llm) -> list[str]:
    history: list[dict] = [{'role': 'system', 'content': 'Answer in one short sentence.'}]
    replies: list[str] = []
    for user in [
        'My name is Alice and I am 30 years old. Remember it and reply with just OK.',
        'What is my name?',
    ]:
        history.append({'role': 'user', 'content': user})
        prompt = llm.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        out = llm.generate(prompt, max_new_tokens=64, temperature=GREEDY_TEMPERATURE, seed=42)
        assert out.text, f'empty completion at turn {len(replies) + 1}'
        assert out.profile.generated_tokens > 0
        replies.append(out.text)
        history.append({'role': 'assistant', 'content': out.text})
    return replies


def _quality_misses(llm, budget: int) -> list[str]:
    misses = []
    for prompt, expected in LLM_QUALITY_PROMPTS:
        # Plugins carry KV / chat state across generate(); each prompt is its own conversation.
        llm.reset()
        formatted = llm.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        out = llm.generate(formatted, max_new_tokens=budget, temperature=GREEDY_TEMPERATURE, seed=LLM_QUALITY_SEED)
        # Greedy decode can loop until the budget runs out, so `length` is only a
        # problem when it truncated before the keyword appeared.
        if expected.lower() not in out.text.lower():
            misses.append(f'{prompt!r} expected={expected!r} stop_reason={out.profile.stop_reason!r} got={out.text!r}')
    return misses


@pytest.mark.llm
@pytest.mark.parametrize(('model', 'device_map'), _LLM_MATRIX)
def test_llm_generate(cached, model, device_map):
    cached(model)
    budget = model.quality_max_new_tokens or LLM_QUALITY_MAX_NEW_TOKENS
    with geniex.AutoModelForCausalLM.from_pretrained(
        model.id,
        device_map=device_map,
    ) as llm:
        replies = _run_multi_turn(llm)
        misses = _quality_misses(llm, budget)
    assert (
        'alice' in replies[-1].lower()
    ), f'model={model.id!r} device_map={device_map!r} expected reply to recall "Alice", got={replies[-1]!r}'
    assert not misses, f'model={model.id!r} device_map={device_map!r} budget={budget} misses: ' + '; '.join(misses)


@pytest.mark.llm
@pytest.mark.parametrize(('model', 'device_map'), _LLM_MATRIX)
def test_llm_greedy_is_deterministic(cached, model, device_map):
    # Separate handles: decode text and logits must match across two fresh loads.
    cached(model)

    def _run_once(seed: int) -> tuple[str, list[list[tuple[int, float]]], list[float]]:
        with geniex.AutoModelForCausalLM.from_pretrained(
            model.id,
            device_map=device_map,
        ) as llm:
            formatted = llm.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': DETERMINISM_PROMPT}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            text = llm.generate(
                formatted,
                max_new_tokens=DETERMINISM_MAX_NEW_TOKENS,
                temperature=GREEDY_TEMPERATURE,
                seed=seed,
            ).text
            top1_rows = llm.forward_logits(PARITY_INPUT_IDS, all_positions=True, top_n=1)
            last_row = llm.forward_logits(PARITY_INPUT_IDS, all_positions=False, top_n=0)[0]
        return text, top1_rows, last_row

    first, ref_top1, ref_last = _run_once(LLM_QUALITY_SEED)
    second, cand_top1, cand_last = _run_once(LLM_QUALITY_SEED + 1)
    assert first, f'empty completion for model={model.id!r} device_map={device_map!r}'
    assert first == second, (
        f'greedy decode diverged across seeds '
        f'model={model.id!r} device_map={device_map!r} first={first!r} second={second!r}'
    )
    agree = parity_top1_agreement(ref_top1, cand_top1)
    kl = parity_kl_divergence(ref_last, cand_last)
    assert agree == 1.0, f'device_map={device_map!r} top1={agree:.3f} != 1.0'
    assert kl <= PARITY_QAIRT_KL_MAX, f'device_map={device_map!r} KL={kl:.6f} > {PARITY_QAIRT_KL_MAX}'


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
@pytest.mark.parametrize(('model', 'device_map'), _VLM_MATRIX)
def test_vlm_generate(cached, model, test_image, quality_image, device_map):
    # QAIRT re-encodes on every generate, so the follow-up turn must re-pass
    # the image; llama_cpp drops it on turn 2 instead.
    cached(model)
    with geniex.AutoModelForVision2Seq.from_pretrained(
        model.id,
        device_map=device_map,
    ) as vlm:
        caption = vlm.generate(
            _vlm_prompt(vlm, quality_image, VLM_QUALITY_PROMPT),
            max_new_tokens=model.quality_max_new_tokens or VLM_QUALITY_MAX_NEW_TOKENS,
            temperature=GREEDY_TEMPERATURE,
            seed=VLM_QUALITY_SEED,
            images=[quality_image],
        )

        vlm.reset()
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
        out1 = vlm.generate(prompt1, max_new_tokens=16, temperature=GREEDY_TEMPERATURE, seed=42, images=[test_image])
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
        out2 = vlm.generate(prompt2, max_new_tokens=16, temperature=GREEDY_TEMPERATURE, seed=42, images=[test_image])

    assert caption.text, f'empty caption for model={model.id!r} device_map={device_map!r}'
    matched = any(kw in caption.text.lower() for kw in VLM_QUALITY_KEYWORDS)
    assert matched, (
        f'caption did not match any expected keyword '
        f'device_map={device_map!r} keywords={VLM_QUALITY_KEYWORDS} '
        f'got={caption.text!r}'
    )
    assert out1.profile.prompt_tokens > 0
    assert isinstance(out2, geniex.GenerateOutput)
    assert out2.profile.prompt_tokens > 0


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['npu'])
def test_llm_generate_input_embd_not_supported(qairt_llm_paths, device_map):
    with geniex.AutoModelForCausalLM.from_pretrained(
        _LLM.id,
        device_map=device_map,
    ) as llm:
        assert llm.embd_dim == 0, 'qairt should not report an embd_dim'
        with pytest.raises(geniex.GenieXError) as excinfo:
            llm.generate(input_embd=[0.0, 0.0], input_embd_dim=2, max_new_tokens=1)
        assert excinfo.value.code == GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['npu'])
def test_llm_forward_logits_prompt(qairt_llm_paths, device_map):
    prompt = 'The capital of France is'
    with geniex.AutoModelForCausalLM.from_pretrained(
        _LLM.id,
        device_map=device_map,
    ) as llm:
        n_prompt = llm.generate(prompt, max_new_tokens=1, temperature=GREEDY_TEMPERATURE, seed=0).profile.prompt_tokens
        rows = llm.forward_logits(prompt=prompt, all_positions=True, top_n=1)
        last = llm.forward_logits(prompt=prompt)
    assert len(rows) == n_prompt
    assert len(last) == 1
    assert max(range(len(last[0])), key=last[0].__getitem__) == rows[-1][0][0]


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['npu'])
def test_chat_template(qairt_llm_paths, device_map):
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
    hi = [{'role': 'user', 'content': 'hi'}]
    asked = [{'role': 'user', 'content': "what's the weather in Paris?"}]
    with geniex.AutoModelForCausalLM.from_pretrained(
        _LLM.id,
        device_map=device_map,
    ) as llm:
        render = llm.tokenizer.apply_chat_template
        roles = render(
            [{'role': 'system', 'content': 'You are a helpful assistant.'}, *hi],
            tokenize=False,
            add_generation_prompt=True,
        )
        with_think = render(hi, tokenize=False, enable_thinking=True)
        without_think = render(hi, tokenize=False, enable_thinking=False)
        default_think = render(hi, tokenize=False)
        from_list = render(asked, tokenize=False, tools=[tool])
        from_str = render(asked, tokenize=False, tools=json.dumps([tool]))

    assert '<|im_start|>system' in roles
    assert '<|im_start|>user' in roles
    assert roles.rstrip().endswith('<|im_start|>assistant')
    assert roles.index('<|im_start|>system') < roles.index('<|im_start|>user')
    assert (
        default_think == with_think
    ), 'default enable_thinking should auto-resolve to True on a thinking-capable model'
    assert with_think != without_think, f'enable_thinking flag did not reach the template: {with_think!r}'
    assert from_list == from_str, 'tools=list[dict] and tools=json.dumps(list[dict]) should render identically'
    assert 'get_weather' in from_list


# NOTE: no `test_chat_template_content_load_override` mirror here — the QAIRT plugin
# silently retains its baked ChatML template when `chat_template_content=` is passed
# to `from_pretrained`, while llama_cpp honours the override. Tracked as a plugin
# asymmetry to close separately; adding an xfail here would only clutter the suite.
