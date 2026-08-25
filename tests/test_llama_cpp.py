# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""llama_cpp plugin: LLM + VLM + precision + MTP."""

from __future__ import annotations

import json
import platform
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
    PARITY_INPUT_IDS,
    PARITY_KL_MAX,
    PARITY_TOP1_MIN,
    VLM_QUALITY_KEYWORDS,
    VLM_QUALITY_MAX_NEW_TOKENS,
    VLM_QUALITY_PROMPT,
    VLM_QUALITY_SEED,
    VLM_QUALITY_TEMPERATURE,
    parity_kl_divergence,
    parity_top1_agreement,
)

_LLM_BACKENDS = ['cpu', 'gpu', 'npu']
_VLM_BACKENDS = ['cpu', 'gpu', 'npu']
_PARITY_CANDIDATES = ['gpu', 'npu', 'hybrid']
_IS_QCS9075M = platform.system() == 'Linux' and platform.machine().lower() in ('aarch64', 'arm64')


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


def _forward_parity(device_map: str) -> tuple[list[list[tuple[int, float]]], list[float]]:
    with geniex.AutoModelForCausalLM.from_pretrained(
        LLAMA_CPP_LLM_MODEL,
        precision=LLAMA_CPP_LLM_PRECISION,
        device_map=device_map,
    ) as llm:
        top1_rows = llm.forward_logits(PARITY_INPUT_IDS, all_positions=True, top_n=1)
        last_row = llm.forward_logits(PARITY_INPUT_IDS, all_positions=False, top_n=0)[0]
    return top1_rows, last_row


@pytest.mark.llm
@pytest.mark.parametrize('device_map', _PARITY_CANDIDATES)
def test_llm_logits_parity(llama_cpp_llm_paths, device_map):
    ref_top1, ref_last = _forward_parity('cpu')
    cand_top1, cand_last = _forward_parity(device_map)
    agree = parity_top1_agreement(ref_top1, cand_top1)
    kl = parity_kl_divergence(ref_last, cand_last)
    assert agree >= PARITY_TOP1_MIN, f'device_map={device_map!r} top1={agree:.3f} < {PARITY_TOP1_MIN}'
    assert kl <= PARITY_KL_MAX, f'device_map={device_map!r} KL={kl:.4f} > {PARITY_KL_MAX}'


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['npu'])
@pytest.mark.skipif(_IS_QCS9075M, reason='draft-mtp not yet supported upstream on QCS9075M / HTP')
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


_MEDIA_MARKER = '<__media__>'


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['cpu'])
def test_chat_template_roles_and_sentinels(llama_cpp_llm_paths, device_map):
    with geniex.AutoModelForCausalLM.from_pretrained(
        LLAMA_CPP_LLM_MODEL,
        precision=LLAMA_CPP_LLM_PRECISION,
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
@pytest.mark.parametrize('device_map', ['cpu'])
def test_chat_template_enable_thinking(llama_cpp_llm_paths, device_map):
    with geniex.AutoModelForCausalLM.from_pretrained(
        LLAMA_CPP_LLM_MODEL,
        precision=LLAMA_CPP_LLM_PRECISION,
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
@pytest.mark.parametrize('device_map', ['cpu'])
def test_chat_template_tools_list_and_json_string_equivalent(llama_cpp_llm_paths, device_map):
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
        LLAMA_CPP_LLM_MODEL,
        precision=LLAMA_CPP_LLM_PRECISION,
        device_map=device_map,
    ) as llm:
        from_list = llm.tokenizer.apply_chat_template(msgs, tokenize=False, tools=[tool])
        from_str = llm.tokenizer.apply_chat_template(msgs, tokenize=False, tools=json.dumps([tool]))
    assert from_list == from_str, 'tools=list[dict] and tools=json.dumps(list[dict]) should render identically'
    assert 'get_weather' in from_list


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['cpu'])
def test_chat_template_content_load_override(llama_cpp_llm_paths, device_map):
    jinja = (
        '{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}'
        '{% if add_generation_prompt %}[assistant] {% endif %}'
    )
    with geniex.AutoModelForCausalLM.from_pretrained(
        LLAMA_CPP_LLM_MODEL,
        precision=LLAMA_CPP_LLM_PRECISION,
        device_map=device_map,
        chat_template_content=jinja,
    ) as llm:
        prompt = llm.tokenizer.apply_chat_template(
            [
                {'role': 'system', 'content': 'be terse'},
                {'role': 'user', 'content': 'hello'},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    assert prompt == '[system] be terse\n[user] hello\n[assistant] '


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['cpu'])
def test_context_shift_fires_on_decode_overflow(llama_cpp_llm_paths, device_map):
    # n_ctx small enough that prompt (~200 tokens) + max_new_tokens exceeds it,
    # forcing slide_window to rotate during decode. Prompt tokens are anchored
    # by n_keep=4 and the rest of the past is compacted to make room.
    n_ctx = 256
    prompt_body = ' '.join(['banana'] * 200)
    prompt = f'Repeat the following words verbatim: {prompt_body}\n\nAnswer: '

    with geniex.AutoModelForCausalLM.from_pretrained(
        LLAMA_CPP_LLM_MODEL,
        precision=LLAMA_CPP_LLM_PRECISION,
        device_map=device_map,
        n_ctx=n_ctx,
    ) as llm:
        out = llm.generate(prompt, max_new_tokens=120, temperature=0.0, seed=0)

    assert out.profile.stop_reason != 'context_length', f'context shift did not save the run: {out.profile}'
    assert out.profile.generated_tokens > 0
    assert out.profile.prompt_tokens + out.profile.generated_tokens > n_ctx


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['cpu'])
def test_context_shift_not_triggered_when_nctx_is_ample(llama_cpp_llm_paths, device_map):
    with geniex.AutoModelForCausalLM.from_pretrained(
        LLAMA_CPP_LLM_MODEL,
        precision=LLAMA_CPP_LLM_PRECISION,
        device_map=device_map,
        n_ctx=2048,
    ) as llm:
        out = llm.generate('Say hi in one word.', max_new_tokens=8, temperature=0.0, seed=0)

    assert out.profile.stop_reason in {'eos', 'length', 'completed'}, out.profile
    assert out.profile.generated_tokens > 0


@pytest.mark.vlm
@pytest.mark.parametrize('device_map', ['cpu'])
@pytest.mark.parametrize('n_images', [0, 1, 2])
def test_mtmd_marker_prepended_per_image(llama_cpp_vlm_paths, test_image, quality_image, device_map, n_images):
    content: list[dict] = [
        {'type': 'image', 'image': test_image},
        {'type': 'image', 'image': quality_image},
    ][:n_images]
    content.append({'type': 'text', 'text': 'describe the picture'})

    with geniex.AutoModelForVision2Seq.from_pretrained(
        LLAMA_CPP_VLM_MODEL,
        precision=LLAMA_CPP_VLM_PRECISION,
        device_map=device_map,
    ) as vlm:
        prompt = vlm.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    assert prompt.count(_MEDIA_MARKER) == n_images
    assert 'describe the picture' in prompt


@pytest.mark.vlm
@pytest.mark.parametrize('device_map', ['cpu'])
def test_mtmd_marker_absent_on_text_only_message(llama_cpp_vlm_paths, device_map):
    with geniex.AutoModelForVision2Seq.from_pretrained(
        LLAMA_CPP_VLM_MODEL,
        precision=LLAMA_CPP_VLM_PRECISION,
        device_map=device_map,
    ) as vlm:
        prompt = vlm.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'plain text, no image'}],
            tokenize=False,
            add_generation_prompt=True,
        )
    assert _MEDIA_MARKER not in prompt
    assert 'plain text, no image' in prompt


@pytest.mark.vlm
@pytest.mark.parametrize('device_map', ['cpu'])
def test_mtmd_generate_rejects_missing_image_path(llama_cpp_vlm_paths, test_image, device_map):
    # apply_chat_template records that content referenced an image, then
    # generate() validates the images=[...] list against the filesystem.
    missing = str(Path(test_image).parent / 'does-not-exist.png')
    with geniex.AutoModelForVision2Seq.from_pretrained(
        LLAMA_CPP_VLM_MODEL,
        precision=LLAMA_CPP_VLM_PRECISION,
        device_map=device_map,
    ) as vlm:
        vlm.tokenizer.apply_chat_template(
            [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': test_image},
                        {'type': 'text', 'text': 'hi'},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        with pytest.raises(FileNotFoundError):
            vlm.generate('irrelevant', images=[missing])
