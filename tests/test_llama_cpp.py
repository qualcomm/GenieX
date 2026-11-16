# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""llama_cpp plugin: LLM + VLM + precision + MTP."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import re
from pathlib import Path

import geniex
import pytest
from geniex._ffi._api import GENIEX_ERROR_COMMON_INVALID_INPUT

from _models import matrix, primary, pull_cells
from _quality_data import (
    DETERMINISM_MAX_NEW_TOKENS,
    DETERMINISM_PROMPT,
    GREEDY_TEMPERATURE,
    LLM_QUALITY_MAX_NEW_TOKENS,
    LLM_QUALITY_PROMPTS,
    LLM_QUALITY_SEED,
    PARITY_INPUT_IDS,
    PARITY_KL_MAX,
    PARITY_TOP1_MIN,
    VLM_QUALITY_KEYWORDS,
    VLM_QUALITY_MAX_NEW_TOKENS,
    VLM_QUALITY_PROMPT,
    VLM_QUALITY_SEED,
    parity_kl_divergence,
    parity_top1_agreement,
)

GENIEX_ERROR_VLM_IMAGE_LOAD = -201001

_LLM = primary('llama_cpp_llm')
_VLM = primary('llama_cpp_vlm')
_MTP_TARGET = primary('llama_cpp_mtp_target')
_MTP_QWEN35 = primary('llama_cpp_mtp_qwen35')
_LLM_MATRIX = matrix('llama_cpp_llm')
_VLM_MATRIX = matrix('llama_cpp_vlm')
# Scored against a cpu reference run of the primary model, so not matrix-driven.
_PARITY_CANDIDATES = ['gpu', 'npu', 'hybrid']
_IS_QCS9075M = platform.system() == 'Linux' and platform.machine().lower() in ('aarch64', 'arm64')


@pytest.mark.parametrize('model', pull_cells('llama_cpp_llm', 'llama_cpp_vlm'))
def test_model_manager_pull(cached, model):
    paths = cached(model)
    assert Path(paths.model_path).is_file(), f'{model.id}: model_path missing: {paths.model_path}'


@pytest.mark.nightly
def test_model_manager_pull_mtp(llama_cpp_mtp_paths):
    for name, paths in llama_cpp_mtp_paths.items():
        assert Path(paths.model_path).is_file(), f'mtp-{name}: model_path missing: {paths.model_path}'


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
        out = llm.generate(prompt, max_new_tokens=64, temperature=GREEDY_TEMPERATURE, seed=42)
        assert out.text, f'empty completion at turn {len(replies) + 1}'
        assert out.profile.generated_tokens > 0
        replies.append(out.text)
        history.append({'role': 'assistant', 'content': out.text})
    return replies


def _chat_prompt(llm, content: str) -> str:
    return llm.tokenizer.apply_chat_template(
        [
            {'role': 'system', 'content': 'Answer in one short sentence.'},
            {'role': 'user', 'content': content},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@pytest.mark.llm
@pytest.mark.parametrize('device_map', _LLM.devices)
def test_llm_prefix_cache(cached, device_map):
    cached(_LLM)
    shared = 'Keep the label ORCHID. Branch details: '
    first = shared + ('amber ' * 80) + 'Say hello briefly.'
    second = shared + ('cobalt ' * 32) + 'Say hello briefly.'
    third = second + ' Add one note.'

    with geniex.AutoModelForCausalLM.from_pretrained(
        _LLM.id,
        precision=_LLM.precision,
        device_map=device_map,
    ) as llm:
        repeated = _chat_prompt(llm, 'Say hello briefly.')
        same = [
            llm.generate(repeated, max_new_tokens=16, temperature=GREEDY_TEMPERATURE, seed=LLM_QUALITY_SEED)
            for _ in range(3)
        ]
        llm.reset()
        rollback = [
            llm.generate(
                _chat_prompt(llm, content), max_new_tokens=8, temperature=GREEDY_TEMPERATURE, seed=LLM_QUALITY_SEED
            )
            for content in (first, second, third)
        ]

    for index, out in enumerate(same + rollback, start=1):
        assert out.text, f'empty completion on call {index}'
        assert out.profile.generated_tokens > 0
    prompt_tokens = same[0].profile.prompt_tokens
    assert prompt_tokens > 1, f'expected a full initial prefill, got {prompt_tokens} tokens'
    # Reusing the exact prompt must re-evaluate its last token for fresh logits.
    assert same[1].profile.prompt_tokens == 1
    assert same[2].profile.prompt_tokens == 1
    # The reused prefix still counts toward the full prompt.
    assert all(out.profile.total_prompt_tokens == prompt_tokens for out in same)
    first_tokens, second_tokens, third_tokens = (out.profile.prompt_tokens for out in rollback)
    assert first_tokens > second_tokens > third_tokens > 0, (
        'expected full prefill, partial rollback, then reuse of the second prompt prefix; '
        f'prompt_tokens={first_tokens, second_tokens, third_tokens}'
    )


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['cpu'])
def test_llm_prefix_cache_fallback(geniex_session, device_map):
    with geniex.AutoModelForCausalLM.from_pretrained(
        'LiquidAI/LFM3-1.7B-GGUF',
        precision='Q4_0',
        device_map=device_map,
    ) as llm:
        prompts = [_chat_prompt(llm, content) for content in ('Say hello briefly.', 'Say goodbye briefly.', 'Hello.')]

        def generate(prompt):
            return llm.generate(prompt, max_new_tokens=8, temperature=GREEDY_TEMPERATURE, seed=LLM_QUALITY_SEED)

        generate(prompts[0])
        for prompt in prompts:
            reused = generate(prompt)
            llm.reset()
            fresh = generate(prompt)
            assert reused.text == fresh.text
            assert reused.profile.prompt_tokens == fresh.profile.prompt_tokens > 1


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
        precision=model.precision,
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
    # Separate handles keep this seed comparison independent of cache reuse.
    cached(model)

    def _greedy_once(seed: int) -> str:
        with geniex.AutoModelForCausalLM.from_pretrained(
            model.id,
            precision=model.precision,
            device_map=device_map,
        ) as llm:
            formatted = llm.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': DETERMINISM_PROMPT}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return llm.generate(
                formatted,
                max_new_tokens=DETERMINISM_MAX_NEW_TOKENS,
                temperature=GREEDY_TEMPERATURE,
                seed=seed,
            ).text

    first = _greedy_once(LLM_QUALITY_SEED)
    second = _greedy_once(LLM_QUALITY_SEED + 1)
    assert first, f'empty completion for model={model.id!r} device_map={device_map!r}'
    assert first == second, (
        f'greedy decode diverged across seeds '
        f'model={model.id!r} device_map={device_map!r} first={first!r} second={second!r}'
    )


@pytest.mark.vlm
@pytest.mark.parametrize(('model', 'device_map'), _VLM_MATRIX)
def test_vlm_generate(cached, model, test_image, quality_image, device_map):
    cached(model)
    with geniex.AutoModelForVision2Seq.from_pretrained(
        model.id,
        precision=model.precision,
        device_map=device_map,
    ) as vlm:
        caption = vlm.generate(
            _vlm_prompt(vlm, quality_image, VLM_QUALITY_PROMPT),
            max_new_tokens=model.quality_max_new_tokens or VLM_QUALITY_MAX_NEW_TOKENS,
            temperature=GREEDY_TEMPERATURE,
            seed=VLM_QUALITY_SEED,
            images=[quality_image],
        )

        # llama_cpp VLM prompts are append-only; a new conversation needs a reset.
        vlm.reset()
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
        out1 = vlm.generate(prompt1, max_new_tokens=16, temperature=GREEDY_TEMPERATURE, seed=42, images=[test_image])
        history.append({'role': 'assistant', 'content': out1.text or '...'})
        history.append({'role': 'user', 'content': [{'type': 'text', 'text': 'What color is it?'}]})
        prompt2 = vlm.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        # Second turn without a bitmap — old char-offset tracking sliced past the
        # image marker while an image was still supplied and mtmd_tokenize failed.
        out2 = vlm.generate(prompt2, max_new_tokens=16, temperature=GREEDY_TEMPERATURE, seed=42, images=[])

    assert caption.text, f'empty caption for model={model.id!r} device_map={device_map!r}'
    matched = any(kw in caption.text.lower() for kw in VLM_QUALITY_KEYWORDS)
    assert matched, (
        f'caption did not match any expected keyword '
        f'model={model.id!r} device_map={device_map!r} keywords={VLM_QUALITY_KEYWORDS} '
        f'got={caption.text!r}'
    )
    assert out1.profile.prompt_tokens > 0
    assert isinstance(out2, geniex.GenerateOutput)
    assert out2.profile.prompt_tokens > 0
    # Turn 2 reuses turn 1's prompt and reply, so only the increment is evaluated.
    assert out2.profile.total_prompt_tokens == (
        out1.profile.total_prompt_tokens + out1.profile.generated_tokens + out2.profile.prompt_tokens
    )


@pytest.mark.vlm
@pytest.mark.parametrize('device_map', ['cpu'])
def test_vlm_stream_cancel_matches_final_output(llama_cpp_vlm_paths, test_image, device_map):
    paths = llama_cpp_vlm_paths
    with geniex.AutoModelForVision2Seq.from_pretrained(
        paths.model_path,
        model_name=_VLM.id,
        device_map=device_map,
        mmproj_path=paths.mmproj_path,
    ) as vlm:
        prompt = _vlm_prompt(vlm, test_image, 'Describe this image in detail.')
        streamer = vlm.generate(
            prompt,
            max_new_tokens=128,
            temperature=GREEDY_TEMPERATURE,
            seed=42,
            images=[test_image],
            stream=True,
        )
        stream = iter(streamer)
        chunks = [next(stream)]
        streamer.cancel()
        chunks.extend(stream)

        assert streamer.output is not None
        assert ''.join(chunks) == streamer.output.text
        assert streamer.output.profile.stop_reason == 'user'


@pytest.mark.vlm
@pytest.mark.parametrize('device_map', ['cpu'])
def test_vlm_context_reserves_one_logits_row(llama_cpp_vlm_paths, device_map, caplog):
    # Generation and decisions read only llama_get_logits_ith(ctx, -1). A
    # default context reserves one logits row per ubatch token instead.
    # init() first: install_log_callback seeds the SDK level from GENIEX_LOG,
    # which would otherwise overwrite an 'info' set before the first call.
    geniex.init()
    geniex.set_log_level('info')
    with caplog.at_level(logging.INFO, logger='geniex'):
        with geniex.AutoModelForVision2Seq.from_pretrained(
            _VLM.id,
            precision=_VLM.precision,
            device_map=device_map,
            n_ctx=512,
        ) as vlm:
            context_line = next((r.message for r in caplog.records if 'context params:' in r.message), None)
            assert context_line is not None, 'VLM context did not log its parameters'
            params = dict(re.findall(r'(\w+)=([^,\s]+)', context_line))
            assert params['n_outputs_max'] == params['n_seq_max'], context_line
            assert params['n_outputs_max_per_seq'] == '1', context_line
            prompt = vlm.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': 'Say hi.'}],
                tokenize=False,
                add_generation_prompt=True,
            )
            out = vlm.generate(prompt, max_new_tokens=1, temperature=GREEDY_TEMPERATURE, seed=42)
    assert out.profile.prompt_tokens > 0


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


def _forward_parity(device_map: str) -> tuple[list[list[tuple[int, float]]], list[float]]:
    with geniex.AutoModelForCausalLM.from_pretrained(
        _LLM.id,
        precision=_LLM.precision,
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
@pytest.mark.parametrize('device_map', ['cpu'])
def test_llm_forward_logits_all_positions(llama_cpp_llm_paths, device_map):
    # This API can request every position, so the LLM context must keep
    # llama.cpp's batch-sized output reservation. Only the VLM path is reduced.
    with geniex.AutoModelForCausalLM.from_pretrained(
        _LLM.id,
        precision=_LLM.precision,
        device_map=device_map,
    ) as llm:
        rows = llm.forward_logits(PARITY_INPUT_IDS, all_positions=True, top_n=1)
    assert len(rows) == len(PARITY_INPUT_IDS)


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['cpu'])
def test_llm_generate_input_embd(llama_cpp_llm_paths, device_map):
    # No public API exposes the model's own token-embedding table, so this
    # can't assert semantic parity with a text prompt — it only proves the
    # input_embd decode path runs end-to-end (batches, advances KV, samples)
    # without crashing on arbitrary (but correctly-shaped) float rows.
    random.seed(0)
    with geniex.AutoModelForCausalLM.from_pretrained(
        _LLM.id,
        precision=_LLM.precision,
        device_map=device_map,
    ) as llm:
        dim = llm.embd_dim
        assert dim > 0, 'llama_cpp should report a positive embd_dim'
        n_rows = 4
        embd = [random.uniform(-1.0, 1.0) for _ in range(n_rows * dim)]
        out = llm.generate(
            input_embd=embd,
            input_embd_dim=dim,
            max_new_tokens=8,
            temperature=GREEDY_TEMPERATURE,
            seed=LLM_QUALITY_SEED,
        )
        assert out.profile.generated_tokens > 0

        wrong_dim = dim + 1
        with pytest.raises(geniex.GenieXError) as excinfo:
            llm.generate(input_embd=[0.0] * wrong_dim, input_embd_dim=wrong_dim, max_new_tokens=1)
        assert excinfo.value.code == GENIEX_ERROR_COMMON_INVALID_INPUT


@pytest.mark.llm
@pytest.mark.nightly
@pytest.mark.parametrize('device_map', ['npu'])
@pytest.mark.skipif(_IS_QCS9075M, reason='draft-mtp not yet supported upstream on QCS9075M / HTP')
def test_mtp_multi_turn(llama_cpp_mtp_paths, device_map):
    # Local paths route through model_name= so the model-manager sees the
    # catalogue id; positional-arg would push the path into resolved_name and
    # trip the org/repo validator.
    with geniex.AutoModelForCausalLM.from_pretrained(
        llama_cpp_mtp_paths['target'].model_path,
        model_name=_MTP_TARGET.id,
        device_map=f'llama_cpp:{device_map}',
        spec_type='draft-mtp',
        spec_draft_model=llama_cpp_mtp_paths['draft'].model_path,
        spec_n_max=3,
    ) as llm:
        replies = _run_multi_turn(llm)
    assert (
        'alice' in replies[-1].lower()
    ), f'device_map={device_map!r} expected reply to recall "Alice", got={replies[-1]!r}'


@pytest.mark.llm
@pytest.mark.parametrize('device_map', _MTP_QWEN35.devices)
@pytest.mark.parametrize('scenario', ['reset', 'identical', 'edit', 'replace'])
def test_mtp_qwen35_cache_lifecycle(cached, device_map, scenario):
    # Qwen3.5 keeps its MTP head in nextn.* tensors and has a separate drafter
    # KV cache. Gemma's shared-cache MTP path does not exercise these contracts.
    if device_map == 'npu' and _IS_QCS9075M:
        pytest.skip('draft-mtp not yet supported upstream on QCS9075M / HTP')
    model_path = os.environ.get('GENIEX_TEST_MTP_MODEL') or cached(_MTP_QWEN35).model_path
    config = dict(
        model_name=_MTP_QWEN35.id,
        device_map=f'llama_cpp:{device_map}',
        n_ctx=512,
        n_batch=128,
        n_ubatch=128,
        n_threads=4,
    )
    questions = {
        'reset': 'Count from one to ten, separated by commas.',
        'identical': 'Count from one to ten, separated by commas.',
        'edit': 'Count from one to eight, separated by commas.',
        'replace': 'What is two plus two?',
    }
    with geniex.AutoModelForCausalLM.from_pretrained(model_path, spec_type='none', **config) as plain:
        prompts = [
            plain.tokenizer.apply_chat_template(
                [
                    {'role': 'system', 'content': 'Answer briefly without explaining.'},
                    {'role': 'user', 'content': question},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for question in ['Count from one to ten, separated by commas.', questions[scenario]]
        ]
        expected = []
        for prompt in prompts:
            plain.reset()
            expected.append(plain.generate(prompt, temperature=GREEDY_TEMPERATURE, max_new_tokens=64).text)

    with geniex.AutoModelForCausalLM.from_pretrained(
        model_path,
        spec_type='draft-mtp',
        spec_draft_model=model_path,
        spec_n_max=3,
        **config,
    ) as llm:
        for index, prompt in enumerate(prompts):
            if scenario == 'reset':
                llm.reset()
            out = llm.generate(prompt, temperature=GREEDY_TEMPERATURE, max_new_tokens=64)
            assert out.text, f'{scenario}: empty completion on call {index + 1}'
            # Setup silently falls back to plain decoding on error. Require
            # actual drafting so a disabled MTP path cannot make this pass.
            assert out.profile.draft_n_total > 0, f'{scenario}: MTP was not active'
            assert out.text == expected[index], f'{scenario}: call {index + 1} differs from a clean plain decode'


_MEDIA_MARKER = '<__media__>'


_WEATHER_TOOL = {
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


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['cpu'])
def test_chat_template(llama_cpp_llm_paths, device_map):
    hi = [{'role': 'user', 'content': 'hi'}]
    asked = [{'role': 'user', 'content': "what's the weather in Paris?"}]
    call = {
        'id': 'call_1',
        'type': 'function',
        'function': {'name': 'get_weather', 'arguments': '{"city": "Paris"}'},
    }
    answered = asked + [
        {'role': 'assistant', 'content': None, 'tool_calls': [call]},
        {'role': 'tool', 'tool_call_id': 'call_1', 'name': 'get_weather', 'content': 'MAGIC_SUNNY_42'},
    ]
    with geniex.AutoModelForCausalLM.from_pretrained(
        _LLM.id,
        precision=_LLM.precision,
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
        from_list = render(asked, tokenize=False, tools=[_WEATHER_TOOL])
        from_str = render(asked, tokenize=False, tools=json.dumps([_WEATHER_TOOL]))
        after = render(answered, tokenize=False, tools=[_WEATHER_TOOL])

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
    # Regression: tool calls used to be flattened into assistant text, so templates
    # that render a tool response only from the tool_calls of the turn before it
    # (Gemma 4, Mistral, Cohere) dropped the tool result from the prompt entirely.
    assert 'MAGIC_SUNNY_42' in after, f'tool response missing from the prompt: {after!r}'
    assert len(after) > len(from_list), 'adding a tool round trip did not grow the prompt'


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['cpu'])
def test_chat_template_content_load_override(llama_cpp_llm_paths, device_map):
    jinja = (
        '{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}'
        '{% if add_generation_prompt %}[assistant] {% endif %}'
    )
    with geniex.AutoModelForCausalLM.from_pretrained(
        _LLM.id,
        precision=_LLM.precision,
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
        _LLM.id,
        precision=_LLM.precision,
        device_map=device_map,
        n_ctx=n_ctx,
    ) as llm:
        out = llm.generate(prompt, max_new_tokens=120, temperature=GREEDY_TEMPERATURE, seed=0)

    assert out.profile.stop_reason != 'context_length', f'context shift did not save the run: {out.profile}'
    assert out.profile.generated_tokens > 0
    assert out.profile.prompt_tokens + out.profile.generated_tokens > n_ctx


@pytest.mark.llm
@pytest.mark.parametrize('device_map', ['cpu'])
def test_context_shift_not_triggered_when_nctx_is_ample(llama_cpp_llm_paths, device_map):
    with geniex.AutoModelForCausalLM.from_pretrained(
        _LLM.id,
        precision=_LLM.precision,
        device_map=device_map,
        n_ctx=2048,
    ) as llm:
        out = llm.generate('Say hi in one word.', max_new_tokens=8, temperature=GREEDY_TEMPERATURE, seed=0)

    assert out.profile.stop_reason in {'eos', 'length', 'completed'}, out.profile
    assert out.profile.generated_tokens > 0


@pytest.mark.vlm
@pytest.mark.parametrize('device_map', ['cpu'])
def test_mtmd_markers(llama_cpp_vlm_paths, test_image, quality_image, device_map):
    images = [{'type': 'image', 'image': test_image}, {'type': 'image', 'image': quality_image}]
    with geniex.AutoModelForVision2Seq.from_pretrained(
        _VLM.id,
        precision=_VLM.precision,
        device_map=device_map,
    ) as vlm:
        prompts = [
            vlm.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': [*images[:n], {'type': 'text', 'text': 'describe the picture'}]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for n in range(3)
        ]
        text_only = vlm.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'plain text, no image'}],
            tokenize=False,
            add_generation_prompt=True,
        )
    for n, prompt in enumerate(prompts):
        assert prompt.count(_MEDIA_MARKER) == n, f'n_images={n}: {prompt!r}'
        assert 'describe the picture' in prompt
    assert _MEDIA_MARKER not in text_only
    assert 'plain text, no image' in text_only


@pytest.mark.vlm
@pytest.mark.parametrize('device_map', ['cpu'])
def test_mtmd_generate_rejects_undecodable_image_and_recovers(llama_cpp_vlm_paths, test_image, device_map, tmp_path):
    corrupt = tmp_path / 'corrupt.jpg'
    corrupt.write_bytes(b'not a decodable image')

    with geniex.AutoModelForVision2Seq.from_pretrained(
        _VLM.id,
        precision=_VLM.precision,
        device_map=device_map,
    ) as vlm:
        prompt = _vlm_prompt(vlm, test_image, 'describe the picture')
        with pytest.raises(geniex.GenieXError) as error:
            vlm.generate(prompt, images=[str(corrupt)])
        assert error.value.code == GENIEX_ERROR_VLM_IMAGE_LOAD

        # A failed media load must leave the same model usable for valid media.
        recovered = vlm.generate(
            prompt,
            images=[test_image],
            max_new_tokens=1,
            temperature=GREEDY_TEMPERATURE,
            seed=42,
        )
    assert recovered.profile.prompt_tokens > 0


@pytest.mark.vlm
@pytest.mark.parametrize('device_map', ['cpu'])
def test_mtmd_generate_rejects_missing_image_path(llama_cpp_vlm_paths, test_image, device_map):
    # apply_chat_template records that content referenced an image, then
    # generate() validates the images=[...] list against the filesystem.
    missing = str(Path(test_image).parent / 'does-not-exist.png')
    with geniex.AutoModelForVision2Seq.from_pretrained(
        _VLM.id,
        precision=_VLM.precision,
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
