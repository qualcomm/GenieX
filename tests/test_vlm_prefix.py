# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import shutil

import geniex
import pytest

from _models import primary

pytestmark = [pytest.mark.llama_cpp, pytest.mark.vlm, pytest.mark.device_cpu]
REBUILDS = os.environ.get('GENIEX_TEST_VLM_REBUILDS') == '1'
EXACT_OUTPUT = os.environ.get('GENIEX_TEST_VLM_EXACT_OUTPUT') == '1' or os.environ.get('GENIEX_TEST_VLM_UBATCH') == '1'


@pytest.fixture(scope='module')
def prefix_model(request):
    model = os.environ.get('GENIEX_TEST_VLM_MODEL')
    projector = os.environ.get('GENIEX_TEST_VLM_MMPROJ')
    if not model:
        request.getfixturevalue('llama_cpp_vlm_paths')
        selected = primary('llama_cpp_vlm')
        model = selected.id
        precision = selected.precision
    else:
        precision = None
    with geniex.AutoModelForVision2Seq.from_pretrained(
        model,
        precision=precision,
        mmproj_path=projector,
        device_map='cpu',
        n_ctx=4096,
        n_threads=4,
        n_threads_batch=4,
        n_ubatch=int(os.environ.get('GENIEX_TEST_VLM_UBATCH', '0')),
    ) as vlm:
        yield vlm


@pytest.fixture
def vlm(prefix_model):
    prefix_model.reset()
    return prefix_model


def history(image):
    return [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Look carefully. '},
                {'type': 'image', 'image': image, 'media_id': 'image-a'},
                {'type': 'text', 'text': ' Describe this picture.'},
            ],
        }
    ]


def generate(vlm, messages):
    prompt = vlm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return vlm.generate(prompt, max_new_tokens=8, temperature=-1, seed=42)


@pytest.mark.parametrize(
    'revision',
    [
        'identical',
        'append',
        'edit_after',
        'edit_before',
        'trim',
        'thinking_removed',
        'replace',
        'reorder',
    ],
)
def test_revised_history_reuses_verified_prefix(vlm, test_image, quality_image, revision):
    previous = history(test_image)
    previous.extend(
        [
            {'role': 'assistant', 'content': '<think>Examine the details.</think>A picture.'},
            {'role': 'user', 'content': 'Describe one detail.'},
        ]
    )
    if revision == 'reorder':
        previous[0]['content'].insert(2, {'type': 'image', 'image': quality_image, 'media_id': 'image-b'})
    first = generate(vlm, previous)
    assert first.profile.media_time > 0
    current = copy.deepcopy(previous)
    if revision == 'append':
        current.extend(
            [
                {'role': 'assistant', 'content': 'A shape.'},
                {'role': 'user', 'content': 'What color?'},
                {'role': 'assistant', 'content': 'Blue.'},
                {'role': 'user', 'content': 'Summarize.'},
            ]
        )
    elif revision == 'edit_after':
        current[0]['content'][-1]['text'] = ' Describe it briefly.'
    elif revision == 'edit_before':
        current[0]['content'][0]['text'] = 'Inspect this. '
    elif revision == 'trim':
        current = current[:1]
    elif revision == 'thinking_removed':
        current[1]['content'] = 'A picture.'
    elif revision == 'replace':
        current[0]['content'][1] = {'type': 'image', 'image': quality_image, 'media_id': 'image-b'}
    elif revision == 'reorder':
        parts = current[0]['content']
        parts[1], parts[2] = parts[2], parts[1]
    reused = generate(vlm, current)
    if not REBUILDS and revision in {'identical', 'append', 'edit_after', 'trim', 'thinking_removed'}:
        assert reused.profile.media_time == 0
    else:
        assert reused.profile.media_time > 0
    vlm.reset()
    rebuilt = generate(vlm, current)
    assert rebuilt.profile.media_time > 0
    assert 0 < reused.profile.prompt_tokens <= rebuilt.profile.prompt_tokens
    if EXACT_OUTPUT:
        assert reused.text == rebuilt.text
    assert reused.profile.generated_tokens == rebuilt.profile.generated_tokens


def test_missing_media_rebuild_resets_retained_state(vlm, test_image, tmp_path):
    image = tmp_path / 'image.png'
    shutil.copyfile(test_image, image)
    messages = history(str(image))
    generate(vlm, messages)
    image.unlink()
    if REBUILDS:
        with pytest.raises(geniex.GenieXError, match='VLM prefix reuse failed'):
            generate(vlm, messages)
    else:
        assert generate(vlm, messages).profile.media_time == 0
    edited = copy.deepcopy(messages)
    edited[0]['content'][0]['text'] = 'Changed before the image. '
    with pytest.raises(geniex.GenieXError, match='VLM prefix reuse failed'):
        generate(vlm, edited)
    # The failed rebuild must also invalidate the earlier matching history.
    with pytest.raises(geniex.GenieXError, match='VLM prefix reuse failed'):
        generate(vlm, messages)
    shutil.copyfile(test_image, image)
    assert generate(vlm, messages).profile.media_time > 0


def test_stable_identity_with_new_path(vlm, test_image, tmp_path):
    messages = history(test_image)
    first = generate(vlm, messages)
    new_path = tmp_path / 'moved.png'
    shutil.copyfile(test_image, new_path)
    messages[0]['content'][1]['image'] = str(new_path)
    reused = generate(vlm, messages)
    assert (reused.profile.media_time > 0) == REBUILDS
    if EXACT_OUTPUT:
        assert reused.text == first.text


@pytest.mark.parametrize('media_id', [None, ''])
def test_implicit_identity_defaults_to_path(vlm, test_image, media_id):
    messages = history(test_image)
    messages[0]['content'][1]['media_id'] = media_id
    first = generate(vlm, messages)
    reused = generate(vlm, messages)
    assert (reused.profile.media_time > 0) == REBUILDS
    if EXACT_OUTPUT:
        assert reused.text == first.text


def test_template_preserves_content_order(vlm, test_image):
    prompt = vlm.tokenizer.apply_chat_template(history(test_image), tokenize=False)
    assert prompt.index('Look carefully.') < prompt.index('<__media__>') < prompt.index('Describe this picture.')


def test_media_modality_mismatch_is_rejected(vlm, test_image):
    messages = history(test_image)
    messages[0]['content'][1]['type'] = 'audio'
    with pytest.raises(geniex.GenieXError):
        generate(vlm, messages)


def test_pending_descriptor_requires_exact_prompt(vlm, test_image):
    prompt = vlm.tokenizer.apply_chat_template(history(test_image), tokenize=False)
    with pytest.raises(geniex.GenieXError, match='VLM prefix reuse failed'):
        vlm.generate(prompt + ' changed', max_new_tokens=1)
    with pytest.raises(geniex.GenieXError, match='VLM prefix reuse failed'):
        vlm.generate(prompt, max_new_tokens=1)


def test_thinking_flag_reaches_vlm_template(vlm):
    thinking = os.environ.get('GENIEX_TEST_VLM_THINKING')
    if thinking == '0' or (thinking is None and not (vlm._meta or {}).get('supports_thinking')):
        pytest.skip('the selected model has no thinking template')
    messages = [{'role': 'user', 'content': 'What is 2 + 2?'}]
    enabled = vlm.tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=True)
    disabled = vlm.tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    assert enabled != disabled
