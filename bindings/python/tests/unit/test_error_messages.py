# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python unit coverage for the dogfood error-message and profile
metadata fixes (issues #715, #725, #726, #727, #736, #739).

These tests bypass the native SDK and exercise the helper functions
directly so they collect and pass even when ``geniex/lib/`` is not staged.
"""

from __future__ import annotations

import json
import os

import pytest

from geniex._ffi import _api as _ffi_api
from geniex.auto import (
    PLUGIN_LLAMA_CPP,
    PLUGIN_QAIRT,
    _read_bundle_metadata,
    _reject_gguf_on_qairt,
    _resolve_model_sources,
)
from geniex.generation.output import ProfileData
from geniex.modeling import GenieXVLM, _apply_meta, _build_vlm_messages, _messages_have_modality

# ---------------------------------------------------------------------------
# #725: model_name mismatch against bundle metadata.json
# ---------------------------------------------------------------------------


def _write_bundle(dir_path, model_id):
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump({'model_id': model_id}, f)


def test_model_name_mismatch_raises_value_error(tmp_path, monkeypatch):
    bundle = tmp_path / 'bundle'
    _write_bundle(bundle, 'real-bundle-name')

    # If the pre-check fails to fire, _mm.pull would be invoked next and
    # would explode without SDK init — fail loudly so a regression is
    # obvious.
    def _pull_should_not_be_called(*_a, **_kw):
        raise AssertionError('_mm.pull should not be reached on mismatch')

    monkeypatch.setattr('geniex.auto._mm.pull', _pull_should_not_be_called)

    with pytest.raises(ValueError, match=r"does not match bundle 'model_id=real-bundle-name'"):
        _resolve_model_sources(
            str(bundle),
            quant=None,
            hf_token=None,
            progress=None,
            model_name='wrong_name',
        )


def test_model_name_match_passes_through(tmp_path, monkeypatch):
    bundle = tmp_path / 'bundle'
    _write_bundle(bundle, 'good_name')

    captured = {}

    def _fake_pull(name, **kwargs):
        captured['name'] = name
        captured['kwargs'] = kwargs

    class _FakePaths:
        model_path = str(bundle / 'metadata.json')
        mmproj_path = None
        tokenizer_path = None

    monkeypatch.setattr('geniex.auto._mm.pull', _fake_pull)
    monkeypatch.setattr('geniex.auto._mm.get_paths', lambda _name: _FakePaths())

    out = _resolve_model_sources(
        str(bundle),
        quant=None,
        hf_token=None,
        progress=None,
        model_name='good_name',
    )
    assert out[0] == _FakePaths.model_path
    assert captured['name'] == 'good_name'


def test_read_bundle_metadata_missing_returns_none(tmp_path):
    assert _read_bundle_metadata(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# #739: .gguf is not allowed on QAIRT
# ---------------------------------------------------------------------------


def test_reject_gguf_on_qairt_raises():
    with pytest.raises(ValueError, match=r'\.gguf models are not supported'):
        _reject_gguf_on_qairt('/cache/Qwen3-0.6B-Q4_0.gguf', PLUGIN_QAIRT, 'npu')


def test_reject_gguf_on_qairt_quotes_user_device_map():
    # The error message should echo the *user's* device_map (not the
    # internal plugin id) so they can map it back to their call.
    with pytest.raises(ValueError, match=r"device_map='qairt:NPU'"):
        _reject_gguf_on_qairt('/x.gguf', PLUGIN_QAIRT, 'qairt:NPU')


def test_reject_gguf_on_llama_cpp_passes():
    # llama_cpp consumes .gguf natively — no rejection.
    _reject_gguf_on_qairt('/cache/Qwen3-0.6B-Q4_0.gguf', PLUGIN_LLAMA_CPP, 'auto')


def test_reject_qairt_bundle_passes():
    # QAIRT directory bundles (no .gguf suffix) must pass.
    _reject_gguf_on_qairt('/bundle/metadata.json', PLUGIN_QAIRT, 'npu')


# ---------------------------------------------------------------------------
# #715: get_runtime_list / get_compute_unit_list raise before init() instead of
# silently returning [] (which made misuse look like "no runtimes available").
# ---------------------------------------------------------------------------


@pytest.fixture
def _runtime_uninitialized(monkeypatch):
    monkeypatch.setattr(_ffi_api, '_initialized', False)


def test_get_runtime_list_raises_before_init(_runtime_uninitialized):
    with pytest.raises(RuntimeError, match=r'get_runtime_list\(\).*geniex\.init\(\)'):
        _ffi_api.get_runtime_list()


def test_get_compute_unit_list_raises_before_init(_runtime_uninitialized):
    with pytest.raises(RuntimeError, match=r'get_compute_unit_list\(\).*geniex\.init\(\)'):
        _ffi_api.get_compute_unit_list('llama_cpp')


# ---------------------------------------------------------------------------
# #736: ProfileData carries backend/device/quant/model_path
# ---------------------------------------------------------------------------


def test_profile_data_defaults_meta_to_none():
    pd = ProfileData()
    assert pd.backend is None
    assert pd.device is None
    assert pd.quant is None
    assert pd.model_path is None


def test_apply_meta_populates_profile():
    pd = ProfileData(generated_tokens=8)
    meta = {
        'backend': 'llama_cpp',
        'device': 'NPU',
        'quant': 'Q4_0',
        'model_path': '/cache/m.gguf',
    }
    out = _apply_meta(pd, meta)
    assert out is pd  # mutates in place
    assert pd.backend == 'llama_cpp'
    assert pd.device == 'NPU'
    assert pd.quant == 'Q4_0'
    assert pd.model_path == '/cache/m.gguf'


def test_apply_meta_none_leaves_profile_unchanged():
    pd = ProfileData(generated_tokens=8, backend='preset')
    _apply_meta(pd, None)
    assert pd.backend == 'preset'


# ---------------------------------------------------------------------------
# VLM media identity reaches the C message structure
# ---------------------------------------------------------------------------


def test_messages_have_modality_detects_image():
    msgs = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': '/x.png'},
                {'type': 'text', 'text': 'caption?'},
            ],
        }
    ]
    assert _messages_have_modality(msgs, 'image') is True
    assert _messages_have_modality(msgs, 'audio') is False


def test_messages_have_modality_string_content():
    msgs = [{'role': 'user', 'content': 'plain text only'}]
    assert _messages_have_modality(msgs, 'image') is False


def test_messages_have_modality_audio():
    msgs = [
        {
            'role': 'user',
            'content': [
                {'type': 'audio', 'audio': '/x.wav'},
                {'type': 'text', 'text': 'transcribe'},
            ],
        }
    ]
    assert _messages_have_modality(msgs, 'audio') is True


def test_messages_have_modality_ignores_prior_turns():
    # Multi-turn VLM: the current turn is text-only, so generate() must
    # not be gated on the image block from an earlier turn -- prior-turn
    # media is already in the KV cache and must not be re-supplied.
    msgs = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': '/cat.png'},
                {'type': 'text', 'text': 'describe'},
            ],
        },
        {'role': 'assistant', 'content': 'a cat on a mat'},
        {'role': 'user', 'content': 'what color?'},
    ]
    assert _messages_have_modality(msgs, 'image') is False
    assert _messages_have_modality(msgs, 'audio') is False


def test_messages_have_modality_detects_current_turn():
    # Same shape as above but the *current* turn references an image ->
    # generate() must still fire the "pass images=[...]" guard.
    msgs = [
        {'role': 'user', 'content': 'hello'},
        {'role': 'assistant', 'content': 'hi'},
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': '/dog.png'},
                {'type': 'text', 'text': 'describe'},
            ],
        },
    ]
    assert _messages_have_modality(msgs, 'image') is True


def test_messages_have_modality_empty():
    assert _messages_have_modality([], 'image') is False


def test_vlm_media_identity_passes_to_c():
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': '/tmp/image.png', 'media_id': 'stable-image'},
                {'type': 'text', 'text': 'describe'},
            ],
        }
    ]
    c_messages, count, refs = _build_vlm_messages(messages)
    assert count == 1
    assert refs
    assert c_messages[0].contents[0].text == b'/tmp/image.png'
    assert c_messages[0].contents[0].media_id == b'stable-image'
    assert c_messages[0].contents[1].media_id is None


@pytest.mark.parametrize('modality', ['image', 'audio'])
def test_qairt_missing_current_media_still_raises(modality):
    vlm = GenieXVLM(None, meta={'backend': 'qairt'})
    setattr(vlm, f'_last_template_has_{modality}', True)
    with pytest.raises(ValueError, match=f'messages reference {modality} content'):
        vlm.generate('prompt')


def test_llama_cpp_defers_historical_media_validation_to_sdk(monkeypatch):
    vlm = GenieXVLM(None, meta={'backend': 'llama_cpp'})
    vlm._last_template_has_image = True
    vlm._last_template_has_audio = True
    monkeypatch.setattr(vlm, '_generate_blocking', lambda *args: 'sdk result')
    assert vlm.generate('prompt') == 'sdk result'


@pytest.mark.parametrize('parameter', ['images', 'audios'])
def test_nonexistent_media_path_raises(tmp_path, parameter):
    vlm = GenieXVLM(None, meta={'backend': 'llama_cpp'})
    with pytest.raises(FileNotFoundError, match='file not found'):
        vlm.generate('prompt', **{parameter: [str(tmp_path / 'missing')]})


# ---------------------------------------------------------------------------
# #716: get_compute_unit_list message names the bad id + available runtimes
# ---------------------------------------------------------------------------


def test_unknown_runtime_message_names_bad_id_and_lists_available():
    from geniex._ffi._api import _unknown_runtime_message

    msg = _unknown_runtime_message('wenwen_runtime', ['qairt', 'llama_cpp'])
    assert 'Unknown runtime: wenwen_runtime' in msg
    assert 'llama_cpp' in msg and 'qairt' in msg


def test_unknown_runtime_message_sorts_available_for_stability():
    from geniex._ffi._api import _unknown_runtime_message

    assert _unknown_runtime_message('x', ['qairt', 'llama_cpp']) == _unknown_runtime_message(
        'x', ['llama_cpp', 'qairt']
    )
