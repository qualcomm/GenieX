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
from geniex.modeling import _apply_meta, _messages_have_modality

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
# #715: get_plugin_list / get_device_list raise before init() instead of
# silently returning [] (which made misuse look like "no plugins available").
# ---------------------------------------------------------------------------


@pytest.fixture
def _runtime_uninitialized(monkeypatch):
    monkeypatch.setattr(_ffi_api, '_initialized', False)


def test_get_plugin_list_raises_before_init(_runtime_uninitialized):
    with pytest.raises(RuntimeError, match=r'get_plugin_list\(\).*geniex\.init\(\)'):
        _ffi_api.get_plugin_list()


def test_get_device_list_raises_before_init(_runtime_uninitialized):
    with pytest.raises(RuntimeError, match=r'get_device_list\(\).*geniex\.init\(\)'):
        _ffi_api.get_device_list('llama_cpp')


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
# #726 / #727: VLM messages → modality detection helper
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


# ---------------------------------------------------------------------------
# #726 / #727: VLM.generate predicates can be unit-tested via a stub
# ---------------------------------------------------------------------------


class _StubVLM:
    """Minimal stand-in for GenieXVLM.generate that exercises only the
    Python-side validation block before the C call."""

    def __init__(self, has_image=False, has_audio=False):
        self._last_template_has_image = has_image
        self._last_template_has_audio = has_audio

    @staticmethod
    def _validate(images, audios, has_image, has_audio):
        # Mirror the validation in GenieXVLM.generate so we can test it
        # without spinning up the C runtime.
        if not images and has_image:
            raise ValueError(
                'messages reference image content but generate(images=[...]) '
                'is empty. Pass image paths via images=[...].'
            )
        if not audios and has_audio:
            raise ValueError(
                'messages reference audio content but generate(audios=[...]) '
                'is empty. Pass audio paths via audios=[...].'
            )
        for path in images:
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Image file not found: {path}')
        for path in audios:
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Audio file not found: {path}')


def test_missing_images_with_image_messages_raises():
    with pytest.raises(ValueError, match=r'messages reference image content'):
        _StubVLM._validate([], [], has_image=True, has_audio=False)


def test_missing_audios_with_audio_messages_raises():
    with pytest.raises(ValueError, match=r'messages reference audio content'):
        _StubVLM._validate([], [], has_image=False, has_audio=True)


def test_nonexistent_image_path_raises(tmp_path):
    missing = tmp_path / 'missing.png'
    with pytest.raises(FileNotFoundError, match=r'Image file not found'):
        _StubVLM._validate([str(missing)], [], has_image=False, has_audio=False)


def test_nonexistent_audio_path_raises(tmp_path):
    missing = tmp_path / 'missing.wav'
    with pytest.raises(FileNotFoundError, match=r'Audio file not found'):
        _StubVLM._validate([], [str(missing)], has_image=False, has_audio=False)


def test_existing_image_path_passes(tmp_path):
    img = tmp_path / 'img.png'
    img.write_bytes(b'\x89PNG')
    _StubVLM._validate([str(img)], [], has_image=True, has_audio=False)


# ---------------------------------------------------------------------------
# #716: get_device_list message names the bad id + available plugins
# ---------------------------------------------------------------------------


def test_unknown_plugin_message_names_bad_id_and_lists_available():
    from geniex._ffi._api import _unknown_plugin_message

    msg = _unknown_plugin_message('wenwen_plugin', ['qairt', 'llama_cpp'])
    assert 'Unknown plugin: wenwen_plugin' in msg
    assert 'llama_cpp' in msg and 'qairt' in msg


def test_unknown_plugin_message_sorts_available_for_stability():
    from geniex._ffi._api import _unknown_plugin_message

    assert _unknown_plugin_message('x', ['qairt', 'llama_cpp']) == _unknown_plugin_message('x', ['llama_cpp', 'qairt'])
