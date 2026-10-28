# SPDX-License-Identifier: BSD-3-Clause
"""Model-free tests against the native SDK and SDK test plugins.

Set GENIEX_TTS_CONTRACT_TEST=1, GENIEX_LIB_PATH and GENIEX_PLUGIN_PATH to run.
"""

import os
import struct
from ctypes import sizeof

import pytest

from geniex._ffi._api import GenieXError
from geniex._ffi._types import geniex_TtsCreateInput, geniex_TtsSynthesizeInput, geniex_TtsSynthesizeOutput
from geniex.tts import GenieXTTS

pytestmark = pytest.mark.skipif(
    os.environ.get('GENIEX_TTS_CONTRACT_TEST') != '1', reason='native TTS test SDK required'
)


def model(plugin='mock_tts'):
    return GenieXTTS(
        'unused', runtime_path='unused', text_library='unused', soc_model='unused', plugin_id=plugin, verify_npu=False
    )


def test_abi():
    assert sizeof(geniex_TtsCreateInput) == 64
    assert sizeof(geniex_TtsSynthesizeInput) == 40
    assert sizeof(geniex_TtsSynthesizeOutput) == 56
    assert geniex_TtsSynthesizeOutput.samples.offset == 8


def test_pcm_and_reuse():
    with model() as tts:
        result = tts.synthesize('test')
        assert result.samples == 40 and result.sample_rate == 24000
        assert struct.unpack('<4f', result.pcm[:16]) == (0, 0.5, -0.5, 0.25)
        assert not result.cancelled
        partial = tts.synthesize('test', on_audio=lambda *_: False)
        assert partial.cancelled and partial.samples == 4
        assert tts.synthesize('test').samples == 40
    with pytest.raises(RuntimeError, match='closed'):
        tts.synthesize('test')
    tts.close()  # idempotent


def test_callback_exception_and_cancel():
    with model() as tts:

        def fail(*_):
            raise ValueError('callback failure')

        with pytest.raises(ValueError, match='callback failure'):
            tts.synthesize('test', on_audio=fail)
        assert tts.synthesize('test', on_audio=lambda *_: tts.cancel()).cancelled
        assert not tts.synthesize('test').cancelled


def test_legacy_unsupported():
    with pytest.raises(GenieXError) as exc:
        model('legacy')
    assert exc.value.code == -100013


def test_input_validation():
    with model() as tts:
        for value in ['', ' ', 'bad\0text']:
            with pytest.raises(ValueError):
                tts.synthesize(value)


def test_callback_reentry_is_rejected():
    with model() as tts:

        def recursive(*_):
            tts.synthesize('nested')

        with pytest.raises(RuntimeError, match='busy'):
            tts.synthesize('test', on_audio=recursive)
