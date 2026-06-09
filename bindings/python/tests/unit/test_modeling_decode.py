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

"""Unit coverage for the lenient UTF-8 decode helper that backs LLM/VLM
generate output. Regression for #888: a generation truncated at
``max_new_tokens`` can land mid-multibyte and was raising UnicodeDecodeError
in the binding layer."""

from __future__ import annotations

from ctypes import c_char_p

from geniex.modeling import _decode_utf8


def test_empty_and_null():
    assert _decode_utf8(None) == ''
    assert _decode_utf8(c_char_p(b'')) == ''


def test_ascii_roundtrip():
    assert _decode_utf8(c_char_p(b'hello')) == 'hello'


def test_well_formed_multibyte_roundtrip():
    payload = '你好 🌏'.encode()
    assert _decode_utf8(c_char_p(payload)) == '你好 🌏'


def test_truncated_multibyte_does_not_raise():
    # '你好' is 6 UTF-8 bytes (3 + 3); chop the last byte to mimic a
    # max_new_tokens cut mid-character.
    truncated = '你好'.encode()[:-1]
    out = _decode_utf8(c_char_p(truncated))
    assert out.startswith('你')
    assert '�' in out


def test_truncated_emoji_does_not_raise():
    # 🌏 is a 4-byte sequence; drop the trailing continuation byte.
    truncated = 'ok 🌏'.encode()[:-1]
    out = _decode_utf8(c_char_p(truncated))
    assert out.startswith('ok ')
    assert '�' in out
