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

"""Unit coverage for the #763 _cmd_chat load-banner: it must read plugin/device
from ``model._meta`` instead of calling ``resolve_device_map`` a second time."""

from __future__ import annotations

import argparse
import re

from geniex import cli as cli_module

ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


class _FakeLLM:
    def __init__(self, meta):
        self._meta = meta

    def close(self):
        pass


def _run_chat(monkeypatch, capsys, meta):
    monkeypatch.setattr(cli_module, '_ensure_downloaded', lambda *a, **kw: None)
    monkeypatch.setattr(
        cli_module.AutoModelForCausalLM,
        'from_pretrained',
        classmethod(lambda cls, *a, **kw: _FakeLLM(meta)),
    )
    monkeypatch.setattr(cli_module, '_run_turn', lambda *a, **kw: None)

    args = argparse.Namespace(
        model='org/repo',
        quant=None,
        hub=None,
        display_name=None,
        chipset=None,
        local_path=None,
        device='auto',
        n_ctx=0,
        prompt='hi',
        system=None,
        max_tokens=8,
        temperature=0.0,
    )
    rc = cli_module._cmd_chat(args)
    out = ANSI_RE.sub('', capsys.readouterr().out)
    return rc, out


def test_banner_prints_backend_and_device_from_meta(monkeypatch, capsys):
    rc, out = _run_chat(monkeypatch, capsys, {'backend': 'llama_cpp', 'device': 'HTP0'})
    assert rc == 0
    assert 'done (llm,' in out
    assert 'llama_cpp:HTP0' in out


def test_banner_falls_back_to_args_device_when_meta_missing(monkeypatch, capsys):
    rc, out = _run_chat(monkeypatch, capsys, None)
    assert rc == 0
    assert 'done (llm,' in out
    assert 'auto' in out


def test_banner_uses_backend_only_when_device_missing(monkeypatch, capsys):
    rc, out = _run_chat(monkeypatch, capsys, {'backend': 'qairt', 'device': None})
    assert rc == 0
    assert 'qairt' in out
    assert 'qairt:' not in out
