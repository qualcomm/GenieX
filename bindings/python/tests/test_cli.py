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

"""Contract tests for the geniex-py CLI.

Proves that cli.py consumes only the public ``geniex`` surface (never
``_ffi``), and that the common subcommands run end-to-end against the
installed wheel layout.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from geniex.cli import _parse_media

_CLI_PATH = Path(__file__).resolve().parent.parent / 'geniex' / 'cli.py'


def test_cli_imports_only_public_surface():
    src = _CLI_PATH.read_text(encoding='utf-8')
    # Strip the module docstring before scanning imports.
    without_docstring = re.sub(r'"""[\s\S]*?"""', '', src, count=1)
    assert '_ffi' not in without_docstring, 'cli.py must not import from geniex._ffi'
    # Only two legitimate ctypes mentions are allowed: none at all in cli.
    assert 'import ctypes' not in without_docstring
    assert 'from ctypes' not in without_docstring


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, '-m', 'geniex.cli', *args],
        capture_output=True,
        text=True,
        env={**os.environ},
    )


def test_cli_devices_returns_zero(geniex_session):
    r = _run_cli(['devices'])
    assert r.returncode == 0, r.stderr
    # Output should contain at least one plugin name.
    assert 'llama_cpp' in r.stdout or 'qairt' in r.stdout


def test_cli_ls_runs(tmp_path):
    # Use an empty, disposable data dir so the test never mutates the user
    # cache. The output may be the "no models cached" hint or a table header.
    env = {**os.environ, 'GENIEX_DATADIR': str(tmp_path)}
    r = subprocess.run(
        [sys.executable, '-m', 'geniex.cli', 'ls'],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() != ''


def test_cli_help_runs():
    # --help on each subcommand must exit 0 without touching the SDK.
    r = subprocess.run(
        [sys.executable, '-m', 'geniex.cli', '--help'],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert 'GenieX Python CLI' in r.stdout


def test_cli_version_prints_three_lines(geniex_session):
    r = _run_cli(['version'])
    assert r.returncode == 0, r.stderr
    assert 'geniex (python):' in r.stdout
    assert 'SDK:' in r.stdout
    assert 'QAIRT:' in r.stdout


def test_cli_verbose_emits_sdk_logs(geniex_session):
    # -v sets level to info; the SDK emits at least one INFO line during
    # plugin enumeration, which the bridge routes to the geniex logger.
    r = _run_cli(['-v', 'devices'])
    assert r.returncode == 0, r.stderr
    assert 'geniex' in r.stderr  # bridge logger name appears in records


def test_cli_log_level_overrides_verbose(geniex_session):
    # --log-level error wins over -vv, so info/debug records are filtered.
    r = _run_cli(['-vv', '--log-level', 'error', 'devices'])
    assert r.returncode == 0, r.stderr
    assert 'INFO' not in r.stderr
    assert 'DEBUG' not in r.stderr


def test_parse_media_extracts_image(tmp_path):
    img = tmp_path / 'cat.png'
    img.write_bytes(b'\x89PNG\r\n\x1a\n')
    prompt, images, audios = _parse_media(f'Describe {img}')
    assert prompt == 'Describe'
    assert images == [str(img)]
    assert audios == []


def test_parse_media_extracts_audio(tmp_path):
    wav = tmp_path / 'sample.wav'
    wav.write_bytes(b'RIFF')
    prompt, images, audios = _parse_media(f'Transcribe {wav}')
    assert audios == [str(wav)]
    assert images == []
    assert 'Transcribe' in prompt and str(wav) not in prompt


def test_parse_media_ignores_missing_file():
    prompt, images, audios = _parse_media('Look at ./does-not-exist.png please')
    assert images == []
    assert audios == []
    assert './does-not-exist.png' in prompt


def test_parse_media_leaves_plain_text_alone():
    prompt, images, audios = _parse_media('Hello world, no media here.')
    assert prompt == 'Hello world, no media here.'
    assert images == []
    assert audios == []
