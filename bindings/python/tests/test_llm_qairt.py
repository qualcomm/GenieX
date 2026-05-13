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

"""QAIRT (NPU) end-to-end test. Skips automatically when not on Snapdragon or
when ``GENIEX_DEVICE_TEST=1`` is not set."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import geniex

from .conftest import QAIRT_MODEL


@pytest.fixture(scope='module')
def qairt_llm(qairt_paths):
    model = geniex.AutoModelForCausalLM.from_pretrained(QAIRT_MODEL)
    yield model
    model.close()


def test_qairt_model_loads(qairt_llm):
    assert isinstance(qairt_llm, geniex.GeniexLLM)


def test_qairt_generate_blocking(qairt_llm):
    out = qairt_llm.generate('Say hi.', max_new_tokens=16, temperature=0.0, seed=42)
    assert out.text
    assert out.profile.generated_tokens > 0


def test_qairt_apply_chat_template(qairt_llm):
    text = qairt_llm.tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'hi'}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(text, str) and text


def test_cli_chat_qairt_single_prompt(qairt_paths):
    # Exercise the full `geniex-py chat -p ...` path on a QAIRT model.
    # Runs in a subprocess to match how a downstream user invokes it and
    # to confirm packaging (console-script + dispatch) is intact.
    env = {**os.environ, 'NO_COLOR': '1'}
    r = subprocess.run(
        [
            sys.executable,
            '-m',
            'geniex.cli',
            'chat',
            QAIRT_MODEL,
            '--max-tokens',
            '8',
            '-p',
            'Say hi.',
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr
    # Should print the `(llm)` dispatch tag and a non-empty generation.
    assert '(llm)' in r.stdout
    assert r.stdout.strip() != ''
