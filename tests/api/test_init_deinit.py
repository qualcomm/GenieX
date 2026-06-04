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

"""Lifecycle and public-surface checks for the geniex package."""

from __future__ import annotations

import geniex


def test_init_deinit_is_idempotent_within_session(geniex_session):
    geniex.init()
    geniex.init()


def test_set_log_level_accepts_known_levels(geniex_session):
    for level in ('trace', 'debug', 'info', 'warn', 'error', 'none'):
        geniex.set_log_level(level)


def test_public_surface_exports():
    expected = {
        'AutoModelForCausalLM',
        'AutoModelForVision2Seq',
        'GenieXError',
        'GenieXLLM',
        'GenieXVLM',
        'GenerateOutput',
        'ProfileData',
        'TextIteratorStreamer',
        'init',
        'deinit',
        'set_log_level',
        'version',
        'get_plugin_version',
        'get_plugin_list',
        'get_device_list',
        'resolve_device_map',
        'model_manager',
    }
    assert expected.issubset(set(geniex.__all__))
    for name in expected:
        assert hasattr(geniex, name), f'{name} missing from geniex module'
