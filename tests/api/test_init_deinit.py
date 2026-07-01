# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

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
        'get_runtime_list',
        'get_compute_unit_list',
        'resolve_device_map',
        'model_manager',
    }
    assert expected.issubset(set(geniex.__all__))
    for name in expected:
        assert hasattr(geniex, name), f'{name} missing from geniex module'
