# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""SDK version metadata."""

from __future__ import annotations

import geniex


def test_version_nonempty(geniex_session):
    v = geniex.version()
    assert isinstance(v, str) and v


def test_qairt_plugin_version_nonempty(geniex_session):
    # Plugin reports its own version; available on hosts without an NPU
    # because the value comes from the shipped library, not the device.
    v = geniex.get_plugin_version('qairt')
    assert isinstance(v, str) and v


def test_llama_cpp_plugin_version_nonempty(geniex_session):
    v = geniex.get_plugin_version('llama_cpp')
    assert isinstance(v, str) and v
