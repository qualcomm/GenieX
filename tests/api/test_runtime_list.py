# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""``geniex.get_runtime_list`` shape and contents."""

from __future__ import annotations

import geniex


def test_runtime_list_is_non_empty_string_list(geniex_session):
    runtimes = geniex.get_runtime_list()
    assert isinstance(runtimes, list) and runtimes
    for r in runtimes:
        assert isinstance(r, str) and r


def test_runtime_list_contains_llama_cpp(geniex_session):
    assert 'llama_cpp' in geniex.get_runtime_list()
