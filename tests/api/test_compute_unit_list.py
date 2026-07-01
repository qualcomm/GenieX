# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""``geniex.get_compute_unit_list`` shape per runtime."""

from __future__ import annotations

import geniex


def test_compute_unit_list_shape_for_each_runtime(geniex_session):
    for runtime in geniex.get_runtime_list():
        compute_units = geniex.get_compute_unit_list(runtime)
        assert isinstance(compute_units, list)
        for entry in compute_units:
            assert isinstance(entry, tuple) and len(entry) == 2
            compute_unit, label = entry
            assert isinstance(compute_unit, str) and compute_unit
            assert isinstance(label, str)
