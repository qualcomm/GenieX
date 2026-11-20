# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from geniex._ffi._types import GENIEX_SAMPLER_FIELDS
from geniex.modeling import _build_sampler


def test_unset_fields_defer_to_plugin_defaults():
    sampler = _build_sampler(None, None, None, None, None, None, None, None, None)
    assert sampler.set_fields == 0
    assert sampler.temperature == 0


@pytest.mark.parametrize('name', list(GENIEX_SAMPLER_FIELDS))
def test_explicit_zero_sets_field_bit(name):
    args = dict.fromkeys(GENIEX_SAMPLER_FIELDS)
    args[name] = 0
    sampler = _build_sampler(*args.values(), None)
    assert sampler.set_fields == GENIEX_SAMPLER_FIELDS[name]
    assert getattr(sampler, name) == 0
