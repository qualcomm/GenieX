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

"""Pytest smoke wrapper around the benchmark runner.

Default-excluded by ``addopts = -m "not benchmark"`` in pytest.ini, so PR
Check is unaffected. Run explicitly with ``pytest tests -m benchmark``.
"""

from __future__ import annotations

import pytest
from _matrix import LLM_CELLS, is_snapdragon_host, skip_reason
from _runner import run_cell

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize('cell', LLM_CELLS, ids=lambda c: c.cell_id)
def test_cell_runs(cell, geniex_session):
    reason = skip_reason(cell, is_snapdragon_host())
    if reason:
        pytest.skip(reason)
    result = run_cell(cell, warmup=0, repeat=1)
    assert result['status'] == 'ok', result.get('error')
    assert result['runs'], 'no measured runs recorded'
    assert result['agg']['decode_tps']['median'] > 0, result['agg']
