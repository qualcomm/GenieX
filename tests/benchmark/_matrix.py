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

"""Benchmark cell definitions and host-capability gating."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path

# tests/_models.py is a flat module (no package), so add tests/ to sys.path
# whether we're imported via `python -m tests.benchmark.run` or pytest.
_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from _models import (  # noqa: E402
    LLAMA_CPP_LLM_MODEL,
    LLAMA_CPP_LLM_QUANT,
    QAIRT_LLM_MODEL,
)


@dataclass(frozen=True)
class BenchCell:
    model: str
    quant: str | None
    backend: str
    device: str

    @property
    def cell_id(self) -> str:
        slug = self.model.split('/')[-1]
        return f'{slug}-{self.backend}-{self.device}'


LLM_CELLS: list[BenchCell] = [
    BenchCell(LLAMA_CPP_LLM_MODEL, LLAMA_CPP_LLM_QUANT, 'llama_cpp', 'cpu'),
    BenchCell(LLAMA_CPP_LLM_MODEL, LLAMA_CPP_LLM_QUANT, 'llama_cpp', 'gpu'),
    BenchCell(LLAMA_CPP_LLM_MODEL, LLAMA_CPP_LLM_QUANT, 'llama_cpp', 'npu'),
    BenchCell(LLAMA_CPP_LLM_MODEL, LLAMA_CPP_LLM_QUANT, 'llama_cpp', 'hybrid'),
    BenchCell(QAIRT_LLM_MODEL, None, 'qairt', 'npu'),
]

_SNAPDRAGON_DEVICES = {'gpu', 'npu', 'hybrid'}


def is_snapdragon_host() -> bool:
    """Mirror conftest._is_snapdragon_host so benchmark/ stays import-isolated."""
    if platform.machine().lower() not in ('arm64', 'aarch64'):
        return False
    if platform.system() == 'Windows' or hasattr(sys, 'getandroidapilevel'):
        return True
    try:
        with open('/sys/firmware/devicetree/base/compatible', 'rb') as f:
            return b'qcom' in f.read()
    except OSError:
        return False


def skip_reason(cell: BenchCell, snapdragon: bool) -> str | None:
    """Return None if the cell can run on this host, else a human-readable reason."""
    if cell.backend == 'qairt' and not snapdragon:
        return 'qairt requires a Snapdragon host'
    if cell.device in _SNAPDRAGON_DEVICES and not snapdragon:
        return f'device {cell.device!r} requires a Snapdragon host'
    return None


def filter_cells(
    cells: list[BenchCell],
    *,
    backend: str | None = None,
    device: str | None = None,
    model_substr: str | None = None,
) -> list[BenchCell]:
    out = cells
    if backend:
        out = [c for c in out if c.backend == backend]
    if device:
        out = [c for c in out if c.device == device]
    if model_substr:
        out = [c for c in out if model_substr.lower() in c.model.lower()]
    return out
