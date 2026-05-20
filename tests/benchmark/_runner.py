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

"""Per-cell benchmark loop: warmup + N measured runs + aggregation."""

from __future__ import annotations

import statistics
from typing import Any

import geniex
from _matrix import BenchCell

# Static prompt — picked so most tokenisers emit roughly 40-60 tokens, which
# is enough to make prefill_speed meaningful but small enough to keep cell
# time bounded. The real prompt_tokens is recorded in each run.
BENCH_PROMPT = (
    'Explain in three short sentences why the speed of light is the same '
    'in every inertial reference frame, and what that implies for time '
    'dilation. Keep it accessible to a curious teenager.'
)

# Fixed generation knobs so cross-backend numbers are comparable.
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0
SEED = 42


def _aggregate(values: list[float]) -> dict[str, float]:
    if not values:
        return {'median': 0.0, 'min': 0.0, 'max': 0.0}
    return {
        'median': statistics.median(values),
        'min': min(values),
        'max': max(values),
    }


def _agg_int(values: list[int]) -> dict[str, int]:
    if not values:
        return {'median': 0}
    return {'median': int(statistics.median(values))}


def run_cell(
    cell: BenchCell,
    *,
    warmup: int = 1,
    repeat: int = 3,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Run one (model, backend, device) cell and return a result dict.

    The caller is responsible for ``geniex.init()`` / ``geniex.deinit()``;
    we only own the per-cell model handle.
    """
    base = {
        'model': cell.model,
        'quant': cell.quant,
        'backend': cell.backend,
        'device': cell.device,
        'cell_id': cell.cell_id,
        'status': 'error',
        'skip_reason': None,
        'error': None,
        'runs': [],
        'agg': {},
    }

    runs: list[dict[str, Any]] = []
    try:
        kwargs: dict[str, Any] = {'device_map': cell.device}
        if cell.quant is not None:
            kwargs['quant'] = cell.quant
        with geniex.AutoModelForCausalLM.from_pretrained(cell.model, **kwargs) as llm:
            for i in range(max(0, warmup)):
                # Suffix the prompt per-iteration so the llama.cpp KV cache
                # doesn't short-circuit prefill on the second+ identical run.
                # Without this, prefill_speed becomes infinite (prompt_time=0).
                llm.generate(
                    f'{BENCH_PROMPT}\n[warmup={i}]',
                    max_new_tokens=max_new_tokens,
                    temperature=TEMPERATURE,
                    seed=SEED,
                )
            for i in range(repeat):
                out = llm.generate(
                    f'{BENCH_PROMPT}\n[run={i}]',
                    max_new_tokens=max_new_tokens,
                    temperature=TEMPERATURE,
                    seed=SEED,
                )
                p = out.profile
                runs.append(
                    {
                        'run_idx': i,
                        'ttft_us': p.ttft,
                        'prompt_tokens': p.prompt_tokens,
                        'gen_tokens': p.generated_tokens,
                        'prefill_tps': p.prefill_speed,
                        'decode_tps': p.decode_speed,
                        'prompt_time_us': p.prompt_time,
                        'decode_time_us': p.decode_time,
                        'stop_reason': p.stop_reason,
                    }
                )
    except Exception as e:  # noqa: BLE001 — record any failure as cell-level error
        base['error'] = f'{type(e).__name__}: {e}'
        base['runs'] = runs
        return base

    base['runs'] = runs
    base['status'] = 'ok'
    base['agg'] = {
        'ttft_ms': _aggregate([r['ttft_us'] / 1000.0 for r in runs]),
        'prefill_tps': _aggregate([r['prefill_tps'] for r in runs]),
        'decode_tps': _aggregate([r['decode_tps'] for r in runs]),
        'gen_tokens': _agg_int([r['gen_tokens'] for r in runs]),
        'prompt_tokens': _agg_int([r['prompt_tokens'] for r in runs]),
    }
    return base


def skipped_cell(cell: BenchCell, reason: str) -> dict[str, Any]:
    return {
        'model': cell.model,
        'quant': cell.quant,
        'backend': cell.backend,
        'device': cell.device,
        'cell_id': cell.cell_id,
        'status': 'skipped',
        'skip_reason': reason,
        'error': None,
        'runs': [],
        'agg': {},
    }
