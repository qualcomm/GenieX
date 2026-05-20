# Benchmark suite

Local-first inference performance harness covering the same model under
`qairt:npu` and `llama_cpp:{cpu,gpu,npu,hybrid}`. Designed to run by hand
today and to be picked up by a Snapdragon CI runner later — the JSON output
is the stable contract.

## Layout

```
tests/benchmark/
├── _matrix.py          # cell list + host-capability gating
├── _runner.py          # warmup + N runs + median aggregation
├── _report.py          # JSON + Markdown + stdout
├── run.py              # `python -m tests.benchmark.run` entry
├── test_benchmark.py   # pytest smoke wrapper (-m benchmark)
└── results/            # output dir (gitignored)
```

## Matrix

Reuses the starter models declared in [tests/_models.py](../_models.py):

| Backend    | Device  | Model                                |
|------------|---------|--------------------------------------|
| llama_cpp  | cpu     | `LLAMA_CPP_LLM_MODEL` (`Q4_0`)       |
| llama_cpp  | gpu     | same                                 |
| llama_cpp  | npu     | same                                 |
| llama_cpp  | hybrid  | same                                 |
| qairt      | npu     | `QAIRT_LLM_MODEL`                    |

Override the QAIRT id with `GENIEX_QAIRT_MODEL=...` (see `tests/_models.py`).

Cells that need Snapdragon hardware (`gpu`, `npu`, `hybrid`, all `qairt`)
record `status: skipped` on non-Snapdragon hosts instead of failing.

## Run it

```bash
# Anywhere — only the llama_cpp cpu cell actually runs
python -m tests.benchmark.run --backend llama_cpp --device cpu

# Snapdragon Windows ARM64 — full matrix (QAIRT model must be cached)
python -m tests.benchmark.run

# Tune samples
python -m tests.benchmark.run --warmup 2 --repeat 5

# Pytest entry (uses `geniex_session` fixture from tests/conftest.py)
pytest tests -m benchmark
```

QAIRT models are not auto-pulled. Populate the cache once with
`geniex-py pull <model>` before running QAIRT cells.

## Per-cell measurement

- Fixed prompt (declared in [_runner.py](_runner.py))
- `max_new_tokens=128`, `temperature=0.0`, `seed=42`
- 1 warmup run (discarded) + N measured runs (default 3)
- Metrics read directly from `GenerateOutput.profile` — no external timing

Reported per cell: median / min / max of `ttft_ms`, `prefill_tps`,
`decode_tps`, plus median `gen_tokens` and `prompt_tokens`. Every individual
run's full profile is preserved under `cells[].runs[]`.

## Output

Each invocation writes two files into `results/` (gitignored):

- `<UTC-timestamp>-<git-sha>.json` — `schema_version: 1`, the long-term
  contract for the planned trend-chart runner
- `<UTC-timestamp>-<git-sha>.md` — per-model human-readable table

JSON shape:

```json
{
  "schema_version": 1,
  "generated_at": "2026-05-20T12:34:56+00:00",
  "git_sha": "abc1234",
  "geniex_version": "0.x.y",
  "host": {"os": "Windows", "machine": "ARM64", "snapdragon": true, "python": "3.12.0"},
  "params": {"warmup": 1, "repeat": 3, "max_new_tokens": 128, "temperature": 0.0, "seed": 42},
  "cells": [
    {
      "model": "...", "quant": "Q4_0", "backend": "llama_cpp", "device": "cpu",
      "cell_id": "...-llama_cpp-cpu",
      "status": "ok",
      "skip_reason": null,
      "error": null,
      "agg": {
        "ttft_ms":      {"median": ..., "min": ..., "max": ...},
        "prefill_tps":  {"median": ..., "min": ..., "max": ...},
        "decode_tps":   {"median": ..., "min": ..., "max": ...},
        "gen_tokens":   {"median": 128},
        "prompt_tokens": {"median": ...}
      },
      "runs": [{"run_idx": 0, "ttft_us": ..., "prefill_tps": ..., ...}]
    }
  ]
}
```

Bump `schema_version` on any breaking shape change and note it here.

## Out of scope (for the first cut)

- CI integration (handed off to the future Snapdragon runner)
- VLM benchmark
- Multi-prompt-length sweep, historical trend charts, memory / power capture
