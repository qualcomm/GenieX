# Geniex Bench

`geniex-bench` runs on real Qualcomm hardware via QDC (Qualcomm Device Cloud),
driven by two workflows that share the artifact build and the QDC plumbing and
report into the GitHub Actions step summary:

| Workflow | Measures | Devices | Result |
|----------|----------|---------|--------|
| [bench.yml](../.github/workflows/bench.yml) | speed — TTFT, prefill / decode tok/s | Linux, Windows, Android | markdown table |
| [accuracy.yml](../.github/workflows/accuracy.yml) | output quality | Linux, Windows, Android | Breeze AI ratings |

## Shared pipeline

### Artifact compilation & upload

The SDK is cross-compiled for 3 platforms via `_build-sdk.yml`, producing a
`geniex-bench` binary for each:

| Platform       | Target Device                       | Binary             | Test Framework     |
|----------------|-------------------------------------|--------------------|--------------------|
| Linux ARM64    | QCS9075M (IoT)                      | `geniex-bench`     | Bash               |
| Windows ARM64  | SC8380XP / SC8480XP (Snapdragon X)  | `geniex-bench.exe` | PowerShell         |
| Android ARM64  | SM8750 / SM8850 (phone)             | `geniex-bench`     | Appium + pytest    |

Each platform gets a zip artifact assembled by `build_*_artifact()` in
`sdk/benchmark/qdc/run_qdc_jobs.py`:

- **Linux / Windows** — pre-built SDK package + entry script
  (`run_linux.sh` or `run_windows.ps1`) + sample prompts + VLM test image. The
  host substitutes the model rows, the sweep and `{MODE}` (`bench` / `accuracy`)
  into the script.
- **Android** — SDK package + pytest suite (`test_bench.py`, `utils.py`) +
  matrix rows + prompts. There is no script to substitute into, so the same
  values travel in `params.json` and pytest skips whichever of `test_bench` /
  `test_accuracy` the mode did not ask for.

The model matrix is defined in `sdk/benchmark/qdc/bench-models.json`
(models across `llama_cpp` and `qairt` plugins).

`--mode` picks the leg: `bench` / `accuracy` run on a device; the rest run on
the host over the artifacts a device leg produced, prefixed by which pipeline
they belong to -- `bench_aggregate`, `accuracy_payload` / `accuracy_report`.

### QDC job execution

QDC interaction lives in `sdk/benchmark/qdc/_qdc.py`:

| Function             | Purpose                                                                 |
|----------------------|-------------------------------------------------------------------------|
| `make_client`        | Creates an authenticated QDC API client (`QDC_API_KEY`, app=`geniex-ci`)|
| `resolve_target`     | Maps chipset name (e.g. `SM8850`) to a QDC target ID                    |
| `submit_and_wait`    | Uploads artifact zip, submits job, polls every 30 s until terminal state. Quota-aware retry with exponential backoff (30 s base, 1 hr budget) |

## bench.yml — speed

```
build-sdk  -->  load-models  -->  bench (device x model matrix)  -->  aggregate (per device)
```

On-device, all platforms perform the same work:

1. Build 3 TSV matrix files (context sizes 512, 1024, 4096) from the model rows.
2. Invoke `geniex-bench --matrix-file <tsv> --output-json-dir <out> --chipset <chip>`.
3. The benchmark binary ([`sdk/benchmark/`](../sdk/benchmark/), module layout in
   [`bench.h`](../sdk/benchmark/bench.h)) runs each cell:
   1 warmup + 3 measured repetitions, writing a per-cell JSON with aggregated
   stats (median / stdev / min / max for TTFT, prefill tok/s, decode tok/s;
   median-only for `media_ms`, non-zero on VLM cells).
   For VLM cells `prefill_tps` is the full prefill (text + media tokens); only
   the encoder time is split out into `media_ms` — see
   [run.md § Performance metrics](run.md#performance-metrics).
4. Results land in `QDC_logs/results/` which QDC auto-collects.

Orchestrated in 4 jobs:

1. **build-sdk** — cross-compile binaries for all 3 platforms.
2. **load-models** — parse `bench-models.json`, emit the matrix + device list.
3. **bench** — one job per (device, model) pair: submit to QDC, wait, download
   per-cell JSON.  Uploaded as artifact `bench-cells-{device}-{model}`.
4. **aggregate** — one job per device: download all matching cell artifacts, call
   `render_aggregate()` which flattens cells, builds a markdown table, and writes
   it to `$GITHUB_STEP_SUMMARY`.

Example output:

```
## QDC Bench -- SM8850

| Model       | Backend   | Device | Ctx | ngl | Test       | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |
|-------------|-----------|--------|----:|----:|------------|----------:|----------------:|---------------:|
| Qwen3-0.6B | llama_cpp | cpu    | 512 |   - | pp42+tg128 | 49.8 +/-2.4 | 102.1 +/-6.2 | 60.9 +/-2.3    |
```

## accuracy.yml — quality

```
build-sdk  -->  load-models  -->  generate (device x model)  -->  grade (Breeze AI)
```

On-device, one pass replaces the sweep, over the same three entry points as
`bench.yml`:

1. One TSV matrix file for every cell (both plugins share the prompt file).
2. `geniex-bench --matrix-file <tsv> --accuracy --prompt-file
   prompts/accuracy_prompts.txt [--think|--no-think]`.
3. Each prompt (segments split on lines that are exactly `---`) runs once
   through the chat template; the response prints as `[gen ]` lines on
   stdout, the only place it exists.
4. The host collects that stdout as a `.log`: direct redirect on Linux,
   `Start-Transcript` on Windows (UTF-16 with a BOM, so `decode_log()` sniffs
   it), and an explicit adb push back on Android (pytest's own stdout isn't
   collected).

Orchestrated in 4 jobs, the first two identical to `bench.yml`:

1. **build-sdk** — cross-compile binaries for all 3 platforms.
2. **load-models** — parse `bench-models.json`, emit the matrix + device list.
3. **generate** — one QDC job per (device, model) cell, though a model with
   more than one compute unit (`--compute` left blank, or given more than one
   value) runs one geniex-bench cell per unit and the job's log carries all of
   them. `--mode accuracy` parses that log into `items.json`, uploaded as
   `accuracy-items-{device}-{model}`. Its report label is `{model}-{device}`,
   not the device-side `cell_id` (which also carries plugin/compute/ctx) —
   those are either implied by the model name or fixed for the whole run, and
   dropping them is what keeps the same model on two chipsets from colliding
   into one row once cells are merged for grading. A `-{compute}` suffix comes
   back only when a job actually produced more than one compute unit's cells,
   to keep those from colliding with each other. `--compute` pins the unit
   outright (bench's sweep excludes `hybrid`); `--prompt-limit N` trims the
   set; ctx is derived as `max(2*tg, 4096)` rather than taken from the timing
   sweep, which would truncate a long answer.
4. **grade** — `--mode accuracy_payload` merges every cell into one globally
   numbered payload
   ([`prompts/grade-rules.md`](../sdk/benchmark/qdc/prompts/grade-rules.md) + the
   items, cell withheld so the grader cannot anchor on the model). Breeze AI
   only exists inside the Qualcomm network, so this round-trips through an
   issue in `qcom-ai-hub/geniex`: the agent downloads the payload and
   comments one row per item (`Item | Rating | Category | Note`), which
   `--mode accuracy_report` turns into the summary below.

100 items do not fit in a step summary, so the report is two aggregates --
scores per cell, then what the deductions were for -- both cell rows x
category/band columns; a perfect 10 (`none`) never appears in the second.
Full detail rides artifacts instead of a third table: `accuracy-items-*` has
the raw items, `grade-item-map.md` is the same prompts/responses tagged with
cell (linked from the summary), and `breeze-grades` is the grader's unparsed
comment (also linked). `grade-payload.md` — what the grader itself downloads,
cell withheld — is uploaded but not linked; it's not useful to a human.

```
| Cell                    | Items | Mean | 0-3 | 4-6 | 7-9 | 10 |
|-------------------------|------:|-----:|----:|----:|----:|---:|
| Qwen3-1.7B-QCS9075M     |   100 |  7.4 |   6 |  18 |  61 | 15 |

| Cell                    | `loop` | `duplicate-word` | `factual` |
|-------------------------|------:|------:|------:|
| Qwen3-1.7B-QCS9075M     |     4 |    12 |     - |
```

The `Category` vocabulary is closed — see
[`prompts/grade-rules.md`](../sdk/benchmark/qdc/prompts/grade-rules.md); anything
off-list is counted as `other`, columns are sorted by total count, and a cell
with no hits for a category shows `-`.

## Key design choices

- **Matrix-driven** — model x device pairs run in parallel on QDC.
- **Platform isolation** — differences are confined to artifact-building and entry
  scripts; the on-device benchmark binary and JSON schema are shared.
- **Common per-cell JSON schema** — every platform produces the same schema
  (`schema_version` `5`), so the aggregator renders uniformly regardless of
  origin. v4 added the `media_us` per-run encoder time and its `media_ms` agg
  median. On a VLM run `prompt_tokens` counts text + media tokens, so
  `prefill_tps` reflects the full prefill. v5 added the optional `power_mode`
  param (set via `--power-mode`; omitted when unset).

## Downloading geniex-bench

Standalone `geniex-bench` archives (binary + runtime libs) are published to S3
on each release. Use the **latest** URLs to always get the current stable build,
or pin to a specific version tag.

### Latest stable (always points to the newest non-prerelease)

| Platform      | URL |
|---------------|-----|
| Linux ARM64   | https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/geniex-bench-linux-arm64.tar.gz |
| Android ARM64 | https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/geniex-bench-android-arm64.tar.gz |
| Windows ARM64 | https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/geniex-bench-windows-arm64.zip |

### Versioned (pinned to a specific release)

Replace `<tag>` with the release tag (e.g. `v1.2.3`):

| Platform      | URL |
|---------------|-----|
| Linux ARM64   | `https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/geniex-bench-linux-arm64-<tag>.tar.gz` |
| Android ARM64 | `https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/geniex-bench-android-arm64-<tag>.tar.gz` |
| Windows ARM64 | `https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/geniex-bench-windows-arm64-<tag>.zip` |

### Quick start

```bash
# Linux / Android (same binary format, different target device)
curl -fsSL https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/geniex-bench-linux-arm64.tar.gz | tar xz
./geniex-bench-linux-arm64-*/bin/geniex-bench --help

# Windows (PowerShell)
Invoke-WebRequest https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/geniex-bench-windows-arm64.zip -OutFile bench.zip
Expand-Archive bench.zip -DestinationPath .
.\geniex-bench-windows-arm64-*\bin\geniex-bench.exe --help
```

Each archive contains `bin/geniex-bench` (or `.exe`) plus `lib/` with all
required runtime shared libraries (libgeniex, llama_cpp plugin, qairt plugin).

### Raw logits mode (`--logits`)

For on-target accuracy metrics (perplexity, MMLU, MMMU), `--logits` runs a
single prefill-only forward pass (`geniex_llm_forward_logits`, no decode loop)
over `-p N` random token ids and writes every position's logits row
(`[n_tokens, vocab]`) to the JSON report. Bypasses the timing machinery
entirely (`--warmup` / `-r` / `-n` are ignored).

```bash
geniex-bench --plugin llama_cpp --device npu -m <model> --logits -p 128 \
  --logits-top-n 20 --output-json logits.json
```

The report keeps only the top-N `[token_id, logit]` pairs per row
(`--logits-top-n`, default 20) so the all-positions output stays small; the JSON
records `top_n` and `truncated_to_top_n` so a consumer never mistakes it for
the full vocabulary. Input is random ids only — the forward-logits API takes
`input_ids` and the bench tool has no tokenizer, so `--prompt-file` is rejected
with `--logits`. Both `llama_cpp` and `qairt` backends support it.

The JSON report (`schema_version` `logits-1`) carries shape metadata plus
`rows`, one row per emitted position, each a top-N array of `[token_id, logit]`
pairs sorted by descending logit:

```json
{
  "schema_version": "logits-1",
  "n_prompt": 128,
  "all_positions": true,
  "n_rows": 128,
  "vocab_size": 151936,
  "top_n": 20,
  "truncated_to_top_n": true,
  "rows": [
    [[9, 6.950917], [1479, 6.472050], ...],
    ...
  ]
}
```
