# Running the runtime-quality benchmark on a Qualcomm QDC device

This is the **collection** half of the workflow. You run the prompt suite
through both runtimes on a Snapdragon device (the QDC), produce a
results CSV + an answers JSON, then download those two files back to your
local machine for scoring (see [SCORE_WITH_CLAUDE.md](SCORE_WITH_CLAUDE.md)).

The script does no scoring; it just collects raw answers, so the QDC
session can be entirely non-interactive once kicked off.

## Prerequisites on the QDC machine

1. The `geniex` **Python package** importable (sanity check:
   `python -c "import geniex"`). The geniex side of the benchmark runs
   **in-process** via this package's API — the `geniex` CLI is **not**
   required (not for inference, and not for pulling models; see step 4).

```powershell

pip install -U -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple geniex==0.2.1rc2

```

2. QAIRT SDK installed; `genie-t2t-run` on `PATH`
   (sanity check: `genie-t2t-run --help`).
3. Python 3.10+. Besides the `geniex` package, no third-party packages are
   required — stdlib only.
4. The model you want to benchmark cached in geniex. Pull it via the Python
   API (same cache the script reads from) — no CLI needed:
   - For Hugging Face / AI Hub models:
     ```bash
     python -c "from geniex import model_manager; model_manager.pull('qualcomm/Qwen3-4B-Instruct-2507')"
     ```
   - For a model file tree on disk (e.g., the `geniex-qairt-plugin`
     `modelfiles/...` directory): drop a `geniex.json` manifest into the
     folder, then pull from the local filesystem hub:
     ```bash
     python -c "from geniex import model_manager; model_manager.pull('<name>', hub='localfs', local_path=r'<path>')"
     ```
     See `geniex-template.json` at the repo root for the manifest shape.
5. This `tools/runtime_benchmark/` directory copied or cloned to the QDC.

## One-shot run

**Pull the model into the geniex cache first** — the script does not pull for
you. It locates the model (and reads its `genie_config.json` for the genie
side) by looking it up in the cache, so an un-pulled model fails fast with
`could not locate genie_config.json for <name>`. Pull via the Python API
(no CLI needed — see prerequisite 4):

```powershell
# Hugging Face / AI Hub model:
python -c "from geniex import model_manager; model_manager.pull('qualcomm/Qwen3-4B-Instruct-2507')"

# …or a local model tree:
python -c "from geniex import model_manager; model_manager.pull('<name>', hub='localfs', local_path=r'<path>')"
```

Then, from `tools/runtime_benchmark/`:

```powershell
$env:PYTHONIOENCODING="utf-8" 
python runtime_quality_benchmark.py --geniex-model qualcomm/Qwen3-4B-Instruct-2507
```

That's the whole command. The script will:

- Locate the cached model under `~/.cache/geniex/models/<name>/`
  (or `$GENIEX_DATADIR/models/<name>/` if set) and read its
  `genie_config.json` to pick the chat template (Llama-3 vs Qwen).
- Load the geniex model once (via `AutoModelForCausalLM.from_pretrained`)
  and reuse it across prompts, resetting the KV cache between each.
- Iterate every prompt in `testing_prompts.md`.
- Run each prompt through the in-process geniex API (applying the model's
  own chat template, like the Go CLI `infer`) and the `genie-t2t-run`
  subprocess (fed a hand-built prompt) in turn.
- Persist the running results to `results/<slug>.csv` after **every**
  prompt so a crashed or interrupted run can be resumed.
- After the last prompt, also write `results/<slug>.answers.json` —
  the file the scoring agent reads.

`<slug>` is the model's basename, lowercased, with `-` and `.` turned
into `_`. Examples:

| `--geniex-model`                      | `<slug>`                  |
| ------------------------------------- | ------------------------- |
| `qualcomm/Qwen3-4B-Instruct-2507`     | `qwen3_4b_instruct_2507`  |
| `qualcomm/Llama-v3.2-1B-Instruct`     | `llama_v3_2_1b_instruct`  |

Expected wall-clock: ~30 minutes for a 1B model, ~2 hours for a 4B model
on Snapdragon X Elite (NPU). Both runtimes generate up to 512 tokens
per prompt; the prompt suite has 100 prompts.

## Useful flags

| flag                  | default                                    | what it does                                                           |
| --------------------- | ------------------------------------------ | ---------------------------------------------------------------------- |
| `--geniex-model`      | _(required)_                               | The cached model name (as passed to `model_manager.pull`).             |
| `--prompts`           | `testing_prompts.md` next to the script    | Different prompt suite. Same markdown shape as the bundled file.       |
| `--genie-config-dir`  | auto-discovered from the geniex cache      | Override only if the model isn't in the standard geniex cache layout.  |
| `--device-map`        | `auto`                                     | `device_map` for the geniex Python API: `auto`/`cpu`/`gpu`/`npu`/`hybrid`/`<plugin>:<device>`. |
| `--out`               | `results/<slug>.csv`                       | Override the CSV path (the answers.json is always derived from it).    |
| `--max-tokens`        | 512                                        | Cap on tokens per prompt (geniex side; genie honors EOS via template). |
| `--timeout`           | 600                                        | Per-prompt timeout in seconds for the `genie-t2t-run` subprocess.      |
| `--limit N`           | 0 (all)                                    | Run only the first N prompts — handy for smoke tests.                  |
| `--resume`            | off                                        | Skip prompts whose `id` already appears in `--out`. Use to continue.   |

## Resumption

Because the CSV is rewritten after every prompt, an interrupted run can
just be restarted with the same arguments plus `--resume`:

```powershell
$env:PYTHONIOENCODING="utf-8" 
python runtime_quality_benchmark.py --geniex-model qualcomm/Qwen3-4B-Instruct-2507 --resume
```

The first time the answers.json is written is when the run finishes, so
if you crash mid-run the CSV is the source of truth — re-running with
`--resume` will fill in the missing rows and emit a fresh answers.json
once everything is done.

## Files to download back to the local machine

After a successful run, only two files are needed for scoring:

```
results/<slug>.csv               # full row data (incl. timing, perf, errors)
results/<slug>.answers.json      # 5-field-per-row payload the scoring agent reads
```

The `.csv` is the one you'll edit during scoring (see SCORE_WITH_CLAUDE.md);
the `.answers.json` is what the agent ingests, so keep both around.

## Performance columns

Besides answers, the CSV records two perf metrics per runtime, per prompt:

| column            | meaning                                                        |
| ----------------- | -------------------------------------------------------------- |
| `*_ttft_ms`       | time to first token, in milliseconds                           |
| `*_tps`           | decode throughput, in tokens/sec                               |

- **geniex** values come straight from the API's `ProfileData` (`ttft`,
  `decode_speed`).
- **genie** values come from the JSON file genie-t2t-run writes via
  `--profile <file>`. The script passes a fresh temp file per prompt and
  reads the `GenieDialog_query` event's `time-to-first-token` (→ ms) and
  `token-generation-rate` (toks/sec). If the profile file is missing or
  malformed the cells are left blank rather than guessed.

These are wall-clock-independent (the `*_seconds` columns still record total
per-prompt wall time, which for genie includes process startup).

## Troubleshooting

- **`could not locate genie_config.json`** — the model isn't in the cache
  the script looked under. Most often this means you skipped the pull step:
  pull it via the Python API (see "One-shot run" above), then confirm the
  cache entry with
  `python -c "from geniex import model_manager; print(model_manager.list_models())"`.
  If it's cached under a non-standard path, pass `--genie-config-dir`
  pointing at the directory that contains `genie_config.json` and the
  `*.bin` shards.
- **`ImportError` / `the geniex Python package is not importable`** — the
  CLI is installed but the Python package isn't on this interpreter's
  `sys.path`. Install it (`pip install geniex`) into the same Python you're
  running the script with.
- **`SDKError(Invalid input parameters or handle)`** from geniex — the
  cached context binaries were built with a different QAIRT version than
  the geniex CLI is linked against. `genie-t2t-run` is more lenient
  about version skew, so this typically shows up as geniex-only failures.
  Pull a fresher build of the model, or run on a QDC image with a
  matching QAIRT.
- **Garbled output starting with `<|im_start|>` or `<|begin_of_text|>`** —
  the chat template detection failed. The script keys off the
  `dialog.context.bos-token` field of `genie_config.json` (128000 → llama3,
  anything else → qwen). If a new model family ships, add it to
  `TEMPLATES` / `detect_template` at the top of the script.
- **UnicodeEncodeError on Windows** — the `PYTHONIOENCODING=utf-8` env
  var on the example commands is load-bearing; without it Python's
  default cp1252 stdout will choke on emoji/CJK in model output.
