# Running the runtime-quality benchmark on a Qualcomm QDC device

This is the **collection** half of the workflow. You run the prompt suite
through both runtimes on a Snapdragon device (the QDC), produce a
results CSV + an answers JSON, then download those two files back to your
local machine for scoring (see [SCORE_WITH_CLAUDE.md](SCORE_WITH_CLAUDE.md)).

The script does no scoring; it just collects raw answers, so the QDC
session can be entirely non-interactive once kicked off.

## Prerequisites on the QDC machine

1. `geniex` CLI installed and on `PATH`
   (sanity check: `geniex version`).
2. QAIRT SDK installed; `genie-t2t-run` on `PATH`
   (sanity check: `genie-t2t-run --help`).
3. Python 3.10+ (no third-party packages required — stdlib only).
4. The model you want to benchmark cached in geniex.
   - For Hugging Face / AI Hub models: `geniex pull <name>`.
   - For a model file tree on disk (e.g., the `geniex-qairt-plugin`
     `modelfiles/...` directory): drop a `geniex.json` manifest into the
     folder, then `geniex pull <name> --model-hub localfs --local-path <path>`.
     See `geniex-template.json` at the repo root for the manifest shape.
5. This `tools/runtime_benchmark/` directory copied or cloned to the QDC.

## One-shot run

From `tools/runtime_benchmark/`:

```bash
PYTHONIOENCODING=utf-8 python runtime_quality_benchmark.py \
    --geniex-model qualcomm/Qwen3-4B-Instruct-2507
```

That's the whole command. The script will:

- Locate the cached model under `~/.cache/geniex/models/<name>/`
  (or `$GENIEX_DATADIR/models/<name>/` if set) and read its
  `genie_config.json` to pick the chat template (Llama-3 vs Qwen).
- Iterate every prompt in `testing_prompts.md`.
- Run each prompt through `geniex infer` and `genie-t2t-run` in turn.
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
| `--geniex-model`      | _(required)_                               | The name `geniex list` shows for the model.                            |
| `--prompts`           | `testing_prompts.md` next to the script    | Different prompt suite. Same markdown shape as the bundled file.       |
| `--genie-config-dir`  | auto-discovered from the geniex cache      | Override only if the model isn't a standard `geniex pull` cache entry. |
| `--out`               | `results/<slug>.csv`                       | Override the CSV path (the answers.json is always derived from it).    |
| `--max-tokens`        | 512                                        | Cap on tokens per prompt (geniex side; genie honors EOS via template). |
| `--timeout`           | 600                                        | Per-prompt per-runtime timeout in seconds.                             |
| `--limit N`           | 0 (all)                                    | Run only the first N prompts — handy for smoke tests.                  |
| `--resume`            | off                                        | Skip prompts whose `id` already appears in `--out`. Use to continue.   |

## Resumption

Because the CSV is rewritten after every prompt, an interrupted run can
just be restarted with the same arguments plus `--resume`:

```bash
PYTHONIOENCODING=utf-8 python runtime_quality_benchmark.py \
    --geniex-model qualcomm/Qwen3-4B-Instruct-2507 \
    --resume
```

The first time the answers.json is written is when the run finishes, so
if you crash mid-run the CSV is the source of truth — re-running with
`--resume` will fill in the missing rows and emit a fresh answers.json
once everything is done.

## Files to download back to the local machine

After a successful run, only two files are needed for scoring:

```
results/<slug>.csv               # full 12-column row data (incl. timing, errors)
results/<slug>.answers.json      # 5-field-per-row payload the scoring agent reads
```

The `.csv` is the one you'll edit during scoring (see SCORE_WITH_CLAUDE.md);
the `.answers.json` is what the agent ingests, so keep both around.

## Troubleshooting

- **`could not locate genie_config.json`** — the model isn't in the cache
  the script looked under. Run `geniex list` to confirm the cache entry
  exists, or pass `--genie-config-dir` pointing at the directory that
  contains `genie_config.json` and the `*.bin` shards.
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
