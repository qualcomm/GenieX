# tools/runtime_benchmark

Compares answer **quality** and **performance** (TTFT, decode tok/s) between
two runtimes on the same model:

- `geniex` — this project's runtime, driven in-process through its Python
  API (`AutoModelForCausalLM`), so the model is loaded once and reused.
- `genie-t2t-run` — Qualcomm QAIRT SDK reference runner (still a subprocess;
  it has no Python API).

Same model, same prompts, same chat template — so quality differences reflect
the runtime, not the inputs.

> **Requires `geniex >= 0.2.3rc3`.** Earlier wheels defaulted
> `apply_chat_template(enable_thinking=False)`, which injects an empty
> `<think>\n\n</think>\n\n` block on non-thinking models (e.g.
> Qwen3-Instruct-2507) and produces nonsense answers ("riddle / Фропт"
> instead of describing gravity, etc.). 0.2.3rc3 auto-detects
> `model.supports_thinking` from the model's chat template and skips the
> suppression block when the model has no thinking mode. See
> [RUN_ON_QDC.md § Prerequisites](RUN_ON_QDC.md#prerequisites-on-the-qdc-machine).

## Workflow

The benchmark splits across two machines:

1. **On a Qualcomm QDC (Snapdragon X Elite) device** — run the prompt
   suite, collect raw answers. See [RUN_ON_QDC.md](RUN_ON_QDC.md).
2. **On the local dev machine, in Claude Code** — score the answers
   against the rubric, emit the final CSV. See
   [SCORE_WITH_CLAUDE.md](SCORE_WITH_CLAUDE.md).

When **only geniex code has changed** (model unchanged), pass
`--skip-genie` on step 1 to skip the genie-t2t-run pass — genie's
answers for a given model are deterministic, so you can reuse the
genie answers from the prior full run. The script writes the new
geniex-only results to `results/<slug>_geniex_only.csv`, and the
scoring agent merges them with the prior `<slug>.answers.json` to
produce a fresh scored CSV. See
[RUN_ON_QDC.md § Geniex-only re-runs](RUN_ON_QDC.md#geniex-only-re-runs)
and
[SCORE_WITH_CLAUDE.md § Shape B](SCORE_WITH_CLAUDE.md#shape-b--geniex-only-re-run-paired-with-a-prior-full-run).

## Files

```
runtime_quality_benchmark.py   # collection script (runs on QDC)
testing_prompts.md             # 100-prompt suite, grouped by category
scoring_rubric.txt             # 0/2/7/10 rubric reference
results/                       # all generated artifacts
RUN_ON_QDC.md                  # how to do step 1
SCORE_WITH_CLAUDE.md           # how to do step 2
```

`results/` already contains finished runs for `Qwen3-4B-Instruct-2507`
and `Llama-v3.2-1B-Instruct` — useful as calibration references.

## Running genie-t2t-run by hand

The benchmark drives `genie-t2t-run` for you. To invoke it directly — e.g. to
sanity-check the QAIRT install or inspect a `--profile` file — set up the
environment and run it from the model directory:

```powershell
$env:QAIRT_HOME = "C:\Qualcomm\AIStack\QAIRT\2.45.0.260326"
$env:Path = "$env:QAIRT_HOME\bin\aarch64-windows-msvc;" + $env:Path
$env:Path = "$env:QAIRT_HOME\lib\aarch64-windows-msvc;" + $env:Path

# Please make sure the architecture matches that of the device
# (v73 for X Elite, v81 for X2 Elite)
$env:ADSP_LIBRARY_PATH = "$env:QAIRT_HOME\lib\hexagon-v73\unsigned"

cd "C:\Users\yichqian\code\geniex-qairt-plugin\modelfiles\qwen2_5_vl_7b_instruct"
genie-t2t-run.exe -c genie_config.json -p "<|im_start|>system\nYou are a helpful AI Assistant.<|im_end|><|im_start|>What is France's capital?\n<|im_end|>\n<|im_start|>assistant\n" --profile profile.log
```

`--profile <file>` writes a `GENIE_PROFILE` JSON file with per-query KPIs
(`time-to-first-token`, `token-generation-rate`, etc.). The benchmark uses
this to record genie's TTFT / TPS; see [RUN_ON_QDC.md](RUN_ON_QDC.md).
