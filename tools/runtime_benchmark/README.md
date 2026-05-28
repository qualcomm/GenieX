# tools/runtime_benchmark

Compares answer quality between two runtimes on the same model:

- `geniex` — this project's runtime, driven in-process through its Python
  API (`AutoModelForCausalLM`), so the model is loaded once and reused.
- `genie-t2t-run` — Qualcomm QAIRT SDK reference runner (still a subprocess;
  it has no Python API).

## Workflow

The benchmark splits across two machines:

1. **On a Qualcomm QDC (Snapdragon X Elite) device** — run the prompt
   suite, collect raw answers. See [RUN_ON_QDC.md](RUN_ON_QDC.md).
2. **On the local dev machine, in Claude Code** — score the answers
   against the rubric, emit the final CSV. See
   [SCORE_WITH_CLAUDE.md](SCORE_WITH_CLAUDE.md).

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

## How to run genie-t2t-run locally

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
