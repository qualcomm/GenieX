# SDK pytest suite

End-to-end tests for the geniex SDK, driven through the public Python
binding so the suite doubles as a public-surface contract check.

```
tests/
├── api/                   # SDK metadata APIs (no model required)
├── assets/                # Real test images (quality_dog.jpg + NOTICE.md)
├── plugins/
│   ├── llama_cpp/         # LLM + VLM × {cpu, npu, hybrid}
│   └── qairt/             # LLM + VLM × {npu}
├── cli/                   # Bazel-driven Go CLI black-box tests (separate)
├── conftest.py            # Top-level fixtures (init, model paths, image)
├── pytest.ini             # Marker registry + suite discovery
├── _models.py             # Model identifiers used by the matrix
└── _quality_data.py       # Keyword-quality prompts shared by both plugins
```

The Go CLI tests under `tests/cli/` are excluded from pytest discovery
(`norecursedirs = cli`) and continue to run via Bazel.

## Matrix

| Plugin    | Device      | LLM | VLM |
|-----------|-------------|-----|-----|
| llama_cpp | `cpu`       | ✅  | ✅  |
| llama_cpp | `npu`       | ✅  | ✅  |
| llama_cpp | `hybrid`    | ✅  | ✅  |
| qairt     | `npu`       | ✅  | ✅  |

= 8 generation cells, plus the API subset. The `gpu` (GPUOpenCL/Adreno)
alias is excluded — it isn't in the production scorecard matrix either.

Each cell carries two layers of coverage:

- **Smoke** (`test_generate_*`) — model loads, `out.text` non-empty,
  `generated_tokens > 0`. Cheap, runs on every cell.
- **Keyword quality** (`test_quality_keywords`) — mirrors the LLM and VLM
  keyword checks in upstream test-llama.cpp's QDC scorecard
  (`scripts/snapdragon/qdc/tests/run_scorecard_posix.py`). Greedy decode
  (`temperature=0`, `seed=1`), 3 short Q/A for LLM (`Paris`, `4`,
  `Mercury`) and any-of-N keyword match for VLM (golden retriever photo
  in `tests/assets/quality_dog.jpg`). Hard-asserts the substring; failure
  means the model wrote gibberish on that backend.

The conftest auto-tags items with markers driven by their location and
`device_map` parameter, so CI selects shards via `-m`:

| Marker          | Source                                                   |
|-----------------|----------------------------------------------------------|
| `api`           | items under `tests/api/`                                 |
| `llama_cpp`     | items under `tests/plugins/llama_cpp/`                   |
| `qairt`         | items under `tests/plugins/qairt/`                       |
| `device_cpu`    | parametrised with `device_map='cpu'`                     |
| `device_npu`    | parametrised with `device_map='npu'`                     |
| `device_hybrid` | parametrised with `device_map='hybrid'`                  |
| `snapdragon`    | any of `npu`, `hybrid`, or `qairt` (auto-applied)        |
| `llm` / `vlm`   | applied per-module via `pytestmark`                      |

`snapdragon`-marked items skip automatically unless
`GENIEX_DEVICE_TEST=1` is set **and** the host is a Snapdragon machine.
QAIRT models are pulled on demand from AI Hub (like the llama_cpp models
from HF), so the device shards need network access but no manual pre-pull.

## Running

```bash
# Anywhere — model-free API checks only
pytest tests -m api

# Anywhere — adds the llama_cpp CPU cells (downloads ~400 MB on first run)
pytest tests -m "api or (llama_cpp and device_cpu)"

# Snapdragon Windows ARM64 — full matrix
GENIEX_DEVICE_TEST=1 pytest tests
```

## Models

The matrix uses one model per modality, aligned across both plugins so a
keyword-quality divergence between llama_cpp and QAIRT traces to backend /
quantization rather than model identity (`tests/_models.py`):

| Modality | llama_cpp (HF GGUF) | QAIRT (AI Hub) |
|----------|---------------------|----------------|
| LLM      | `unsloth/Qwen3-4B-GGUF` Q4_0 | `qualcomm/Qwen3-4B` |
| VLM      | `unsloth/Qwen2.5-VL-7B-Instruct-GGUF` Q4_0 + mmproj-F16 | `qualcomm/Qwen2.5-VL-7B-Instruct` |

The LLM is Qwen3-4B **base**, not Instruct-2507: Instruct-2507 emits a long
`<think>` preamble before the answer that, on the 256-token budget the
suite uses, pushes the keyword off the end of the completion and turns
`test_quality_keywords` into a thinking-budget test rather than a
backend-quality test (seen on QDC NPU/Hybrid). The base model answers
directly.

Override the QAIRT model identifiers without editing the suite:

```bash
GENIEX_QAIRT_MODEL=qualcomm/<other-llm> \
GENIEX_QAIRT_VLM_MODEL=qualcomm/<other-vlm> \
GENIEX_DEVICE_TEST=1 pytest tests -m qairt
```

## CI

GitHub runners run only the model-free `api` shard
([.github/workflows/_test.yml](../.github/workflows/_test.yml)), since they
have no Snapdragon hardware. The model-running cells (`llama_cpp` / `qairt`
across every device) run on a real QDC X Elite (SC8380XP) Windows ARM64
device, added to the PR graph by
[pr-check.yml](../.github/workflows/pr-check.yml) via the reusable
[_qdc-pytest.yml](../.github/workflows/_qdc-pytest.yml) (after build-sdk, so
the SDK artifact is shared). The harness under [qdc/](qdc/) reuses
[sdk/benchmark/qdc/_qdc.py](../sdk/benchmark/qdc/_qdc.py) for submit / poll /
log-collect, and the QDC POWERSHELL framework runs
[qdc/windows/run_pytest.ps1](qdc/windows/run_pytest.ps1) on the device — it
fetches a portable Python ARM64, installs pytest, then runs this `tests/`
tree directly against the windows-arm64 SDK.

## Boundary with `bindings/python/tests/`

This directory is the home of SDK + plugin coverage. Tests under
`bindings/python/tests/` cover the binding layer itself (CLI wrapper,
progress callbacks, model_manager Python surface, local pull paths) and
do not launch real generation. Anything that reasons about device
selection, plugin behaviour, or model output belongs here.
