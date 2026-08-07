# SDK pytest suite

End-to-end tests for the geniex SDK, driven through the public Python
binding so the suite doubles as a public-surface contract check.

```
tests/
├── assets/                # Real test images (quality_dog.jpg + NOTICE.md)
├── qdc/                   # QDC device dispatcher + per-platform entry scripts
├── test_api.py            # SDK metadata + resolve APIs (no model required)
├── test_llama_cpp.py      # llama_cpp plugin — LLM + VLM + precision + MTP
├── test_qairt.py          # qairt plugin     — LLM + VLM + precision
├── conftest.py            # Top-level fixtures (init, model paths, image)
├── pytest.ini             # Marker registry + suite discovery
├── _models.py             # Model identifiers used by the matrix
└── _quality_data.py       # Keyword-quality prompts shared by both plugins
```

## What each plugin covers

Every plugin file ships the same four cases, plus an explicit
model-manager-pull check that runs first:

| Test                          | What it proves                                                     |
|-------------------------------|--------------------------------------------------------------------|
| `test_model_manager_pull`     | AI Hub / HuggingFace pull for every model this file needs works.   |
| `test_llm_multi_turn`         | Two-turn "Alice" conversation; turn-2 must recall the name.        |
| `test_vlm_multi_turn`         | Two-turn conversation with an image on turn 1.                     |
| `test_llm_quality_keywords`   | Greedy decode, 3 short Q/A; keyword substring must land.           |
| `test_vlm_quality_keywords`   | Golden-retriever caption must match one of the canonical keywords. |
| `test_mtp_multi_turn`         | (llama_cpp only) same "Alice" convo with `spec_type='draft-mtp'`.  |

Backends per plugin: `llama_cpp` on `cpu` + `gpu` + `npu`; `qairt` on `npu` only.
Model-manager pull failures are treated as FAIL, not SKIP, so a broken
download surfaces as a red CI leg instead of a silent green skip.

## Marker registry

The conftest auto-tags items by location and `device_map` value:

| Marker          | Source                                                   |
|-----------------|----------------------------------------------------------|
| `api`           | items in `tests/test_api.py`                             |
| `llama_cpp`     | items in `tests/test_llama_cpp.py`                       |
| `qairt`         | items in `tests/test_qairt.py`                           |
| `device_cpu`    | parametrised with `device_map='cpu'`                     |
| `device_gpu`    | parametrised with `device_map='gpu'`                     |
| `device_npu`    | parametrised with `device_map='npu'`                     |
| `snapdragon`    | any `device_map` in {`gpu`, `npu`} cell (auto-applied)   |
| `llm` / `vlm`   | applied per-test via `@pytest.mark.llm` / `.vlm`         |

`snapdragon`-marked and `qairt` items skip automatically unless
`GENIEX_DEVICE_TEST=1` is set **and** the host is a Snapdragon machine.
QAIRT models are pulled on demand from AI Hub (like the llama_cpp models
from HF), so the device shards need network access but no manual pre-pull.

## Running

```bash
# Anywhere — model-free API checks only
pytest tests -m api

# Anywhere — adds the llama_cpp CPU cells (downloads ~400 MB on first run)
pytest tests -m "api or (llama_cpp and device_cpu)"

# Snapdragon Windows ARM64 or Qualcomm Linux — full matrix
GENIEX_DEVICE_TEST=1 pytest tests
```

## Models

The matrix uses one model per modality, aligned across both plugins so a
keyword-quality divergence between llama_cpp and QAIRT traces to backend /
quantization rather than model identity. Manifest: `tests/models.json` (edit
this file to swap or add models); loader: `tests/_models.py`.

| Modality | llama_cpp (HF GGUF) | QAIRT (AI Hub) |
|----------|---------------------|----------------|
| LLM      | `unsloth/Qwen3-4B-GGUF` Q4_0 | `qualcomm/Qwen3-4B` |
| VLM      | `unsloth/gemma-4-E2B-it-GGUF` Q4_0 + mmproj-F16 | `qualcomm/Qwen2.5-VL-7B-Instruct` |
| MTP      | `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` + `RachidAR/gemma-4-...-assistant-q4_0-gguf` | — (llama_cpp only) |

The LLM is Qwen3-4B **base**, not Instruct-2507: Instruct-2507 emits a long
`<think>` preamble before the answer that, on the 256-token budget the
suite uses, pushes the keyword off the end of the completion and turns
`test_llm_quality_keywords` into a thinking-budget test rather than a
backend-quality test.

Override the QAIRT model identifiers without editing the suite:

```bash
GENIEX_QAIRT_MODEL=qualcomm/<other-llm> \
GENIEX_QAIRT_VLM_MODEL=qualcomm/<other-vlm> \
GENIEX_DEVICE_TEST=1 pytest tests -m qairt
```

## CI

The project splits into two test workflows:

- **Unit Test** ([`_unit-test.yml`](../.github/workflows/_unit-test.yml)) — runs on
  every PR via [`pr-check.yml`](../.github/workflows/pr-check.yml). Covers
  `test-go`, `test-python`, and `test-sdk-ci` (`-m api`) on
  linux-arm64 + windows-arm64 GitHub runners; no QDC hardware.
- **QDC Test** ([`_qdc-test.yml`](../.github/workflows/_qdc-test.yml))
  — runs on `workflow_dispatch` and on `v*` tag push (via
  [`test.yml`](../.github/workflows/test.yml)). Not on PR. One leg per
  platform; each leg runs the full plugin set (`-m "llama_cpp or qairt"`):

  | Platform | Device   | Framework   |
  |----------|----------|-------------|
  | Linux    | QCS9075M | BASH        |
  | Windows  | SC8480XP | POWERSHELL  |

Each platform's entry script lives under [qdc/](qdc/): Linux uses the
image's preinstalled python 3.12; Windows bootstraps python.org's arm64
embed zip. Both reuse
[sdk/benchmark/qdc/_qdc.py](../sdk/benchmark/qdc/_qdc.py) for submit /
poll / log-collect.

Android (SM8850) is not wired yet — QDC's SM8850 image has neither
python nor termux preinstalled, so it needs a CLI-driven harness that is
out of scope for this iteration.

## Boundary with `bindings/python/tests/`

This directory is the home of SDK + plugin coverage. Tests under
`bindings/python/tests/` cover the binding layer itself (CLI wrapper,
progress callbacks, model_manager Python surface, local pull paths) and
do not launch real generation. Anything that reasons about device
selection, plugin behaviour, or model output belongs here.
