# SDK pytest suite

End-to-end tests for the geniex SDK, driven through the public Python
binding so the suite doubles as a public-surface contract check.

```
tests/
├── api/                   # SDK metadata APIs (no model required)
├── plugins/
│   ├── llama_cpp/         # LLM + VLM × {cpu, npu, hybrid}
│   └── qairt/             # LLM + VLM × {npu}
├── cli/                   # Bazel-driven Go CLI black-box tests (separate)
├── conftest.py            # Top-level fixtures (init, model paths, image)
├── pytest.ini             # Marker registry + suite discovery
└── _models.py             # Model identifiers used by the matrix
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

Override the QAIRT model identifiers without editing the suite:

```bash
GENIEX_QAIRT_MODEL=aihub/<other-llm> \
GENIEX_QAIRT_VLM_MODEL=aihub/<other-vlm> \
GENIEX_DEVICE_TEST=1 pytest tests -m qairt
```

## CI

GitHub runners run only the model-free `api` shard
([.github/workflows/_test.yml](../.github/workflows/_test.yml)), since they
have no Snapdragon hardware. The model-running cells (`llama_cpp` / `qairt`
across every device) run on a real QDC Android phone, added to the PR graph
by [pr-check.yml](../.github/workflows/pr-check.yml) via the reusable
[_qdc-pytest.yml](../.github/workflows/_qdc-pytest.yml) (after build-sdk, so
the SDK artifact is shared). Its harness under [qdc/](qdc/) reuses the
benchmark's QDC submit/poll/collect plumbing and the
[android/](android/) adb scripts to drive the device.

## Boundary with `bindings/python/tests/`

This directory is the home of SDK + plugin coverage. Tests under
`bindings/python/tests/` cover the binding layer itself (CLI wrapper,
progress callbacks, model_manager Python surface, local pull paths) and
do not launch real generation. Anything that reasons about device
selection, plugin behaviour, or model output belongs here.
