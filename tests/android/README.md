# Running the `tests/` pytest suite on Android via adb

Scripts that reproduce the [#682](https://github.com/qcom-ai-hub/geniex/issues/682)
harness: a portable Python (extracted from Termux, ~22 MB) runs the SDK's
`tests/` pytest suite directly in `adb shell` on a Snapdragon device — no Termux
APK install, no on-device compiler.

## On-device layout

```
/data/local/tmp/
├── termux-usr/    ~22 MB   portable Python 3.13 + pip + pytest + tqdm
├── geniex/                 pkg-geniex (Android NDK build) + flattened plugin libs
├── tests/                  tests/ (without models/ and android/)
└── .cache/geniex/models/   models (pulled manually, see below)
```

## Prerequisites

- An adb-connected Snapdragon Android device (`adb devices` shows it).
- `sdk/pkg-geniex` already built for Android (`arm64-android-snapdragon-debug`).
  Out of scope here — run `/build` or the docker preset first.
- Models already cached on-device under `~/.cache/geniex/models/` (step 3 below).

## 1. Build the portable Python (one-time, needs network)

```bash
tests/android/build_python_tarball.sh        # → tests/android/dist/python-android-portable.tar.gz
```

Downloads the Termux aarch64 packages, extracts `usr/`, and layers in
`pytest`/`tqdm` via pip on the device. The tarball is reusable across runs and
devices of the same arch; it is gitignored, not committed.

## 2. Deploy

```bash
tests/android/run_on_device.sh deploy
```

Idempotent. Pushes the portable Python (skipped if already present — add
`--force-python` to redeploy), `pkg-geniex`, `tests/`, and the geniex Python
bindings, then **flattens** `lib/qairt/htp-files/*.so` and `lib/llama_cpp/*.so`
up into `lib/` (symlinks) — the qairt plugin `dlopen`s `GENIEX_LIB_PATH/libQnn*.so`
flat, and the llama_cpp Hexagon FastRPC layer resolves the ggml-htp skels off a
flat `ADSP_LIBRARY_PATH=lib`. Both are SDK-side layout bugs tracked in #682; the
flatten is the workaround.

## 3. Pull models (manual — out of scope for these scripts)

Models are assumed present on-device. Pull them with the geniex CLI inside an
adb shell, sequentially (concurrent pulls trigger HEAD-retry failures). Add an
HTTP proxy via env vars only if your network needs one:

```bash
adb shell '
export PREFIX=/data/local/tmp/termux-usr
export LD_LIBRARY_PATH=$PREFIX/lib:/data/local/tmp/geniex/lib:/data/local/tmp/geniex/lib/qairt:/data/local/tmp/geniex/lib/qairt/htp-files:/system/lib64
export GENIEX_LIB_PATH=/data/local/tmp/geniex/lib
export HOME=/data/local/tmp
# export http_proxy=http://<host>:<port> https_proxy=http://<host>:<port>   # only if needed
$PREFIX/bin/python3 -m geniex.cli pull qualcomm/Qwen3-4B-Instruct-2507
'
```

## 4. Run pytest

No default selection — you pass the `-m` / paths explicitly:

```bash
# API metadata subset (13/13 pass; test_device_list excluded — SDK SIGABRT)
tests/android/run_on_device.sh test -- -m api --ignore=api/test_device_list.py

# Real QAIRT/NPU inference
tests/android/run_on_device.sh test -- plugins/qairt/test_qairt_llm.py
```

pytest's exit code is propagated. `tests/android/run_on_device.sh shell` opens an
interactive shell with the same env for manual debugging.

### Status of test groups on Android

| Group | Status |
|---|---|
| `tests/api/` (minus `test_device_list`) | 13/13 pass |
| `tests/api/test_device_list.py` | excluded — `geniex_get_device_list` SIGABRTs (SDK bug) |
| `tests/plugins/qairt/test_qairt_llm.py` | NPU inference runs; currently fails in a binding strict-UTF-8-decode bug (#888), not the harness |
| `tests/plugins/qairt/test_qairt_vlm.py` | not yet verified |
| `tests/plugins/llama_cpp/` | runs once libs are flattened (done by `deploy`) |
