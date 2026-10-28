# Phonon through the QAIRT plugin

This source-build feature adds streaming TTS to the existing `qairt` plugin.
It uses `geniex-qairt-plugin`'s `TTSPipeline` and `PhononModel`, which execute
through the same `Model`/`Graph`, QNN loader, buffers, runtime configuration and
logging as QAIRT LLM/VLM models. There is no separate Phonon plugin or Gradium
QNN engine linked into the SDK.

Build the SDK normally with `GENIEX_PLUGIN_QAIRT=ON`. The QAIRT submodule must
contain the companion Phonon pipeline change. This is a source-only API preview;
`geniex pull` / `geniex infer` TTS catalogue and CLI support are not implemented.

Companion implementation: https://github.com/qualcomm/geniex-qairt-plugin/pull/81.
Windows validation details and measured observations are recorded in that
change's `models/phonon/VALIDATION.md`.

See the submodule's `models/phonon/README.md` for model export requirements,
Gradium text-frontend build instructions and the native pipeline API. Supply a
compiled QNN context for the exact SoC, its complete bundle, and a trusted native
`ptts_text` library. GGUF alone cannot be executed by this QAIRT pipeline.
No models or SDK payloads are added to this repository.

## Python

Set `GENIEX_LIB_PATH` to the newly built `geniex.dll` (or `libgeniex.so`) and
`GENIEX_PLUGIN_PATH` to the plugin directory containing `qairt/`. Make the built
QAIRT dependency DLLs available through your platform's loader search path.
Select a compatible QAIRT runtime **before** initialization:

```python
from geniex import GenieXTTS, set_qairt_runtime_path

set_qairt_runtime_path("C:/Qualcomm/AIStack/QAIRT/2.45.0.260326")
with GenieXTTS("C:/models/phonon-40m-qnn",
              text_library="C:/trusted/ptts_text.dll",
              soc_model="SC8380XP", verify_npu=True) as tts:
    result = tts.synthesize("Hello from Phonon on the Hexagon NPU.")
    print(result.first_audio_ms, result.rtf)
    print(result.prefill_npu_ms, result.step_npu_ms)
```

`plugin_id` defaults to `qairt`. Only the NPU device is accepted; CPU and GPU
requests fail. `verify_npu=True` enables shared QNN BASIC profiling and requires
positive accelerator event times for both graphs. No profiling environment
variable or alternate provider is needed. CPU work still includes text
preparation, state copies, scheduling and audio delivery.

For a WAV-producing example run `examples/python/phonon_tts.py --help`.
Callbacks receive float32 PCM bytes and sample rate. Return `False` or call
`cancel()` to stop; a subsequent request can reuse the same model. Native graph
errors invalidate the session. Destroy handles before SDK deinitialization.

## Validation

The model-free `GENIEX_TTS_TESTS=ON` suite verifies C API contracts and legacy
plugin compatibility. Run `ctest --test-dir <sdk-build> --output-on-failure`.
Hardware regression tests require local models and the native text frontend:

```powershell
python sdk/plugins/qairt/test_tts_device.py `
  --models C:/models/phonon-31m-qnn C:/models/phonon-40m-qnn `
  --runtime C:/Qualcomm/AIStack/QAIRT/2.45.0.260326/lib/aarch64-windows-msvc `
  --text-library C:/trusted/ptts_text.dll --soc SC8380XP --out tts-tests.json
```

Tested on Windows ARM64 / Snapdragon X Elite. Android and Linux hardware,
including VENTUNO Q, have not been validated by this revision. They require
their own compatible context binaries, drivers and text frontend. Reported
first-PCM/RTF measurements exclude model load and speaker playback.
