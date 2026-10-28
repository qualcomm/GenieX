# Platform portability and model formats

The portable component is the speech API and generation engine, not a single
binary or a guarantee that a GGUF file runs on every accelerator.

## GGUF and QNN are different execution routes

A Phonon GGUF checkpoint stores weights and metadata. An inference engine must
implement Phonon's architecture and kernels for the selected processor. A
generic LLM GGUF loader does not gain those capabilities by reading the container.
The Gradium Rust engine has its own device build features; this contribution
does not wrap that GGUF route.

The `phonon_qnn` plugin consumes exported QNN bundles: metadata, tokenizer/BOS/
voice assets and compiled prefill/step contexts. The context must be compatible
with the SoC and QAIRT runtime. The native libraries and Rust text frontend must
also match the OS and ABI. Never reuse a Windows DLL on Android, or assume an
X Elite context works on a phone. Adding a SoC name to metadata does not port it.

Gradium's [QNN Android guide](https://github.com/gradium-ai/xn-ptts/blob/main/qnn/android/README.md)
documents the existing Android engine, packaging and tested mobile targets. Its
documented bundles/runtime use QAIRT 2.50.0; our Windows measurements use
2.45.0.260326. Do not mix those artifacts without compatibility validation.

## Keep one application-facing interface

```text
Application: text + voice -> PCM stream + cancellation + metrics
                         |
                  geniex_tts.h / ITts
                         |
                phonon_qnn, strict HTP
                         |
          Shared metadata-driven Phonon C++ engine
                         |
   Native OS loader + text frontend + QNN runtime + SoC context
```

Windows uses DLLs and its audio API; Android uses ARM64 shared libraries and
AudioTrack; Linux uses ARM64 shared libraries and a chosen audio sink. Playback
stays outside the inference plugin. The C callbacks carry PCM without assuming
any playback API. Python, Go and Kotlin can share that lifecycle.

Model size, graph dimensions and voices come from the bundle, rather than
Windows-specific 31M/40M conditionals. Further languages/checkpoints can use the
same interface if compatible, but each requires its own model and hardware tests.

## Support matrix

| Target | Shared API/native source | Runtime and model assets | Validation here |
|---|---|---|---|
| Windows ARM64 / X Elite | Built and exercised | SC8380XP bundles, QAIRT 2.45.0.260326 | 31M/40M English NPU inference verified |
| Android ARM64 / Snapdragon | Kotlin/JNI and shared engine compile checks | Android QAIRT, packaged text library, SoC bundle required | Compilation only; no device inference or complete APK claim |
| Linux ARM64 / Snapdragon or Dragonwing | Common API/source intended for reuse | Linux QAIRT/FastRPC, text library and target context required | Not built or run here |
| Other processors/OSes | API can accommodate another backend | Separate implemented and tested backend required | No support claimed by this QNN HTP plugin |

## Android compile check

This model-free check compiles the plugin, its four Phonon engine translation
units and JNI bridge to Android ARM64 objects. It does not link the complete
GenieX SDK, build an AAR/APK, or prove QNN runtime compatibility.

```sh
python sdk/plugins/phonon/check-android-compile.py \
  --ndk /path/to/android-ndk \
  --qairt /path/to/qairt-sdk \
  --phonon-source /path/to/reviewed-xn-ptts \
  --json-include /path/to/nlohmann-json/single_include \
  --out /path/to/android-compile-check
```

The JSON headers are the verified 3.11.3 dependency fetched by the Phonon CMake
build. Explicit reviewed dependencies are used; no code comes from model folders.

## What remains for Android and Linux releases

1. Build/link the full GenieX SDK, Phonon plugin and Rust text frontend for the
   OS/ABI with compatible toolchains.
2. Package matching QAIRT host and HTP stub/skel libraries under Qualcomm's
   distribution terms. Configure trusted native and DSP loader paths first.
3. Select a context for the actual SoC and runtime; reject unsupported targets.
   Android can obtain its SoC from `Build.SOC_MODEL`.
4. Test library loading, callbacks, cancellation, repeated/multichunk synthesis
   and actual playback. Test Android debug and minified release APKs; preserve
   classes referenced by JNI.
5. Require positive prefill and step HTP accelerator events and run the negative
   tests. Keep CPU/GPU fallback disabled.

A future `phonon_gguf` plugin could implement the same `ITts` interface through
Gradium's Rust engine for explicitly selected CPU/GPU targets. That would be a
separate backend, not automatic fallback from this NPU plugin. It is not
implemented in this PR.
