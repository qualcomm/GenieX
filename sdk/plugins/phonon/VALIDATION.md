# Local validation: 22 October 2026

GenieX base: `9d456b3131f799e651d87e769ef56e046ff96ac3`.
Local branch: `feat/phonon-tts`. Initial measurements below predate the shared-runtime follow-up.

Hardware: Snapdragon X Elite Windows ARM64 laptop, SC8380XP, Windows build 26200.
QAIRT host and v73 DSP runtime: `2.45.0.260326`.
Native build: MSVC ARM64 19.50, Release; Rust ARM64 model-manager; Gradium source
`9ee0eb5706d18ef1e401dc8b9d732c886089a5ab` supplied externally and enforced by CMake.

## Functional results

| Check | Result |
|---|---|
| Rebuilt current GenieX SDK and optional Phonon plugin | Passed |
| Native C/C++ API contracts, including C header compilation | Passed |
| Existing Python unit suite plus new TTS tests | **62 passed** |
| Model-free SDK API checks applicable to the QAIRT/Phonon build | **10 passed**, six llama.cpp-specific cases deselected because that plugin was disabled |
| Go cgo build, streaming, cancellation, panic containment and reuse | Passed |
| Go real NPU synthesis, both English models | Passed, two requests per model |
| Android Kotlin class and ARM64 Android JNI object compilation | Passed |
| Android ARM64 plugin and all four shared engine translation units | Passed, NDK 28.1.13356709, API 31, QAIRT 2.45 headers; no linking/runtime claim |
| Kotlin result constructor vs JNI signature | `(IIJDDDDII)V` verified |
| Real-device regression scenarios | **29 passed** across both models |
| Gradium model-free logging/bundle/FP16/Windows loader tests | **4 passed** |
| Clang-format 18.1.8 and Ruff 0.6.9 checks, gofmt, git whitespace checks | Passed |
| GitHub Actions execution | Not run; model-free Windows workflow added |

Device scenarios cover NPU-only enumeration, CPU/GPU rejection, wrong SoC,
conflicting runtime assertions, post-init runtime locking, inherited selection,
missing native text frontend, verification without profiling,
callback and explicit cancellation, callback exceptions, model reuse,
noncumulative graph counters, multichunk speech, and invalid voices.

Both graphs report positive
`QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_MICROSEC` on every completed
measured request. These are accelerator events, not wall-clock timing alone.
The plugin fixes the engine backend to `htp`; it does not retry another backend.
The tests do not claim that text normalization or playback runs on the NPU.

## Resident Qwen and Phonon integration

Built the same SDK with both QAIRT and Phonon enabled, llama.cpp disabled, and
the pinned logging-capable Gradium engine. The current QAIRT plugin and its
pinned core/processor dependencies were built from source; this test does not
mix an older prebuilt QAIRT plugin with the new SDK.

`test_mixed.py` passed **17 checks** in a single Windows ARM64 process:

- Qwen3-4B-Instruct-2507 remained resident while Phonon 31M, then 40M, were
  created, used repeatedly and destroyed. Five Qwen requests returned the
  expected `NPU READY`, `NPU STILL READY` and `QUEUE READY` messages.
- Eight completed speech requests produced valid, nonsilent mono PCM WAVs.
  Each reported positive prefill and step HTP accelerator events. Both speech
  models also handled callback cancellation and subsequent reuse.
- Two deliberately overlapping application requests queued behind active speech
  through an application lock, then Qwen completed. This validates that scheduling
  example, not simultaneous graph execution or an SDK-wide automatic scheduler.
- The process loaded `QnnHtp.dll` and `QnnSystem.dll` from the selected
  `QAIRT/2.45.0.260326/lib/aarch64-windows-msvc` directory. Module enumeration
  found no `QnnCpu.dll` or `QnnGpu.dll`. No alternative provider retry is implemented.
- The real vendor callback produced `Phonon QNN:` messages through the GenieX
  logging sink, confirming the embedding hook works beyond its unit test.

The fixed-text speech calls in this diagnostic run had first-PCM ranges of
**18.80–25.55 ms** (31M) and **17.44–33.20 ms** (40M). Their RTF ranges were
**0.1043–0.1300** and **0.1040–0.1142**. Qwen first-token times were
**105.5–143.6 ms**, after model loading, for these short controlled prompts.
Verbose QNN diagnostics were enabled and Qwen stayed resident; this is a small
integration observation, not a standalone performance benchmark or a comparison
against the historical numbers below. Queue-gated callback timings are excluded
from those speech ranges. No audio naturalness or full waveform parity claim is made.

## Measured latency

Text: “Hello from Phonon, running locally through GenieX on the Snapdragon NPU.”
These initial/repeat measurements used engine revision `b2896b9`, before the
shared-runtime and embedding-logging follow-up; they are retained as baseline.
Default voice, 24 kHz mono. Six consecutive requests per model; medians below
use the last five requests after the model has been loaded. Output durations
vary slightly with generation. Audio was checked for valid format, sample
counts and nonsilence, not scored for naturalness.

| English model | Initial warm first PCM median | Initial warm RTF median | Repeat warm first PCM median | Repeat warm RTF median |
|---|---:|---:|---:|---:|
| Phonon 31M | 10.66 ms | 0.0595 | 15.12 ms | 0.0631 |
| Phonon 40M | 12.44 ms | 0.0645 | 23.78 ms | 0.0817 |

The independent Python callback timer in the repeat measured **15.30 ms** and
**23.93 ms** respectively. SDK/model initialization was about **1.02 s** and
**1.07 s** in that repeat and is excluded from first-PCM latency.
The repeat's 40M warm range was **16.04–81.46 ms**. Laptop load and scheduling
affect latency; these measurements are not a fixed 14 ms guarantee. A separate
run with overlapping test activity is retained as a load observation, not used
as the single-model benchmark. Speaker/output-device latency is not measured.

## Evidence and reproducibility

The working session retains JSON, WAV and logs outside the source checkout in
`outputs/geniex-phonon-upstream/`: initial `31/` and `40/` results, `latest/`
repeat results, `device-tests.json`, Python test/build/Go logs, and a local
`test-local.ps1` launcher. Paths are local artifacts, not files expected in a
fresh upstream clone. Portable setup and test commands are in [README.md](README.md).

The shared-runtime follow-up retains `shared-runtime-device-tests.json`,
`shared-runtime-{python-unit,api}-tests.log`, `shared-runtime-go-*.log`,
`android-runtime-compile/`, `mixed-session.log` and `mixed/results.json` plus
eight WAVs. The local `test-mixed-local.ps1` launcher recreates that integration
run; add `-Play` to listen afterward. Its paths are specific to the test machine.

## Before promoting the draft

The proposed C speech API and optional factory export need maintainer review;
this is a working contribution candidate, not an accepted upstream API.
The Windows Gradium engine dependency currently comes from its Windows support
PR. Agree on an upstream dependency revision/package strategy before release.

No CLI speech command, model catalogue entry,
Linux/VENTUNO Q execution, Android device test, or non-English model validation
is claimed. No model weights, QAIRT libraries, credentials or generated audio
are included in this contribution.
