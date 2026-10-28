# Architecture study: Phonon speech in GenieX

22 October 2026. This is a source comparison and recommendation, not maintainer
approval or a new hardware benchmark. The initial implementation audited plugin
and FFI interfaces and proved local NPU execution; this broader pipeline study
was completed after the draft was opened. The original audit was not sufficient
to call the implementation the recommended upstream architecture.

## Source revisions and method

Inspected current GenieX main `9d456b3131f799e651d87e769ef56e046ff96ac3`, its pinned
QAIRT dependency `2c3d4fcd2e3c8c917da6308c1459ccf83086b29f`, and its pinned llama.cpp
dependency `e467e5b5e5ba43071d6151b3d881deb9300d000e`. Read plugin factories,
LLM/VLM wrappers, model dispatch/factories, generic graph/runtime interfaces,
audio-generation helpers, runtime/logging changes, package configuration and
the model-manager modality schema. Compared those with draft #1581 and the
external Gradium engine it uses. No third-party source was modified.

## Existing implemented precedents

| Precedent | Observed implementation | Implication for Phonon |
|---|---|---|
| GenieX `llama_cpp` and `qairt` plugins | Thin SDK wrappers around separate engines, with pinned submodules and installed native dependencies | A dedicated Gradium engine plugin is a legitimate option; it should follow the dependency/runtime/package conventions |
| QAIRT Qwen2.5-VL pipeline | Separate vision encoder and language model, configured through one `QnnRuntimeConfig`, composed by a family factory | Model-specific multi-graph loops can live below a shared SDK interface; a Phonon pipeline need not masquerade as an LLM |
| QAIRT Gemma4 pipeline | Family-specific preprocessing, tensor geometry validation and encoder logic over reusable `Model`/`Graph` execution | A `TtsPipeline` over the existing QAIRT core is an alternative worth evaluating; it is not already implemented |
| Pinned llama.cpp Qwen3-TTS | Dedicated backbone/audio-generation stages and waveform production through `libmtmd` | A common GenieX speech API should be able to expose a second real backend, rather than encode only one model's assumptions |
| Pinned llama.cpp Pocket TTS | Flow-matching latent generation followed by a decoder producing PCM | Flow matching does not intrinsically rule out GGUF/ggml; the architecture and its operations must actually be implemented |

Evidence:

- [Plugin modality factories](https://github.com/qualcomm/GenieX/blob/9d456b3131f799e651d87e769ef56e046ff96ac3/sdk/include/plugin/Plugin.h)
  and [pinned dependencies](https://github.com/qualcomm/GenieX/blob/9d456b3131f799e651d87e769ef56e046ff96ac3/.gitmodules).
- [QAIRT dispatcher](https://github.com/qualcomm/geniex-qairt-plugin/blob/2c3d4fcd2e3c8c917da6308c1459ccf83086b29f/models/dispatch.h),
  [Qwen2.5-VL factory](https://github.com/qualcomm/geniex-qairt-plugin/blob/2c3d4fcd2e3c8c917da6308c1459ccf83086b29f/models/qwen2_5_vl/qwen2_5_vl.cpp),
  [Gemma4 implementation](https://github.com/qualcomm/geniex-qairt-plugin/blob/2c3d4fcd2e3c8c917da6308c1459ccf83086b29f/models/gemma4/gemma4.cpp).
- [Generic Model](https://github.com/qualcomm/geniex-qairt-plugin/blob/2c3d4fcd2e3c8c917da6308c1459ccf83086b29f/core/include/model.h)
  and [Graph](https://github.com/qualcomm/geniex-qairt-plugin/blob/2c3d4fcd2e3c8c917da6308c1459ccf83086b29f/core/include/graph.h)
  support named tensor access, graph execution and raw buffer operations.
- [Pinned llama.cpp speech tool](https://github.com/ggml-org/llama.cpp/blob/e467e5b5e5ba43071d6151b3d881deb9300d000e/tools/tts/README.md),
  [generic speech driver](https://github.com/ggml-org/llama.cpp/blob/e467e5b5e5ba43071d6151b3d881deb9300d000e/tools/tts/tts.cpp),
  [Pocket TTS flow/PCM stages](https://github.com/ggml-org/llama.cpp/blob/e467e5b5e5ba43071d6151b3d881deb9300d000e/tools/mtmd/models/pockettts-gen.cpp).

These dependency implementations are not equivalent to a shipped GenieX speech
API. At this revision GenieX's plugin factories expose `ILlm` and `IVlm`; its
normal generation wrappers stream text tokens. Its llama.cpp audio-input path
is also different from the `libmtmd` audio-generation path in `llama-tts`.
No Phonon architecture entry was found in the pinned llama.cpp registry.
Qwen3-TTS/Pocket TTS architecture entries do not prove their full NPU execution
through GenieX; that was not tested in this study.

## Material gaps found at study revision 4c5e30b

1. **Runtime selection differs from GenieX's contract.** GenieX deliberately
   latches QAIRT selection before init and documents one runtime per process.
   The Phonon draft accepts a per-model `runtime_path` and passes it directly to
   Gradium's loader, bypassing that governance. Isolated 2.45 Phonon tests do not
   validate a process that has already loaded another QNN runtime for Qwen.
   GenieX is currently compiled against QAIRT 2.50.40 headers; its documentation
   distinguishes provider compatibility from actual model-inference validation.
   Neither a version number nor an API query alone settles bundle compatibility.
   See [runtime API rationale, PR #1441](https://github.com/qualcomm/GenieX/pull/1441)
   and [current runtime contract](https://github.com/qualcomm/GenieX/blob/9d456b3131f799e651d87e769ef56e046ff96ac3/notes/run.md#using-a-custom-qnn-library).
2. **Logging is not fully integrated.** Existing plugins forward engine logs to
   GenieX; the external Gradium QNN logger still writes to stderr. A dependency
   logging callback is preferable to hiding or redirecting stderr. See
   [QAIRT logging integration, PR #1470](https://github.com/qualcomm/GenieX/pull/1470).
3. **Dependency/package strategy is developmental.** `GENIEX_PHONON_SOURCE`
   intentionally accepts a reviewed checkout, but does not itself enforce the
   documented revision. Existing engines are pinned and packaged with their
   dependencies. The Gradium Windows support PR remains a release dependency.
4. **Plugin lifetime and mixed sessions need evidence.** Existing llama.cpp
   code tracks HTP use and yields idle sessions for foreign-plugin handoff.
   Phonon standalone tests do not cover LLM/TTS switching or resource contention.
   Do not infer that a new unload/release hook alone makes active models safe.
5. **The new modality ABI is a proposal.** The optional factory export preserves
   the existing Plugin vtable, but is a new extension mechanism. It does not
   cover legacy in-process registration. Maintainers should choose its public
   contract; it is not an existing endorsed TTS extension convention.
6. **Platform packaging is incomplete.** Existing plugins set nested UNIX
   RUNPATHs such as `$ORIGIN:$ORIGIN/..`. The Phonon target does not yet reproduce
   that per-target setup. Object compilation is not a linked Android package
   test. See [Android loader/package fixes, PR #1571](https://github.com/qualcomm/GenieX/pull/1571).

The model manager currently distinguishes `llm` and `vlm`, not `tts`. Catalogue,
auto-loading, CLI and server speech routes remain explicit future work. Likewise,
the existing VLM encoder path can retry on CPU; that behavior must not be copied
into the strict NPU Phonon path merely for consistency.

Historical Kokoro/Bark/OuteTTS issues are not proof that current GenieX already
has a usable TTS API. [Kokoro issue #387](https://github.com/qualcomm/GenieX/issues/387)
was archived as belonging to the pre-rewrite Nexa SDK.

## Recommendation

**Keep a backend-neutral TTS API, and keep #1581 as a design/validation draft.**
The dedicated Phonon plugin is reasonable as a way to reuse Gradium's tested
generation loop, similar to GenieX wrapping llama.cpp. It is not yet a complete
upstream integration. Resolve shared runtime selection, logging, pinning,
packaging and mixed-session tests before calling it ready to merge.

Evaluate these two implementation choices explicitly with maintainers:

- **Thin Gradium plugin:** preserve the optimized upstream Phonon loop, add a
  proper dependency logging/runtime configuration interface, and make it honor
  the same process-wide QAIRT policy as GenieX. This is the lower-reimplementation
  option, provided those integration contracts can be enforced.
- **TTS in `geniex-qairt-plugin`:** implement a Phonon model/pipeline over its
  generic `Model`/`Graph` core. This follows Qwen/Gemma multi-graph precedents and
  reuses runtime infrastructure, but requires porting and parity testing of
  Phonon's cache, tensor layout, flow/noise schedule, chunking and waveform loop.
  It is not a small dispatcher change, and should not duplicate Gradium code
  without a maintenance agreement.

A later llama.cpp speech adapter could implement the same TTS interface for
Qwen3-TTS/Pocket TTS. A Phonon GGUF route separately needs Phonon's architecture,
conversion and device-operation coverage. No automatic CPU/GPU fallback should
be added to the verified QNN NPU route.

## Before promoting the draft

- Agree on the modality ABI and engine ownership/dependency route.
- Use one chosen compatible QNN runtime for the complete process; validate or
  recompile model contexts as required. Reject conflicting runtime requests.
- Run Qwen -> Phonon -> Qwen in one process, with multiple utterances and
  cancellation, then test an overlap request with explicit scheduling semantics.
- Verify both Phonon graphs' accelerator events and compare output quality and
  latency after integration changes, not only after standalone load.
- Link/package a real Android application and run it on supported hardware;
  perform Linux/VENTUNO Q validation separately.

At study revision these were recommended next steps, not completed tests or
maintainer requirements. The follow-up below records subsequent implementation;
the historical findings above describe the earlier draft.

## Implementation follow-up

The current Phonon plugin requires the existing process-wide runtime setter
before SDK initialization and inherits that selection. Its old per-model path
field is an optional consistency assertion, preserving the struct layout while
rejecting conflicting directories. Common SDK-root/host/DSP path resolution now
lives in `internal/qnn_paths.h`, shared with QAIRT. Existing core latch semantics
are unchanged. Applications must explicitly choose a compatible runtime; model
metadata does not select or change it.

The Gradium dependency exposes an embedding log callback. The plugin forwards
its diagnostics through the GenieX sink without redirecting process stderr.
CMake enforces the documented exact source commit and a clean runner source
tree. The Phonon target now sets the nested UNIX RUNPATH used by other plugins.
These changes do not establish complete Android/Linux packaging.

`test_mixed.py` provides the resident Qwen/Phonon switching and queued-request
check. It uses explicit application scheduling, not an implicit promise of safe
simultaneous graph execution. [VALIDATION.md](VALIDATION.md) records the actual
hardware results and the checks still needed. ABI acceptance, engine ownership,
upstream dependency acceptance, and Android/Linux device validation remain open.
