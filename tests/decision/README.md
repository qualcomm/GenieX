# Native decision adapter tests

These C++ tests compile the decision preparation/readout code directly, without
Rust, the model manager, or Snapdragon hardware. The default suite uses a tiny
synthetic GGUF vocabulary: no model download is needed.

```sh
cmake -S tests/decision -B build-decision-tests -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=OFF
cmake --build build-decision-tests --target decision-tests -j
ctest --test-dir build-decision-tests --output-on-failure
```

Coverage includes protocol detection and unknown-type rejection, missing-template
rejection for Liquid, existing OpenJEV/LEV/Nimble/generic preparation, LEV's nine
predicate ratings and reversed choice pass, Liquid predicate ordering and token
variants, choice and score schema rendering, Omni mask positions/type columns,
escaping and state truncation, bucket temperature and score bias, and invalid
readout/calibration values (including empty/truncated metadata). Liquid preserves
literal SDK media-marker text and d1-2B escapes only special/control tokens, not
ordinary added tokens. Image and audio rejection are checked explicitly.

## Real CPU inference and native reference

Enable the optional llama.cpp reference comparison, then pass an actual GGUF:

```sh
cmake -S tests/decision -B build-decision-tests \
  -DGENIEX_DECISION_REFERENCE_TESTS=ON -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=OFF
cmake --build build-decision-tests --target decision-tests llama-server -j
./build-decision-tests/decision-tests /path/to/d1-2B-Q4_K_M.gguf
```

This runs predicate, choice, and score inference on CPU (`n_gpu_layers=0`). It
compares the adapter's token stream with llama.cpp's native `fill_task`, and its
probabilities with the native readout/calibration (absolute tolerance `1e-6`).
It also checks repeated requests, context overflow rejection, and a longer
predicate (339/347/341 tokens for the tested models). The causal d1-3B long case
crosses logical/physical batch boundaries (128/64); non-causal models use one
complete batch and reject a too-small explicit physical batch.

Reference preparation cases compare exact tokens, verbalizer groups, mask
positions, and question-type columns for Unicode, reserved spellings, literal
SDK media markers, long states, 2/26/27/255 choices, and 2/3/5/6/10 score levels.
These long-state preparation comparisons are not long-context inference claims.
Reference comparison is test-only; the plugin does **not** link llama.cpp's HTTP
server.

The protocol contracts were checked against:

- [Liquid d1-3B `prompt.py`](https://huggingface.co/LiquidAI/d1-3B/blob/main/prompt.py):
  ChatML prompt, yes/no casing groups, single-token choice aliases and leading-space
  variants, digit score levels, max-pooling before temperature/softmax.
- [Liquid d1-omni-600M `prompt.py`](https://huggingface.co/LiquidAI/d1-omni-600M/blob/main/prompt.py)
  and `modeling_d1.py`: independently tokenized/budgeted prompt pieces, false/true
  mask order, non-causal full-question inference, three question-type columns,
  bucket calibration. Media have separate token budgets and are not calibrated
  like text. Those media paths are intentionally not implemented in this PR.
- [Liquid d1-2B `prompt.py`](https://huggingface.co/LiquidAI/d1-2B/blob/main/prompt.py),
  `runner.py`, and `calibration.json`: LFM3 non-causal isolated questions,
  A..Z/AA.. codes, special-token escaping, and the published temperature. Its
  extraction/box/estimate/multiple-choice capabilities are outside GenieX's schema.
- Pinned llama.cpp
  [server-decision.cpp](https://github.com/ggml-org/llama.cpp/blob/d05706b4f32a545814856e98ca518292585443b9/tools/server/server-decision.cpp),
  `server-context.cpp`, and `src/models/lfm2.cpp` / `lfm3.cpp`.

The dependency advances from `19cce6919` to `d05706b4f` (seven commits), the first
upstream commit implementing d1-2B's memoryless LFM3 runtime and decision protocol.
LFM2 architecture, vision/audio projectors, and d1/Omni decision functionality
were already present at the old pin. All five existing GenieX llama.cpp patches
pass `git apply --check` at the new pin. Projectors were inspected, not executed.

## CLI JSON and independent HTTP comparison

Import a GGUF into GenieX, then compare CLI JSON with a separate native server:

```sh
python tests/decision/compare_systemone.py \
  --geniex /path/to/geniex --llama-server ./build-decision-tests/bin/llama-server \
  --gguf /path/to/d1-2B-Q4_K_M.gguf --model-name local/d1-2b:Q4_K_M \
  --data-dir ./data --work-dir ./decision-comparison
```

Use the **same GGUF and CPU build settings** for both executables. The helper
uses one question per native request and one native slot, matching GenieX's
independent-question evaluation. Shared-prefix reuse and padded multi-sequence
batches can change quantized arithmetic; a successful load is not a parity test.
The helper compares distributions (`1e-5` tolerance), predicate/choice mapping,
and GenieX's expected-score contract. It intentionally does not compare d1-2B's
native median-score or maximum-probability confidence with GenieX's unchanged
expected-level/normalized-confidence summaries. Numerical comparisons use the
native llama.cpp implementation with the same GGUF, not full-precision
Transformers inference.

For OpenCL GPU comparison, build both executables with `GGML_OPENCL=ON` and
`GGML_HEXAGON=OFF`, then add `--compute gpu --threads 8` on IQ-9075. The helper
selects `GPUOpenCL` in the native server, offloads all model layers, disables flash
attention, and retains the SDK's 512-token GPU physical batch on IQ-9075. The
CLI caps QCS6490 / RB3 Gen 2 GPU batches at 256; the helper's native settings do not
automatically mirror chipset overrides. CPU remains the default; set `--threads`
to match the SDK's default CPU thread count on your host.
Raw CLI debug logs are saved alongside `comparison.json` and `llama-server.log`
so device selection and actual layer offload can be checked. GPU offload does not
imply every operator or the probability summarization runs on GPU.

The real SDK `/v1/decisions` integration test is opt-in too:

```sh
export GENIEX_LIQUID_GGUFS='{"lfm2-d1":"/models/3b/d1-3B-Q4_K_M.gguf","lfm2-d1-omni":"/models/omni/d1-omni-600M-Q8_0.gguf","lfm3-d1":"/models/2b/d1-2B-Q4_K_M.gguf"}'
# Each source directory should contain only the requested language GGUF.
cd cli
go test -cover -run TestLiquidDecisionsCPU -v ./server/handler
```

It imports local files into an isolated test cache, calls the actual handler
through HTTP, and checks all three public question kinds and zero generated
tokens, caller-order/name mapping, and expected-score summaries. It compares all
three kinds through a VLM handle without a projector against the HTTP results,
verifies the native protocol/probability output, and checks image, audio, and
mixed-media rejection. For cache-less models it also checks safe rejection of
LLM/VLM generation and real chat/completions requests, followed by successful
decisions. All-position forward logits are rejected; d1-2B's last-position forward
still works. No network model downloads happen in this test. Normal Go tests skip
it unless the map is provided. The test accepts a subset of the three models.

## CI and scope

PR Check's `test-decision-protocols` job builds this model-free CTest suite on
Linux without the SDK, model downloads, or accelerator libraries. The opt-in
real-GGUF comparisons and SDK integration remain separate checks; record actual
models, quantizations, modalities, backends, and results in the PR's Test plan.

Omni budgets text to at most 16384 tokens, independently of the configured context.
The prepared question must still fit both context and physical batch; a 4096-token
context may reject a truncated long input. GPU defaults generally allow 512 tokens
per question (CLI QCS6490 / RB3 Gen 2: 256), Linux NPU 128, and Windows/Android NPU 1024. CPU defaults follow context
size; SDK-explicit limits are honored. Use CPU for questions exceeding accelerator
limits. d1-2B reserves one final-token logits row per sequence; only Omni reserves
all mask rows. Model-free checks cover `n_seq_max` 1, 2, and 4; real d1-2B checks
load these contexts and require identical predicate/choice/score probabilities.

Media support would need batched embedding assembly and protocol-specific media
formatting/budgets rather than the existing VLM next-token-logits loop. In
particular, Omni reads masks in a complete non-causal batch and accepts images
**or** one audio clip (up to 30 seconds), never both.
