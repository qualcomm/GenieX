# Handoff: HTP weight-buffer / n_ubatch fix on IQ9 (geniex#1683) — round 3, dynamic ladder (verified on-device)

Branch: `fix/htp-weight-buffer-mbuf`. Issue:
https://github.com/qcom-ai-hub/geniex/issues/1683

This supersedes `HANDOFF-1683-htp-mbuf.md` and `HANDOFF-1683-htp-mbuf-followup.md`
(both removed — their round-1/round-2 content is summarized and superseded below).

> Internal device access (SSH tunnel host, credentials, on-device paths) is
> intentionally **not** included here since this repo is public.

## Where we left off (round 2) and what was still wrong

Round 2 landed two fixes in `sdk/plugins/llama_cpp/src/params.cpp` / `vlm.cpp`:
NPU `n_ubatch` default lowered 1024 → 256, and VLM `n_ctx` made device-aware
(4096 on NPU vs. 16384 elsewhere). Re-testing on IQ9 with the packaged `geniex`
CLI showed **zero behavioral change** — gemma-4-E4B and Qwen3-ASR-1.7B still
failed identically to round 1 (same buffer sizes, same SIGABRT-on-teardown for
Qwen3-ASR). The round-2 fix didn't reach the actual failure point.

## Round-3 bisection (raw llama.cpp on-device)

Dropped to raw `llama-cli`/`llama-mtmd-cli` (already deployed at
`/data/llama.cpp` on the device) to bisect without burning more build/deploy
cycles. Needed `ADSP_LIBRARY_PATH=./lib` in addition to `LD_LIBRARY_PATH` for
these raw binaries to open an HTP session at all (the packaged `geniex` CLI
sets this internally via `GENIEX_PLUGIN_PATH`; `third-party/llama.cpp/scripts/snapdragon/run.py`
confirms both env vars are required).

Findings:
- `-c 1024 -ub 128 -b 128` passes cleanly for **both** gemma-4-E4B (clean
  generation, ~33 t/s prompt / ~9.3-9.7 t/s decode) and Qwen3-ASR-1.7B (real
  transcription, no errors) with `GGML_HEXAGON_MBUF=64`.
- Isolated that `n_ctx=1024` was never the real constraint: `n_ctx=4096` (the
  round-2 VLM default) + `n_ubatch=128` + `n_batch=128` also passes. **`n_ubatch=128`
  is the load-bearing parameter**, not `n_ctx` or `n_batch`.
- So the round-2 default of 256 was still too high — the actual minimal viable
  value on this IQ9 unit is **128**.

This confirms the underlying issue isn't really an `n_ubatch` *value* problem —
it's that ggml-hexagon's per-ubatch compute buffer needs one contiguous
fastrpc/CMA allocation, and how much contiguous space is available depends on
what's already resident (model size, KV cache, other sessions). No single
static default is safe for every model/device combo, which is why round 2's
"just lower the constant further" approach kept chasing the wrong number.

## Round-3 fix attempt #1: ladder on `llama_init_from_model()` — didn't work

First cut: `init_context_with_ubatch_ladder()` called `llama_init_from_model()`
and halved `n_ubatch` on retry whenever it returned `nullptr`. Verified on
IQ9 with the packaged CLI — **context creation always succeeded on the first
try** (`n_ubatch=256`), but the real user prompt then failed exactly as
before, just one step later (`geniex_vlm_generate` instead of
`geniex_vlm_create`). No behavioral change.

Root cause: on this backend, `llama_init_from_model()`'s `sched_reserve()`
doesn't actually `fastrpc_mmap` the LM's compute buffer — that's deferred to
the **first real `llama_decode()` call**, which happens inside `generate()`,
well after the ladder's only checkpoint has already returned success. The
ladder was watching the wrong call.

## Round-3 fix attempt #2: warmup decode, and an exception-handling bug

Fix: force that first `llama_decode()` to happen *inside* `create()`, where
the ladder can still see it fail and retry, via a 1-token `warmup_decode()`
right after `llama_init_from_model()` succeeds (then `llama_memory_clear()`
to reset the KV cache before the caller's real generation).

First deploy of this **also** showed no retry — `create()` now failed
instead of `generate()`, but still on the first attempt at `n_ubatch=256`.
Traced it to `third-party/llama.cpp/ggml/src/ggml-hexagon/ggml-hexagon.cpp`:
the `fastrpc_mmap` wrapper **throws `std::runtime_error`** directly on
failure rather than returning an error code. That exception unwound straight
past the ladder's loop (no try/catch), out through `create()`, and was only
caught by `sdk/src/vlm.cpp`'s top-level C-API wrapper — several stack frames
too late for the ladder to react.

Fix: wrapped `warmup_decode()`'s body in `try { ... } catch (const
std::exception&)`, converting the thrown failure into a plain `false` return
the ladder loop already knew how to handle.

## Verified on IQ9 (packaged `geniex` CLI, `--compute npu`)

All three models, rebuilt SDK → CLI → redeployed, same session
(`GGML_HEXAGON_DEVICES=2`, `GGML_HEXAGON_MBUF=64`):

1. **gemma-4-E4B** — first attempt at `n_ubatch=256` throws
   (`ggml-hex: fastrpc_mmap failed`), ladder logs
   `context init/warmup failed at n_ubatch=256; retrying with n_ubatch=128`,
   retry succeeds → `The capital of France is **Paris**.` (9.1 tok/s).
2. **Qwen3-ASR-1.7B** — succeeds on the **first** attempt at `n_ubatch=256`
   (no retry needed — this model was never the one needing 128, confirming
   a single static floor would've over-penalized it); real transcription
   output, 16.6 tok/s, no crash.
3. **gemma-4-E2B** — succeeds on the first attempt as before (no retry log),
   18.0 tok/s — confirms zero regression/cost for models that already fit.

## How the ladder works now

`sdk/plugins/llama_cpp/src/params.cpp`'s `init_context_with_ubatch_ladder()`:

- Builds context params as before (`build_context_params()`, unchanged table
  of per-platform/device starting ceilings — Linux/Windows NPU still starts
  at 256, Android NPU at 1024).
- On NPU, when the caller hasn't pinned `n_ubatch` explicitly: calls
  `llama_init_from_model()`, then a 1-token `warmup_decode()` to force the
  real HTP buffer allocation to happen now. Both the `nullptr`-return path
  and the thrown-exception path count as failure.
- On failure, halves `n_ubatch` and retries, down to a floor of **128** (the
  smallest value confirmed to work for every model tested this session).
- Skips the ladder entirely (single attempt) on CPU/GPU, or when the caller
  pinned `n_ubatch` themselves — an explicit override is a deliberate choice,
  not a default this code should second-guess.
- Logs a warning on each retry (`[Optimise] context init/warmup failed at
  n_ubatch=X; retrying with n_ubatch=Y`) so a field failure is traceable from
  logs alone, and a warmup-decode exception is logged separately
  (`[Optimise] warmup decode threw: ...`).

`sdk/plugins/llama_cpp/src/llm.cpp` and `vlm.cpp` call this helper instead of
`build_context_params()` + `llama_init_from_model()` directly. The
speculative-decoding draft context (`llm.cpp`'s `setup_speculative()`) is
**not** laddered — it already has its own non-fatal fallback path (failure
there just disables speculative decoding and logs a warning).

## Known gaps / not done this round

- `n_batch` was only ever tested at its 2048 default alongside the ladder;
  whether `n_batch` matters independently at `n_ubatch=128` specifically is
  still unconfirmed.
- The ladder floor (128) and starting ceilings (256/1024 per platform) are
  still just the values confirmed on this one IQ9 unit for the 3 models
  tested — if a future model fails even at 128, the ladder returns `nullptr`
  same as before (caller sees `GENIEX_ERROR_COMMON_MODEL_LOAD`); it won't
  retry below the floor.
- Draft/speculative-decoding context creation is unladdered (see above).
- The warmup decode adds one extra `llama_decode()` (and on retry, a full
  context teardown/rebuild) to model-load latency — not benchmarked this
  round; expect it to be small relative to model load time but worth
  profiling before calling this perf-neutral.
- `ggml-hexagon.cpp`'s `opt_mbuf`/`GGML_HEXAGON_MBUF` env var cap (round-1 fix,
  default 64 MiB via `plugin.cpp`) is unrelated to this ladder and unchanged.

## Build/deploy notes carried over from earlier rounds

- Raw `llama-cli`/`llama-mtmd-cli` run non-interactively with `< /dev/null` do
  **not** exit on EOF — they loop indefinitely printing empty prompts. Always
  wrap such invocations in `timeout N` when scripting; a stray one of these
  ran for 75+ minutes last round and filled `/tmp` (17GB tmpfs) with log
  output, producing confusing zero-byte log files for unrelated commands.
- `cmake --build build-linux --target geniex` does **not** rebuild the
  llama_cpp plugin (`geniex_llama_cpp`/`libgeniex_plugin.so` is a separate
  target) — build without a `--target` restriction, or target
  `geniex_llama_cpp` explicitly.
- NFS-mounted Bazel output base crashes the Bazel JVM server silently; use
  `--output_user_root` pointed at local disk.
