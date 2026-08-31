# Managed chat-cache lineage

The OpenAI-compatible server owns one cached, mutable model handle. Raw
`GenieX-KeepCache: true` can retain that handle between calls, but it carries no
proof that the next request extends the conversation that produced the state.
The managed protocol adds that proof and commits cache state transactionally.

## Invariants

- There is at most one committed lineage because there is one mutable model
  handle.
- A request has at most one pending transaction.
- Reuse requires the same session, parent revision, model generation, complete
  cache identity and exact transcript prefix.
- Any non-reuse decision resets the model before generation.
- Only a successful generation commits the assistant output and new revision.
- Error, cancellation or disconnect clears both pending and committed lineage
  and resets the model. If reset fails, the handle is destroyed.
- Unmanaged chat, legacy completion and logits traffic clears managed lineage
  before touching the model, including raw `GenieX-KeepCache` requests.
- A model unload, reload or out-of-band reset changes the monotonic generation
  and prevents a planned hit from being committed as reused.

The cache identity includes the model name, full model and tokenizer artifact
digests, runtime, SDK and plugin versions, resolved model/device parameters,
chat template options, grammar settings and reasoning mode. Session is checked
separately. Revisions are SHA-256 digests over a versioned JSON envelope of the
identity and committed messages.

## Protocol boundary

The request headers are `GenieX-Cache-Session` and, after the first successful
call, `GenieX-Cache-Parent`. Managed headers cannot be combined with
`GenieX-KeepCache`.

Version one supports only scalar text content with `system`, `user` and
`assistant` roles. It rejects VLM input, native tool-call messages, separated
reasoning, speculative decoding, system-only warm-up and unknown message forms
before generation.

The final blocking response or final streaming finish chunk contains
`geniex_cache`. Its public vocabulary is:

- status: `cold`, `reused`, or `reset`;
- reason: `first_request`, `exact_extension`, `branch`, `session_switch`, or
  `parent_mismatch`.

A missing final record is an uncommitted request and must not be reused by a
client. Session IDs separate state lineage; they are not authentication,
authorization or tenant isolation.

## Test gates

Handler tests cover deterministic revisions, 1,000 randomized branch
mutations, session switching, stale parents, identity changes, model reload,
abort behavior, response metadata and invalid v1 message forms. Service tests
cover reset generation and destroy-on-reset-failure. CORS tests cover both new
headers.

Run the normal coverage graph. Race coverage is required on Linux because the
Go race detector is not supported on Windows ARM64:

```bash
bazelisk coverage //... --combined_report=lcov
bazelisk test --@io_bazel_rules_go//go/config:race //cli/server/handler:handler_test //cli/server/service:service_test
```
