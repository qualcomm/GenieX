# model-manager-core examples

Throwaway utilities that aren't part of the library API. Built with the
normal `cargo run --release --example <name>` workflow.

## `bench_pull`

Times a single `HfHub` download and prints throughput. Handy for:

- Tuning `GENIEX_DL_FILE_CONCURRENCY` / `GENIEX_DL_CHUNK_CONCURRENCY` /
  `GENIEX_DL_CHUNK_SIZE` against your network.
- Sanity-checking that `HTTPS_PROXY` / `HTTP_PROXY` get picked up.
- Smoke-testing a transport change before the full test suite.

```bash
cargo run --release --example bench_pull -p model-manager-core -- \
    unsloth/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

It wipes the output directory on start, so back-to-back runs measure
cold downloads. Set `GENIEX_DATADIR` to control where bytes land; the
default is a temp dir under the OS temp root.
