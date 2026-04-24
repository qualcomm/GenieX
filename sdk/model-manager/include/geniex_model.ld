/*
 * GNU ld version script applied to libgeniex.so when the Rust model
 * manager is linked in (GENIEX_MODEL_MANAGER=ON; see sdk/src/CMakeLists.txt).
 *
 * `--whole-archive` pulls every symbol from libgeniex_model.a into the
 * shared library's dynamic symbol table, which would otherwise expose
 * the entire Rust runtime + hf-hub + ring + ureq dependency graph as
 * part of libgeniex's public ABI (a binary-size and stability hazard).
 *
 * This script restricts exported symbols to the `geniex_*` prefix.
 * Matching covers:
 *   - the existing C++ API (already marked with default visibility)
 *   - the new `geniex_model_*` FFI surface
 * Everything else is localized (hidden).
 */
{
    global:
        geniex_*;
    local:
        *;
};
