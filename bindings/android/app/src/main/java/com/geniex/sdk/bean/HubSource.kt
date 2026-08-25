package com.geniex.sdk.bean

/**
 * Mirrors `geniex_HubSource` from sdk/pkg-geniex/include/geniex_model.h.
 * Integer values MUST stay in lock-step with the C enum — the JNI bridge
 * passes `value` directly to the Rust FFI.
 */
enum class HubSource(val value: Int) {
    AUTO(0),
    HUGGINGFACE(1),
    MODELSCOPE(2),
    AIHUB(3),
    VOLCES(4),
    DOCKER(5),
    LOCALFS(127),
}
