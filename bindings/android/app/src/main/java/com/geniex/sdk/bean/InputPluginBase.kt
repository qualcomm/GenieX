package com.geniex.sdk.bean

interface InputPluginBase {

    /**
     * [PluginIdValue] to use for the model.
     */
    val plugin_id: String?

    /**
     * Device alias to use for this model. `null` / empty selects the
     * plugin's preferred default — `HYBRID` for `llama_cpp` and `NPU`
     * for `qairt`. See [DeviceIdValue] for the available aliases.
     */
    val device_id: String?
}

/**
 * User-facing device aliases. The JNI layer (`jniutils::resolve_device`)
 * translates these to the concrete `device_id` and `n_gpu_layers` the
 * SDK plugins consume. Mirrors `bindings/go/device.go` and
 * `bindings/python/geniex/auto.py`.
 *
 * - [CPU]    — no accelerator, forces `nGpuLayers = 0`.
 * - [GPU]    — Adreno via OpenCL (`llama_cpp`).
 * - [NPU]    — pinned single HTP session (`HTP0`). Deterministic but
 *              slower than [HYBRID] on LLM workloads.
 * - [HYBRID] — `llama_cpp` per-tensor HTP+CPU scheduler. Leaves
 *              `device_id` empty and forces `nGpuLayers = 999`; the
 *              fast path on Snapdragon. No-op for `qairt`, which has
 *              only an NPU device.
 */
enum class DeviceIdValue(val value: String?) {
    /** Pure CPU inference. */
    CPU("cpu"),

    /** GPU (Adreno/OpenCL). */
    GPU("gpu"),

    /** Pinned Hexagon NPU session. */
    NPU("npu"),

    /** Per-tensor HTP+CPU hybrid (llama_cpp fast path). */
    HYBRID("hybrid")
}

enum class PluginIdValue(val value: String?) {
    LLAMA_CPP("llama_cpp"), QAIRT("qairt")
}
