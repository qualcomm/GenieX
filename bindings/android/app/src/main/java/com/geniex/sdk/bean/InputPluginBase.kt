package com.geniex.sdk.bean

interface InputPluginBase {

    /**
     * [RuntimeIdValue] to use for the model.
     */
    val runtime_id: String?

    /**
     * Compute unit alias to use for this model. `null` / empty selects the
     * runtime's preferred default — `HYBRID` for `llama_cpp` and `NPU`
     * for `qairt`. See [ComputeUnitValue] for the available aliases.
     */
    val compute_unit: String?
}

/**
 * User-facing compute unit aliases. The JNI layer (`jniutils::resolve_device`)
 * translates these to the concrete `compute_unit` and `n_gpu_layers` the
 * SDK runtimes consume. Mirrors `bindings/go/device.go` and
 * `bindings/python/geniex/auto.py`.
 *
 * - [CPU]    — no accelerator, forces `nGpuLayers = 0`.
 * - [GPU]    — Adreno via OpenCL (`llama_cpp`).
 * - [NPU]    — pinned single HTP session (`HTP0`). Deterministic but
 *              slower than [HYBRID] on LLM workloads.
 * - [HYBRID] — `llama_cpp` per-tensor HTP+CPU scheduler. Leaves
 *              `compute_unit` empty and passes `nGpuLayers` through
 *              (-1 = all layers); the fast path on Snapdragon. No-op for
 *              `qairt`, which has only an NPU compute unit.
 */
enum class ComputeUnitValue(val value: String?) {
    /** Pure CPU inference. */
    CPU("cpu"),

    /** GPU (Adreno/OpenCL). */
    GPU("gpu"),

    /** Pinned Hexagon NPU session. */
    NPU("npu"),

    /** Per-tensor HTP+CPU hybrid (llama_cpp fast path). */
    HYBRID("hybrid")
}

enum class RuntimeIdValue(val value: String?) {
    LLAMA_CPP("llama_cpp"), QAIRT("qairt")
}
