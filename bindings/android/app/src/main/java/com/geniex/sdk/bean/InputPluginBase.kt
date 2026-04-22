package com.geniex.sdk.bean

interface InputPluginBase {

    /**
     * [PluginIdValue] to use for the model
     */
    val plugin_id: String?

    /**
     * Device to use for the model, NULL for default device.
     * When using the [PluginIdValue.LLAMA_CPP] plugin, The default value is [DeviceIdValue.CPU],
     * you can use either the [DeviceIdValue.GPU] or the [DeviceIdValue.NPU].
     */
    val device_id: String?
}

/**
 * Device identifiers for model execution.
 * These are user-friendly names that get translated to internal device strings.
 */
enum class DeviceIdValue(val value: String?) {
    /** Default CPU device */
    CPU(null),
    /** GPU device (OpenCL) */
    GPU("gpu"),
    /** NPU device (Hexagon) - uses all available NPU cores */
    NPU("dev0")
}

enum class PluginIdValue(val value: String?) {
    LLAMA_CPP("llama_cpp"), QAIRT("qairt")
}