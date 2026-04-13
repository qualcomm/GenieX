package com.geniex.sdk.bean

data class CVModelConfig(
    /* Capabilities */
    val capabilities: CVCapability,
    /* detection model path */
    val det_model_path: String,
    /* recognition model path */
    val rec_model_path: String,
    /* Character dictionary path */
    val char_dict_path: String,

    // NPU
    /**
     * NPU loaded model's dir.
     */
    val npu_model_folder_path: String? = null,
    /**
     *
     * NPU native library dir. The default path is
     * @see android.content.pm.ApplicationInfo.nativeLibraryDir
     */
    val npu_lib_folder_path: String? = null
)

enum class CVCapability {
    OCR,
    CLASSIFICATION,
    SEGMENTATION,
    CUSTOM
}