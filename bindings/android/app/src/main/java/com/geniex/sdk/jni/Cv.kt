package com.geniex.sdk.jni

import com.geniex.sdk.bean.CVCreateInput
import com.geniex.sdk.bean.CVResult
import com.geniex.sdk.bean.ProfilingData

/**
 * JNI interface for Computer Vision operations
 */
internal class Cv {
    // Create CV model handle
    external fun create(input: CVCreateInput): Long
    // Destroy CV model handle
    external fun destroy(handle: Long): Int
    // Perform CV inference on image (e.g., OCR, object detection)
    external fun infer(handle: Long, imagePath: String): ArrayList<CVResult>?
    // Get profiling data for CV operations
    external fun getProfilingData(handle: Long): ProfilingData
}

