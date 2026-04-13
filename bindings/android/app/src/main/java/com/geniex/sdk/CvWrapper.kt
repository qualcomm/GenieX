package com.geniex.sdk

import android.system.Os
import com.geniex.sdk.GeniexSdk.Companion.KEY_NPU_LIB_FOLDER_PATH
import com.geniex.sdk.bean.CVCreateInput
import com.geniex.sdk.bean.CVResult
import com.geniex.sdk.bean.RerankerCreateInput
import com.geniex.sdk.jni.Cv
import com.geniex.sdk.utils.ModeConfigUtil
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.Closeable

/**
 * CvWrapper - High-level API for Computer Vision (CV) operations
 * Provides coroutine-based API for image processing tasks on Android
 */
class CvWrapper private constructor(
    private val modelPath: String,
    private val tokenizerPath: String?,
    private val device: String?,
    private val dispatcher: CoroutineDispatcher
) : Closeable {

    // Native CV bridge instance
    private val cv = Cv()
    private var handle: Long = 0

    companion object {
        @JvmStatic
        fun builder() = Builder()
    }

    /**
     * Builder for creating CvWrapper instances
     */
    class Builder {
        private var cvCreateInput: CVCreateInput? = null
        var modelPath: String? = null
        var tokenizerPath: String? = null
        var device: String? = null
        var dispatcher: CoroutineDispatcher = Dispatchers.IO

        fun modelPath(path: String) = apply { this.modelPath = path }
        fun tokenizerPath(path: String) = apply { this.tokenizerPath = path }
        fun device(dev: String) = apply { this.device = dev }
        fun dispatcher(dispatcher: CoroutineDispatcher) = apply { this.dispatcher = dispatcher }

        fun createInput(cvCreateInput: CVCreateInput) =
            apply {
                val npu_model_folder_path = ModeConfigUtil.getNpuModelFolderPath(
                    cvCreateInput.plugin_id,
                    cvCreateInput.config
                )
                val npu_lib_folder_path = ModeConfigUtil.getNpuLibFolderPath(cvCreateInput.config)
                this.cvCreateInput = cvCreateInput.copy(
                    config = cvCreateInput.config.copy(
                        npu_lib_folder_path = npu_lib_folder_path,
                        npu_model_folder_path = npu_model_folder_path
                    )
                )
            }

        /**
         * Build the CvWrapper instance and initialize the native handle
         * @return Result containing the wrapper or error
         */
        suspend fun build(): Result<CvWrapper> = withContext(dispatcher) {
            try {
                val wrapper = CvWrapper(
                    cvCreateInput?.model_name ?: throw IllegalArgumentException("model_name required"),
                    tokenizerPath,
                    device,
                    dispatcher
                )
                wrapper.handle = wrapper.cv.create(input = cvCreateInput!!)
                if (wrapper.handle == 0L) {
                    throw IllegalStateException("Failed to create CV model")
                }
                Result.success(wrapper)
            } catch (e: Exception) {
                Result.failure(e)
            }
        }
    }

    /**
     * Perform CV inference on image (e.g., OCR, object detection)
     * @param imagePath Path to image file (jpg, png, etc.)
     * @return Result containing inference result text (e.g., OCR text)
     */
    suspend fun infer(imagePath: String): Result<ArrayList<CVResult>> = withContext(dispatcher) {
        if (handle == 0L) {
            return@withContext Result.failure(IllegalStateException("CV not initialized"))
        }
        runCatching {
            cv.infer(handle, imagePath)
                ?: throw IllegalStateException("CV inference failed or returned null")
        }
    }

    override fun close() {
        destroy()
    }

    /**
     * Clean up native resources
     * Should be called when done with the CV model
     */
    fun destroy(): Int {
        var result = 0
        if (handle != 0L) {
            result = cv.destroy(handle)
            handle = 0L
        }
        return result
    }
}

