package com.geniex.sdk

import com.geniex.sdk.bean.ImageGenCreateInput
import com.geniex.sdk.bean.ImageGenImg2ImgInput
import com.geniex.sdk.bean.ImageGenTxt2ImgInput
import com.geniex.sdk.jni.ImgGen
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.Closeable
import java.io.File

class ImgGenWrapper private constructor(
    private val imageGenCreateInput: ImageGenCreateInput,
    private val dispatcher: CoroutineDispatcher
) : Closeable {

    private val imgGen = ImgGen()
    private var handle: Long = 0

    companion object {
        @JvmStatic
        fun builder() = Builder()
    }

    class Builder {
        private var imageGenCreateInput: ImageGenCreateInput? = null
        private var dispatcher: CoroutineDispatcher = Dispatchers.IO

        fun createInput(imageGenCreateInput: ImageGenCreateInput) =
            apply {
                this.imageGenCreateInput = imageGenCreateInput.copy(
                    model_path = File(imageGenCreateInput.model_path).let {
                        if (it.isFile) {
                            it.parentFile!!.absolutePath
                        } else {
                            it.absolutePath
                        }
                    }
                )
            }

        fun dispatcher(dispatcher: CoroutineDispatcher) = apply { this.dispatcher = dispatcher }

        // Build the LlmWrapper instance and initialize the native handle
        suspend fun build(): Result<ImgGenWrapper> = withContext(dispatcher) {
            try {
                val input =
                    imageGenCreateInput ?: throw IllegalArgumentException("modelPath required")
                val wrapper = ImgGenWrapper(input, dispatcher)
                wrapper.handle = wrapper.imgGen.create(input)
                Result.success(wrapper)
            } catch (e: Exception) {
                Result.failure(e)
            }
        }
    }

    suspend fun txt2Img(imageGenTxt2ImgInput: ImageGenTxt2ImgInput): Int {
        return imgGen.txt2Img(imageGenTxt2ImgInput, handle)
    }

    suspend fun img2Img(imageGenImg2ImgInput: ImageGenImg2ImgInput): Int {
        return imgGen.img2Img(imageGenImg2ImgInput, handle)
    }

    override fun close() {
        destroy()
    }

    fun destroy(): Int {
        var result = 0
        if (handle != 0L) {
            result = imgGen.destroy(handle)
            handle = 0L
        }
        return result
    }
}