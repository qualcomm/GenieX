package ai.geniex.transform

import android.content.Context
import com.geniex.sdk.bean.VlmContent
import java.io.File

class VlmContentTransfer(context: Context, val content: VlmContent) {

    private var filesDir: File = File(context.filesDir, "geniex_vlm_files")

    init {
        if (!filesDir.exists()) {
            filesDir.mkdirs()
        }
    }
    suspend fun forBase64(): VlmContent? {
        val imageFile = File(filesDir, "${System.currentTimeMillis()}.jpg")
        val result = ImageUtils.saveBase64ToFile(content.text, imageFile)
        if (result) {
            return VlmContent(content.type, imageFile.absolutePath)
        }
        return null
    }

    suspend fun forUrl(): VlmContent? {
        val imageFile = File(filesDir, "${System.currentTimeMillis()}.jpg")
        val result = DownloadUtils.downloadImage(content.text, imageFile)
        if (result) {
            return VlmContent(content.type, imageFile.absolutePath)
        }
        return null
    }
}
