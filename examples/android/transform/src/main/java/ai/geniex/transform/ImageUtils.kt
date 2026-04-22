package ai.geniex.transform

import android.util.Base64
import java.io.File
import java.io.FileOutputStream

class ImageUtils {
    companion object {
        fun saveBase64ToFile(base64String: String?, filePath: String): Boolean {
            return saveBase64ToFile(base64String, File(filePath))
        }

        fun saveBase64ToFile(base64String: String?, file: File): Boolean {
            return try {
                val pureBase64 = base64String?.substringAfter(",")
                val decodedBytes = Base64.decode(pureBase64, Base64.DEFAULT)
                FileOutputStream(file).use { fos ->
                    fos.write(decodedBytes)
                }
                true
            } catch (e: Exception) {
                e.printStackTrace()
                false
            }
        }
    }
}

