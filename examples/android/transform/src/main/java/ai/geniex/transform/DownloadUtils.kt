package ai.geniex.transform

import android.text.TextUtils
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

class DownloadUtils {
    companion object {
        // Download image to specified path
        fun downloadImage(imageUrl: String?, saveFile: File): Boolean {
            if (TextUtils.isEmpty(imageUrl) || imageUrl?.startsWith("http", true) == false) {
                return false
            }
            return try {
                val url = URL(imageUrl)
                val connection = url.openConnection() as HttpURLConnection
                connection.connectTimeout = 15000
                connection.readTimeout = 15000
                connection.doInput = true
                connection.connect()
                if (connection.responseCode == HttpURLConnection.HTTP_OK) {
                    val inputStream = connection.inputStream
                    // Ensure directory exists
                    saveFile.parentFile?.mkdirs()
                    FileOutputStream(saveFile).use { outputStream ->
                        inputStream.copyTo(outputStream)
                    }
                    inputStream.close()
                    true
                } else {
                    false
                }
            } catch (e: Exception) {
                e.printStackTrace()
                false
            }
        }
    }
}