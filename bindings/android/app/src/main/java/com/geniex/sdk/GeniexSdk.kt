package com.geniex.sdk

import android.content.Context
import android.content.res.AssetManager
import android.system.Os
import android.util.Log
import com.geniex.sdk.bean.PluginIdValue
import java.io.File
import java.io.IOException
import java.util.zip.ZipEntry
import java.util.zip.ZipFile


class GeniexSdk private constructor() {

    interface InitCallback {
        fun onSuccess()
        fun onFailure(reason: String)
    }

    external fun registerPlugin(pluginLibPath: String): Int

    /**
     * @param callback Use for Checking the context environment of GeniexSdk.When an exception occurs,
     * the [InitCallback.onFailure] will be invoked.
     */
    fun init(context: Context, callback: InitCallback? = null) {
        val nativeLibPath = context.applicationInfo.nativeLibraryDir
        Os.setenv("ADSP_LIBRARY_PATH", nativeLibPath, true)

        extractHtpAssets(context)

        val exceptionResult = StringBuilder()
        arrayOf(
            PluginIdValue.LLAMA_CPP.value,
            PluginIdValue.QAIRT.value
        ).forEach { pluginName ->
            File(
                nativeLibPath,
                "libgeniex_plugin_${pluginName}.so"
            ).let { pluginSoFile ->
                if (pluginSoFile.exists()) {
                    pluginSoFile.absolutePath.let { pluginPath ->
                        Log.d(TAG, "Loading plugin: $pluginPath")
                        if (registerPlugin(pluginPath) != 0) {
                            exceptionResult.append("Cannot registerPlugin $pluginName\n")
                        }
                    }
                } else {
                    exceptionResult.append("Cannot find ${pluginSoFile.name} in $nativeLibPath\n")
                }
            }
        }
        val soFile = checkSoFile(context, nativeLibPath)
        if (soFile != null) {
            exceptionResult.append("Cannot find $soFile in $nativeLibPath")
        }

        if (exceptionResult.isEmpty()) {
            callback?.onSuccess()
        } else {
            callback?.onFailure(exceptionResult.toString())
        }
    }

    /**
     * Extracts HTP runtime libraries from bundled assets into app-internal storage,
     * preserving the directory structure (htp-files/, htp-files-v81/, htp-files-v85/).
     * The C++ QNN plugin selects the correct version at runtime based on detected HTP arch.
     */
    private fun extractHtpAssets(context: Context) {
        val htpBaseDir = File(context.filesDir, "npu")
        val versionFile = File(htpBaseDir, ".htp_version")
        val currentVersion = try {
            val info = context.packageManager.getPackageInfo(context.packageName, 0)
            "${info.versionCode}_${info.lastUpdateTime}"
        } catch (e: Exception) {
            "unknown"
        }

        if (versionFile.exists() && versionFile.readText() == currentVersion) {
            Log.d(TAG, "HTP assets already extracted for version $currentVersion")
            Os.setenv(KEY_QNN_HTP_PATH, htpBaseDir.absolutePath, true)
            return
        }

        val assetManager = context.assets
        var totalExtracted = 0
        for (dirName in HTP_ASSET_DIRS) {
            try {
                val assetFiles = assetManager.list("npu/$dirName") ?: continue
                if (assetFiles.isEmpty()) continue
                val targetDir = File(htpBaseDir, dirName)
                targetDir.mkdirs()
                for (fileName in assetFiles) {
                    val targetFile = File(targetDir, fileName)
                    assetManager.open("npu/$dirName/$fileName").use { input ->
                        targetFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                }
                totalExtracted += assetFiles.size
                Log.d(TAG, "Extracted ${assetFiles.size} HTP files to ${targetDir.absolutePath}")
            } catch (e: IOException) {
                Log.d(TAG, "No HTP assets for $dirName: ${e.message}")
            }
        }

        if (totalExtracted > 0) {
            versionFile.parentFile?.mkdirs()
            versionFile.writeText(currentVersion)
        }
        Os.setenv(KEY_QNN_HTP_PATH, htpBaseDir.absolutePath, true)
        Log.d(TAG, "GENIEX_QNN_HTP_PATH set to ${htpBaseDir.absolutePath}")
    }

    private fun checkSoFile(context: Context, nativeLibPath: String): String? {

        val nativeLibPathSoFileNames = File(nativeLibPath).listFiles()?.map {
            it.name
        } ?: arrayListOf<String>()
        val allSoNames = getApkSoFileNames(context)
        if (nativeLibPathSoFileNames.size == allSoNames.size) {
            return null
        }
        var result: String? = null
        allSoNames.forEach { name ->
            if (!nativeLibPathSoFileNames.contains(name)) {
                result = name
                return@forEach
            }
        }
        return result
    }

    private fun getApkSoFileNames(context: Context): ArrayList<String> {
        val nameList = arrayListOf<String>()
        val apkPath: String = context.packageCodePath
        try {
            val soFilePrefix = "lib/arm64-v8a/"
            val zipFile = ZipFile(apkPath)
            val entries = zipFile.entries()
            while (entries.hasMoreElements()) {
                val entry: ZipEntry = entries.nextElement()
                val entryName = entry.name
                if (entryName.startsWith(soFilePrefix) && !entry.isDirectory) {
                    nameList.add(entryName.replace(soFilePrefix, ""))
                }
            }
            zipFile.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
        return nameList
    }

    companion object {
        private const val TAG = "GeniexSdk"
        internal const val KEY_QNN_HTP_PATH = "GENIEX_QNN_HTP_PATH"
        private val HTP_ASSET_DIRS = listOf("htp-files", "htp-files-v81", "htp-files-v85")
        const val PLUGIN_ID_QAIRT = "qairt"
        const val PLUGIN_ID_LLAMA_CPP = "llama_cpp"

        init {
            System.loadLibrary("npu_jni")
        }

        @Volatile
        private var instance: GeniexSdk? = null

        fun getInstance() = instance ?: synchronized(this) {
            instance ?: GeniexSdk().also { instance = it }
        }

    }
}
