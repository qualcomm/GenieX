// Copyright 2024-2026 Nexa AI, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.nexa.demo.bean

import android.annotation.SuppressLint
import android.content.Context
import android.text.TextUtils
import com.nexa.demo.FileConfig
import com.nexa.demo.utils.ModelFileListingUtil
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

/**
 * Extended downloadable file with fallback URL support.
 * Primary URL is S3, fallback URL is Hugging Face.
 */
data class DownloadableFileWithFallback(
    val file: File,
    val primaryUrl: String,      // S3 URL
    val fallbackUrl: String      // HuggingFace URL
)

@SuppressLint("UnsafeOptInUsageError")
@Serializable
data class ModelData(
    val id: String,
    val displayName: String,
    val modelName: String,
    /**
     * support plugin_id
     * 0: default, will use cpu;
     * 0x1: cpu, 0x10:gpu, 0x100:npu
     * cpu: 1,
     * gpu: 16,
     * npu: 256,
     * cpu_gpu: 17,
     * cpu_npu: 257,
     * gpu_npu: 272,
     * cpu_gpu_npu: 273
     */
    val pluginIds: Int? = 0,
    val baseUrl: String? = null,
    val mmprojOrTokenName: String? = null,
    val tokenName: String = "",
    val embeddingName: String = "",
    val extConfigName: String = "",
    val modelUrl: String? = null,
    val mmprojOrTokenUrl: String? = null,
    val embeddingUrl: String? = null,
    val extConfigUrl: String? = null,
    val sizeGb: Double? = 0.0,
    val params: String? = null,
    val features: List<String>? = null,
    val type: String? = null,
    /**
     * Used to indicate the version number of ModelData, mainly for handling the storage location of downloaded files.
     * 0: Default value — files are stored directly under the files/models directory.
     * 1: Files are stored under the files/models/modelId/ directory.
     */
    val versionCode: Int? = 0,
    // NPU-Vision-name
    val patchEmbedName: String = "",
    val vitModelName: String = "",
    val vitConfigFileName: String = "",
    val audioEncoderHelper0Name: String = "",
    val audioEncoderHelper1Name: String = "",
    val audioEncoderModelName: String = "",
    val audioEncoderConfigFileName: String = "",
    // NPU-Vision-url
    val tokenUrl: String? = null,
    val patchEmbedPathUrl: String? = null,
    val vitModelPathUrl: String? = null,
    val vitConfigFilePathUrl: String? = null,
    val audioEncoderHelper0PathUrl: String? = null,
    val audioEncoderHelper1PathUrl: String? = null,
    val audioEncoderModelPathUrl: String? = null,
    val audioEncoderConfigFilePathUrl: String? = null,
    val files: ArrayList<DownloadFileConfig>? = null
) {
    var isSupport = true
}

fun ModelData.modelDir(context: Context): File =
    if (versionCode == 1) {
        File(FileConfig.modelsDir(context), id).apply { if (!exists()) mkdirs() }
    } else {
        FileConfig.modelsDir(context)
    }


fun ModelData.modelFile(context: Context): File? =
    modelUrl?.takeIf { it.isNotBlank() }?.let {
        File(modelDir(context), modelName)
    }

fun ModelData.mmprojTokenFile(context: Context): File? =
    mmprojOrTokenUrl?.takeIf { it.isNotBlank() }?.let {
        File(modelDir(context), mmprojOrTokenName)
    }

fun ModelData.tokenFile(context: Context): File? =
    tokenUrl?.takeIf { it.isNotBlank() }?.let {
        File(modelDir(context), tokenName)
    }

fun ModelData.embederFile(context: Context): File? =
    embeddingUrl?.takeIf { it.isNotBlank() }?.let {
        File(modelDir(context), embeddingName)
    }

fun ModelData.extConfigFile(context: Context): File? =
    extConfigUrl?.takeIf { it.isNotBlank() }?.let {
        File(modelDir(context), extConfigName)
    }

fun ModelData.patchEmbedFile(context: Context): File? =
    patchEmbedPathUrl?.takeIf { it.isNotBlank() }?.let {
        File(modelDir(context), patchEmbedName)
    }

fun ModelData.vitModelFile(context: Context): File? =
    vitModelPathUrl?.takeIf { it.isNotBlank() }?.let {
        File(modelDir(context), vitModelName)
    }

fun ModelData.vitConfigFile(context: Context): File? =
    vitConfigFilePathUrl?.takeIf { it.isNotBlank() }?.let {
        File(modelDir(context), vitConfigFileName)
    }

fun ModelData.audioEncoderHelper0File(context: Context): File? =
    audioEncoderHelper0PathUrl?.takeIf { it.isNotBlank() }?.let {
        File(modelDir(context), audioEncoderHelper0Name)
    }

fun ModelData.audioEncoderHelper1File(context: Context): File? =
    audioEncoderHelper1PathUrl?.takeIf { it.isNotBlank() }?.let {
        File(modelDir(context), audioEncoderHelper1Name)
    }

fun ModelData.audioEncoderModelFile(context: Context): File? =
    audioEncoderModelPathUrl?.takeIf { it.isNotBlank() }?.let {
        File(modelDir(context), audioEncoderModelName)
    }

fun ModelData.audioEncoderConfigFile(context: Context): File? =
    audioEncoderConfigFilePathUrl?.takeIf { it.isNotBlank() }?.let {
        File(modelDir(context), audioEncoderConfigFileName)
    }

private fun ModelData.getRealUrl(url: String) = if (TextUtils.isEmpty(baseUrl)) {
    url
} else {
    if (url.startsWith("http://", true) || url.startsWith("https://", true)) {
        url
    } else {
        if (baseUrl!!.endsWith("/")) {
            "$baseUrl$url"
        } else {
            "$baseUrl/$url"
        }
    }
}

fun ModelData.downloadableFiles(modelDir: File): List<DownloadableFile> = listOfNotNull(
    modelUrl?.takeIf { it.isNotBlank() }?.let {
        DownloadableFile(File(modelDir, modelName), getRealUrl(it))
    },
    mmprojOrTokenUrl?.takeIf { it.isNotBlank() }?.let {
        DownloadableFile(File(modelDir, mmprojOrTokenName), getRealUrl(it))
    },
    tokenUrl?.takeIf { it.isNotBlank() }?.let {
        DownloadableFile(File(modelDir, tokenName), getRealUrl(it))
    },
    embeddingUrl?.takeIf { it.isNotBlank() }?.let {
        DownloadableFile(File(modelDir, embeddingName), getRealUrl(it))
    },
    extConfigUrl?.takeIf { it.isNotBlank() }?.let {
        DownloadableFile(File(modelDir, extConfigName), getRealUrl(it))
    },
    patchEmbedPathUrl?.takeIf { it.isNotBlank() }?.let {
        DownloadableFile(File(modelDir, patchEmbedName), getRealUrl(it))
    },
    vitModelPathUrl?.takeIf { it.isNotBlank() }?.let {
        DownloadableFile(File(modelDir, vitModelName), getRealUrl(it))
    },
    vitConfigFilePathUrl?.takeIf { it.isNotBlank() }?.let {
        DownloadableFile(File(modelDir, vitConfigFileName), getRealUrl(it))
    },
    audioEncoderHelper0PathUrl?.takeIf { it.isNotBlank() }?.let {
        DownloadableFile(File(modelDir, audioEncoderHelper0Name), getRealUrl(it))
    },
    audioEncoderHelper1PathUrl?.takeIf { it.isNotBlank() }?.let {
        DownloadableFile(File(modelDir, audioEncoderHelper1Name), getRealUrl(it))
    },
    audioEncoderModelPathUrl?.takeIf { it.isNotBlank() }?.let {
        DownloadableFile(File(modelDir, audioEncoderModelName), getRealUrl(it))
    },
    audioEncoderConfigFilePathUrl?.takeIf { it.isNotBlank() }?.let {
        DownloadableFile(File(modelDir, audioEncoderConfigFileName), getRealUrl(it))
    }
).let {
    val temp = arrayListOf<DownloadableFile>()
    files?.takeIf { it.isNotEmpty() }?.let { allFiles ->
        allFiles.forEach { fileConfig ->
            temp.add(
                DownloadableFile(
                    File(
                        modelDir,
                        fileConfig.path + File.separator + fileConfig.name
                    ), if (TextUtils.isEmpty(fileConfig.url)) {
                        getRealUrl(fileConfig.name)
                    } else {
                        getRealUrl(fileConfig.url!!)
                    }
                )
            )
        }
    }
    it + temp
}


fun ModelData.allModelFilesExist(modelDir: File): Boolean {
    val files = this.downloadableFiles(modelDir).map { it.file }
    return files.all { it.exists() && it.length() > 0 }
}

fun ModelData.getNonExistModelFile(modelDir: File): String? {
    this.downloadableFiles(modelDir).forEach {
        if (!(it.file.exists() && it.file.length() > 0)) {
            return it.file.absolutePath.replace("/data/user/0", "/data/data")
        }
    }
    return null
}

fun ModelData.getNexaManifest(context: Context): NexaManifestBean? {
    try {
        val str = File(modelDir(context), "nexa.manifest").bufferedReader().use { it.readText() }
        return Json {
            ignoreUnknownKeys = true
        }.decodeFromString<NexaManifestBean>(str)
    } catch (e: Exception) {
        e.printStackTrace()
        return null
    }
}

fun ModelData.getSupportPluginIds(): ArrayList<String> {
    val pluginIds = arrayListOf<String>()
    if (this.pluginIds == 0) {
        pluginIds.add("cpu")
    } else {
        if (this.pluginIds!! and 0x100 == 0x100) {
            pluginIds.add("npu")
        }
        if (this.pluginIds and 0x10 == 0x10) {
            pluginIds.add("gpu")
        }
        if (this.pluginIds and 0x1 == 0x1) {
            pluginIds.add("cpu")
        }
    }
    return pluginIds
}

/**
 * Checks if this model is an NPU model.
 * NPU models are identified by:
 * - "NPU" (case-insensitive) in the model id, OR
 * - ".nexa" suffix in modelName
 */
fun ModelData.isNpuModel(): Boolean {
    return id.contains("NPU", ignoreCase = true) || 
           id.contains("npu", ignoreCase = true) ||
           modelName.endsWith(".nexa", ignoreCase = true)
}

/**
 * Downloads files for NPU models using a dynamically fetched file list.
 * This overload is used when files are fetched from S3/HF listing instead of
 * being specified in model_list.json.
 *
 * @param modelDir Directory where model files will be stored
 * @param npuFileNames List of file names fetched from listing
 * @return List of DownloadableFile objects representing files to download
 */
fun ModelData.downloadableFilesWithNpuList(
    modelDir: File,
    npuFileNames: List<String>
): List<DownloadableFile> {
    val npuFiles = arrayListOf<DownloadableFile>()
    npuFileNames.forEach { fileName ->
        val url = if (baseUrl.isNullOrEmpty()) {
            fileName
        } else {
            if (baseUrl.endsWith("/")) "$baseUrl$fileName" else "$baseUrl/$fileName"
        }
        npuFiles.add(DownloadableFile(File(modelDir, fileName), url))
    }
    return npuFiles
}

/**
 * Downloads files for NPU models with fallback URL support.
 * Primary URL is S3, fallback URL is Hugging Face.
 *
 * @param modelDir Directory where model files will be stored
 * @param npuFileNames List of file names fetched from listing
 * @param useHfUrls If true, use HF URLs as primary (when S3 listing failed)
 * @return List of DownloadableFileWithFallback objects
 */
fun ModelData.downloadableFilesWithFallback(
    modelDir: File,
    npuFileNames: List<String>,
    useHfUrls: Boolean = false
): List<DownloadableFileWithFallback> {
    val npuFiles = arrayListOf<DownloadableFileWithFallback>()
    val repoId = if (!baseUrl.isNullOrEmpty()) {
        ModelFileListingUtil.getHfRepoId(baseUrl)
    } else {
        "NexaAI/$id"
    }

    npuFileNames.forEach { fileName ->
        val s3Url = if (baseUrl.isNullOrEmpty()) {
            fileName
        } else {
            if (baseUrl.endsWith("/")) "$baseUrl$fileName" else "$baseUrl/$fileName"
        }
        val hfUrl = ModelFileListingUtil.getHfDownloadUrl(repoId, fileName)
        
        // If useHfUrls is true, swap primary and fallback
        if (useHfUrls) {
            npuFiles.add(DownloadableFileWithFallback(File(modelDir, fileName), hfUrl, s3Url))
        } else {
            npuFiles.add(DownloadableFileWithFallback(File(modelDir, fileName), s3Url, hfUrl))
        }
    }

    return npuFiles
}

/**
 * Converts a list of DownloadableFile to DownloadableFileWithFallback.
 * Adds HuggingFace fallback URLs for each file.
 */
fun List<DownloadableFile>.withFallbackUrls(): List<DownloadableFileWithFallback> {
    return map { df ->
        val fallbackUrl = ModelFileListingUtil.getHfUrlForGgufFile(df.url)
        DownloadableFileWithFallback(df.file, df.url, fallbackUrl)
    }
}