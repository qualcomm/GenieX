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

package com.nexa.demo.utils

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import org.xmlpull.v1.XmlPullParser
import org.xmlpull.v1.XmlPullParserFactory
import java.io.StringReader

/**
 * Utility class to list and download model files from S3 or Hugging Face Hub.
 * Provides fallback mechanism: tries S3 first, then falls back to HF if S3 fails.
 */
object ModelFileListingUtil {
    private const val TAG = "ModelFileListingUtil"
    private const val HF_OWNER = "NexaAI"

    /**
     * Result of file listing operation
     */
    data class FileListResult(
        val files: List<String>,
        val source: Source,
        val repoId: String? = null  // Only set for HF source
    ) {
        enum class Source { S3, HUGGINGFACE, FAILED }
    }

    /**
     * Represents a downloadable file with both S3 and HF URLs
     */
    data class DownloadableFileWithFallback(
        val fileName: String,
        val s3Url: String,
        val hfUrl: String
    )

    /**
     * Extracts repo name from S3 URL.
     * 
     * Examples:
     * - https://...../embeddinggemma-300m-npu-mobile/ -> embeddinggemma-300m-npu-mobile
     * - https://...../LFM2-1.2B-GGUF/LFM2-1.2B-Q4_0.gguf -> LFM2-1.2B-GGUF
     */
    fun extractRepoNameFromS3Url(s3Url: String): String {
        val path = s3Url.removePrefix("https://").removePrefix("http://")
            .substringAfter("/")  // Remove host
            .trimEnd('/')
        
        val segments = path.split("/").filter { it.isNotEmpty() }
        
        // Check if last segment is a file (has extension)
        return if (segments.isNotEmpty()) {
            val lastSegment = segments.last()
            if (lastSegment.contains(".") && !lastSegment.endsWith("/")) {
                // It's a file, get the parent directory name
                if (segments.size >= 2) segments[segments.size - 2] else lastSegment
            } else {
                // It's a directory
                lastSegment
            }
        } else {
            ""
        }
    }

    /**
     * Constructs HF repo ID from S3 URL.
     * @return repo ID in format "NexaAI/{repo_name}"
     */
    fun getHfRepoId(s3Url: String): String {
        val repoName = extractRepoNameFromS3Url(s3Url)
        return "$HF_OWNER/$repoName"
    }

    /**
     * Constructs HF download URL for a file.
     * @param repoId HF repo ID (e.g., "NexaAI/embeddinggemma-300m-npu-mobile")
     * @param fileName File name to download
     * @return Full HF download URL
     */
    fun getHfDownloadUrl(repoId: String, fileName: String): String {
        return "https://huggingface.co/$repoId/resolve/main/$fileName?download=true"
    }

    /**
     * Lists all files under a given S3 base URL, with fallback to Hugging Face.
     *
     * @param baseUrl The S3 base URL
     * @param client OkHttpClient instance
     * @return FileListResult containing files and source information
     */
    suspend fun listFilesWithFallback(
        baseUrl: String,
        client: OkHttpClient
    ): FileListResult = withContext(Dispatchers.IO) {
        // Try S3 first
        val s3Files = listFilesFromS3(baseUrl, client)
        if (s3Files.isNotEmpty()) {
            Log.d(TAG, "Successfully listed ${s3Files.size} files from S3")
            return@withContext FileListResult(s3Files, FileListResult.Source.S3)
        }

        // Fallback to Hugging Face
        Log.w(TAG, "S3 listing failed, falling back to Hugging Face")
        val repoId = getHfRepoId(baseUrl)
        val hfFiles = listFilesFromHuggingFace(repoId, client)
        if (hfFiles.isNotEmpty()) {
            Log.d(TAG, "Successfully listed ${hfFiles.size} files from HuggingFace: $repoId")
            return@withContext FileListResult(hfFiles, FileListResult.Source.HUGGINGFACE, repoId)
        }

        Log.e(TAG, "Failed to list files from both S3 and HuggingFace")
        FileListResult(emptyList(), FileListResult.Source.FAILED)
    }

    /**
     * Lists all files under a given S3 base URL.
     */
    suspend fun listFilesFromS3(baseUrl: String, client: OkHttpClient): List<String> = withContext(Dispatchers.IO) {
        try {
            val urlWithoutProtocol = baseUrl.removePrefix("https://").removePrefix("http://")
            val hostEndIndex = urlWithoutProtocol.indexOf('/')
            if (hostEndIndex == -1) {
                Log.e(TAG, "Invalid S3 URL format: $baseUrl")
                return@withContext emptyList()
            }

            val host = urlWithoutProtocol.substring(0, hostEndIndex)
            val prefix = urlWithoutProtocol.substring(hostEndIndex + 1).trimEnd('/')

            val listUrl = "https://$host/?list-type=2&prefix=$prefix/"
            Log.d(TAG, "Listing S3 bucket: $listUrl")

            val request = Request.Builder()
                .url(listUrl)
                .get()
                .build()

            val response = client.newCall(request).execute()
            if (!response.isSuccessful) {
                Log.e(TAG, "Failed to list S3 bucket: ${response.code}")
                return@withContext emptyList()
            }

            val xmlContent = response.body?.string() ?: return@withContext emptyList()
            Log.d(TAG, "S3 response received, parsing XML...")

            val files = parseS3ListResponse(xmlContent, prefix)
            Log.d(TAG, "Found ${files.size} files from S3: $files")
            files
        } catch (e: Exception) {
            Log.e(TAG, "Error listing S3 files: ${e.message}", e)
            emptyList()
        }
    }

    /**
     * Lists all files from a Hugging Face repository.
     * 
     * @param repoId HF repo ID (e.g., "NexaAI/embeddinggemma-300m-npu-mobile")
     * @param client OkHttpClient instance
     * @return List of file names in the repository
     */
    suspend fun listFilesFromHuggingFace(repoId: String, client: OkHttpClient): List<String> = withContext(Dispatchers.IO) {
        try {
            // HF API endpoint to list files in a repo
            val apiUrl = "https://huggingface.co/api/models/$repoId/tree/main"
            Log.d(TAG, "Listing HuggingFace repo: $apiUrl")

            val request = Request.Builder()
                .url(apiUrl)
                .get()
                .build()

            val response = client.newCall(request).execute()
            if (!response.isSuccessful) {
                Log.e(TAG, "Failed to list HuggingFace repo: ${response.code}")
                return@withContext emptyList()
            }

            val jsonContent = response.body?.string() ?: return@withContext emptyList()
            Log.d(TAG, "HuggingFace response received, parsing JSON...")

            val files = parseHuggingFaceResponse(jsonContent)
            Log.d(TAG, "Found ${files.size} files from HuggingFace: $files")
            files
        } catch (e: Exception) {
            Log.e(TAG, "Error listing HuggingFace files: ${e.message}", e)
            emptyList()
        }
    }

    /**
     * Parses the S3 ListObjectsV2 XML response and extracts file names.
     */
    private fun parseS3ListResponse(xmlContent: String, prefix: String): List<String> {
        val files = mutableListOf<String>()
        try {
            val factory = XmlPullParserFactory.newInstance()
            factory.isNamespaceAware = false
            val parser = factory.newPullParser()
            parser.setInput(StringReader(xmlContent))

            var eventType = parser.eventType
            var currentTag = ""
            val prefixWithSlash = if (prefix.endsWith("/")) prefix else "$prefix/"

            while (eventType != XmlPullParser.END_DOCUMENT) {
                when (eventType) {
                    XmlPullParser.START_TAG -> {
                        currentTag = parser.name
                    }
                    XmlPullParser.TEXT -> {
                        if (currentTag == "Key") {
                            val key = parser.text
                            if (key.startsWith(prefixWithSlash) && key.length > prefixWithSlash.length) {
                                val relativePath = key.removePrefix(prefixWithSlash)
                                if (!relativePath.endsWith("/") && relativePath.isNotEmpty()) {
                                    files.add(relativePath)
                                }
                            }
                        }
                    }
                    XmlPullParser.END_TAG -> {
                        currentTag = ""
                    }
                }
                eventType = parser.next()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing S3 XML response: ${e.message}", e)
        }
        return files
    }

    /**
     * Parses the Hugging Face API JSON response and extracts file names.
     * The response is an array of objects with "path" and "type" fields.
     */
    private fun parseHuggingFaceResponse(jsonContent: String): List<String> {
        val files = mutableListOf<String>()
        try {
            val jsonArray = JSONArray(jsonContent)
            for (i in 0 until jsonArray.length()) {
                val item = jsonArray.getJSONObject(i)
                val type = item.optString("type", "")
                val path = item.optString("path", "")
                
                // Only include files, not directories
                if (type == "file" && path.isNotEmpty()) {
                    files.add(path)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing HuggingFace JSON response: ${e.message}", e)
        }
        return files
    }

    /**
     * Builds downloadable file entries with both S3 and HF fallback URLs.
     * 
     * @param baseUrl S3 base URL
     * @param fileNames List of file names to download
     * @return List of DownloadableFileWithFallback entries
     */
    fun buildDownloadableFilesWithFallback(
        baseUrl: String,
        fileNames: List<String>
    ): List<DownloadableFileWithFallback> {
        val repoId = getHfRepoId(baseUrl)
        return fileNames.map { fileName ->
            val s3Url = if (baseUrl.endsWith("/")) "$baseUrl$fileName" else "$baseUrl/$fileName"
            val hfUrl = getHfDownloadUrl(repoId, fileName)
            DownloadableFileWithFallback(fileName, s3Url, hfUrl)
        }
    }

    /**
     * Gets the HF download URL for a specific .gguf file from its S3 URL.
     * Used for non-NPU models that have direct file URLs.
     * 
     * @param s3FileUrl Full S3 URL to the .gguf file
     * @return HF download URL for the same file
     */
    fun getHfUrlForGgufFile(s3FileUrl: String): String {
        val repoId = getHfRepoId(s3FileUrl)
        val fileName = s3FileUrl.substringAfterLast("/")
        return getHfDownloadUrl(repoId, fileName)
    }
}
