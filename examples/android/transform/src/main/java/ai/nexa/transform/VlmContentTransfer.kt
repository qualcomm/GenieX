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

package ai.nexa.transform

import android.content.Context
import com.geniex.sdk.bean.VlmContent
import java.io.File

class VlmContentTransfer(context: Context, val content: VlmContent) {

    private var filesDir: File = File(context.filesDir, "nexa_vlm_files")

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
