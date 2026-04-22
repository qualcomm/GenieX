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

