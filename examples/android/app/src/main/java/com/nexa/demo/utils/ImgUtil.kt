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

package com.nexa.demo.utils;

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Matrix
import android.media.ExifInterface
import android.os.Build
import android.provider.MediaStore
import androidx.core.graphics.createBitmap
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class ImgUtil {
    companion object {
        /** Downscale imageFile to maxSize on longest edge, rotate upright, save to outFile (JPEG/WebP), return result file */
        fun downscaleAndSave(
            imageFile: File,
            outFile: File,
            maxSize: Int = 448,
            format: Bitmap.CompressFormat = Bitmap.CompressFormat.JPEG,
            quality: Int = 90
        ): File {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(imageFile.absolutePath, bounds)

            val inSample = run {
                val (h, w) = bounds.outHeight to bounds.outWidth
                var s = 1
                var halfH = h / 2
                var halfW = w / 2
                while (halfH / s >= maxSize && halfW / s >= maxSize) s *= 2
                s
            }

            val opts = BitmapFactory.Options().apply {
                inJustDecodeBounds = false
                inSampleSize = inSample
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            var bmp = BitmapFactory.decodeFile(imageFile.absolutePath, opts) ?: error("decode fail")

            val exif = ExifInterface(imageFile.absolutePath)
            val orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
            )
            val matrix = Matrix().apply {
                when (orientation) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> postRotate(90f)
                    ExifInterface.ORIENTATION_ROTATE_180 -> postRotate(180f)
                    ExifInterface.ORIENTATION_ROTATE_270 -> postRotate(270f)
                    ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> postScale(-1f, 1f)
                    ExifInterface.ORIENTATION_FLIP_VERTICAL -> postScale(1f, -1f)
                    ExifInterface.ORIENTATION_TRANSPOSE -> {
                        postRotate(90f); postScale(-1f, 1f)
                    }

                    ExifInterface.ORIENTATION_TRANSVERSE -> {
                        postRotate(270f); postScale(-1f, 1f)
                    }
                }
            }
            if (!matrix.isIdentity) {
                val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
                if (rotated !== bmp) {
                    bmp.recycle(); bmp = rotated
                }
            }

            val scale = maxOf(bmp.width, bmp.height).toFloat() / maxSize
            val targetW = (bmp.width / scale).toInt().coerceAtLeast(1)
            val targetH = (bmp.height / scale).toInt().coerceAtLeast(1)
            val resized = if (bmp.width != targetW || bmp.height != targetH)
                Bitmap.createScaledBitmap(bmp, targetW, targetH, true) else bmp
            if (resized !== bmp) bmp.recycle()

            // 6) Re-compress and save to outFile (this step actually reduces file size)
            FileOutputStream(outFile).use { fos ->
                resized.compress(format, quality, fos)
            }
            if (!resized.isRecycled) resized.recycle()
            return outFile
        }

        fun squareCrop(imageFile: File, outFile: File, size: Int = 448): File {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(imageFile.absolutePath, bounds)

            val options = BitmapFactory.Options().apply {
                inSampleSize = 1
                inJustDecodeBounds = false
            }
            val bitmap = BitmapFactory.decodeFile(imageFile.absolutePath, options)
            val cropped = createBitmap(size, size)
            val canvas = Canvas(cropped)
            canvas.drawBitmap(
                bitmap,
                (size - bitmap.width).toFloat() / 2,
                (size - bitmap.height).toFloat() / 2,
                null
            )
            if (!bitmap.isRecycled) bitmap.recycle()
            FileOutputStream(outFile).use { fos ->
                cropped.compress(Bitmap.CompressFormat.JPEG, 100, fos)
            }
            if (!cropped.isRecycled) cropped.recycle()
            return outFile
        }

        fun saveImageToGallery(context: Context, imgPath: String) {
            saveImageToGallery(context, BitmapFactory.decodeFile(imgPath))
        }

        fun saveImageToGallery(context: Context, file: File) {
            saveImageToGallery(context, BitmapFactory.decodeFile(file.absolutePath))
        }

        fun saveImageToGallery(context: Context, bitmap: Bitmap) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val resolver = context.contentResolver
                val contentValues = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, "Image_${System.currentTimeMillis()}.jpg")
                    put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                    put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/Saved Images")
                }

                val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
                uri?.let {
                    resolver.openOutputStream(it).use { outputStream ->
                        if (!bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream!!)) {
                            // throw IOException("Unable to save image")
                        }
                        outputStream.close()
                    }
                }
            } else {
                // Old Android save method
                // Please implement according to your needs
            }
        }
    }
}
