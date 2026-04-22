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

import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.io.RandomAccessFile
import java.lang.Long
import java.security.MessageDigest
import java.util.Locale
import java.util.zip.CRC32
import kotlin.ByteArray
import kotlin.CharArray
import kotlin.Exception
import kotlin.Int
import kotlin.String
import kotlin.Throws
import kotlin.also
import kotlin.math.max
import kotlin.math.min
import kotlin.text.uppercase


class MD5Utils {
    companion object {
        private val HEX_ARRAY: CharArray = "0123456789ABCDEF".toCharArray()

        fun bytesToHex(bytes: ByteArray?): String? {
            if (bytes == null) return null

            val hexChars = CharArray(bytes.size * 2)
            for (i in bytes.indices) {
                val v = bytes[i].toInt() and 0xFF
                hexChars[i * 2] = HEX_ARRAY[v ushr 4]
                hexChars[i * 2 + 1] = HEX_ARRAY[v and 0x0F]
            }
            return String(hexChars)
        }

        fun getFileMD5(file: File): String? {
            try {
                val md5 = MessageDigest.getInstance("MD5")
                val fis = FileInputStream(file)
                val buffer = ByteArray(1024)
                var length: Int
                while ((fis.read(buffer).also { length = it }) != -1) {
                    md5.update(buffer, 0, length)
                }
                fis.close()
                val digest = md5.digest()
                val sb = StringBuilder()
                for (b in digest) {
                    sb.append(Integer.toHexString((b.toInt() and 0xFF) or 0x100).substring(1, 3))
                }
                return sb.toString().uppercase(Locale.getDefault())
            } catch (e: Exception) {
                e.printStackTrace()
                return null
            }
        }

        fun getFileSHA256(file: File): String? {
            try {
                val digest = MessageDigest.getInstance("SHA-256")
                val fis = FileInputStream(file)
                val buffer = ByteArray(1024)
                var length: Int
                while ((fis.read(buffer).also { length = it }) != -1) {
                    digest.update(buffer, 0, length)
                }
                fis.close()
                val hash = digest.digest()

                val hexString = java.lang.StringBuilder()
                for (b in hash) {
                    val hex = Integer.toHexString(0xff and b.toInt())
                    if (hex.length == 1) hexString.append('0')
                    hexString.append(hex)
                }
                return hexString.toString()
            } catch (e: java.lang.Exception) {
                e.printStackTrace()
                return null
            }
        }

        fun getFileCRC32(file: File): String? {
            try {
                val crc32 = CRC32()
                val fis = FileInputStream(file)
                val buffer = ByteArray(1024)
                var length: Int
                while ((fis.read(buffer).also { length = it }) != -1) {
                    crc32.update(buffer, 0, length)
                }
                fis.close()
                val value = crc32.value
                return Long.toHexString(value).uppercase(Locale.getDefault())
            } catch (e: java.lang.Exception) {
                e.printStackTrace()
                return null
            }
        }

        fun getQuickCRC32(file: File): kotlin.Long {
            try {
                val crc32 = CRC32()
                val raf = RandomAccessFile(file, "r")
                val fileSize = raf.length()
                val buffer = ByteArray(1024 * 1024)

                // start
                raf.seek(0)
                var read = raf.read(buffer)
                if (read > 0) crc32.update(buffer, 0, read)

                // middle
                if (fileSize > 2 * buffer.size) {
                    raf.seek(fileSize / 2)
                    read = raf.read(buffer)
                    if (read > 0) crc32.update(buffer, 0, read)
                }

                // end
                if (fileSize > buffer.size) {
                    raf.seek(fileSize - buffer.size)
                    read = raf.read(buffer)
                    if (read > 0) crc32.update(buffer, 0, read)
                }

                raf.close()
                return crc32.getValue()
            } catch (e: java.lang.Exception) {
                e.printStackTrace()
                return -1
            }
        }

        @Throws(IOException::class)
        fun getFilePartialHash(file: File, partSize: kotlin.Long): String? {
            val fileLength = file.length()
            if (fileLength < partSize * 3) {
                return getFileSHA256(file)
            }
            val digest = MessageDigest.getInstance("SHA-256")
            val fis = FileInputStream(file)
            val buffer = ByteArray(8192)

            // read start
            var bytesRead: kotlin.Long = 0
            while (bytesRead < partSize) {
                val length =
                    fis.read(buffer, 0, min(buffer.size.toLong(), partSize - bytesRead).toInt())
                if (length == -1) break
                digest.update(buffer, 0, length)
                bytesRead += length.toLong()
            }

            // read middle
            val middleStart = (fileLength - partSize) / 2
            fis.getChannel().position(middleStart)
            bytesRead = 0
            while (bytesRead < partSize) {
                val length =
                    fis.read(buffer, 0, min(buffer.size.toLong(), partSize - bytesRead).toInt())
                if (length == -1) break
                digest.update(buffer, 0, length)
                bytesRead += length.toLong()
            }

            // read end
            fis.getChannel().position(fileLength - partSize)
            bytesRead = 0
            while (bytesRead < partSize) {
                val length =
                    fis.read(buffer, 0, min(buffer.size.toLong(), partSize - bytesRead).toInt())
                if (length == -1) break
                digest.update(buffer, 0, length)
                bytesRead += length.toLong()
            }

            fis.close()
            return bytesToHex(digest.digest())
        }

        fun getPartialFileHash(file: File, sampleCount: Int): String? {
            try {
                val digest = MessageDigest.getInstance("SHA-256")
                val raf = RandomAccessFile(file, "r")
                val fileSize = raf.length()

                for (i in 0..<sampleCount) {
                    val position = (fileSize * i) / sampleCount
                    raf.seek(position)

                    val buffer = ByteArray(4096)
                    val bytesRead = raf.read(buffer)
                    if (bytesRead > 0) {
                        digest.update(buffer, 0, bytesRead)
                    }
                }

                raf.seek(0)
                val header = ByteArray(1024)
                val headerRead = raf.read(header)
                if (headerRead > 0) digest.update(header, 0, headerRead)

                raf.seek(max(0, fileSize - 1024))
                val footer = ByteArray(1024)
                val footerRead = raf.read(footer)
                if (footerRead > 0) digest.update(footer, 0, footerRead)

                raf.close()
                return bytesToHex(digest.digest())
            } catch (e: java.lang.Exception) {
                e.printStackTrace()
                return null
            }
        }
    }
}