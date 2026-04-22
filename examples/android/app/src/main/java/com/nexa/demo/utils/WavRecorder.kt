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

import android.Manifest
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.Visualizer
import android.media.audiofx.Visualizer.OnDataCaptureListener
import android.util.Log
import androidx.annotation.RequiresPermission
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import kotlin.math.log10


class WavRecorder(
    private val sampleRate: Int = 16000,
    private val channelConfig: Int = AudioFormat.CHANNEL_IN_MONO,
    private val audioFormat: Int = AudioFormat.ENCODING_PCM_16BIT,
    private val maxDurationMs: Long = MAX_DURATION_MINUTE * 1000L,
    private val onVolumeDbChangeListener: OnVolumeDbChangeListener = object :
        OnVolumeDbChangeListener {
        override fun onVolumeDbChange(volumeDb: Double) {
        }
    },
    private val onMaxDurationReached: (() -> Unit)? = null
) {
    private var recorder: AudioRecord? = null
    @Volatile
    private var isRecording = false
    private var recordingThread: Thread? = null
    private var visualizer: Visualizer? = null

    companion object {
        const val MAX_DURATION_MINUTE = 20
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun startRecording(outputFile: File) {
        val minBuffer = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
        recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            minBuffer
        )

        recorder?.startRecording()
        isRecording = true

        recordingThread = Thread {
            writePcmToWav(outputFile, minBuffer)
        }.apply { start() }
    }

    fun stopRecording() {
        isRecording = false
        try {
            recorder?.stop()
            recorder?.release()
            visualizer?.enabled = false
            visualizer?.release()
        } catch (_: Exception) {
        }
        recorder = null
        recordingThread = null
        visualizer = null
    }

    private fun writePcmToWav(outputFile: File, bufferSize: Int) {
        val data = ByteArray(bufferSize)
        var totalAudioLen: Long = 0
        val startTime = System.currentTimeMillis()
        FileOutputStream(outputFile).use { fos ->
            writeWavHeader(fos, sampleRate, 1, 16, 0)
            var nowTime = System.currentTimeMillis()
            while (isRecording) {
                val currentTime = System.currentTimeMillis()
                val elapsedTime = currentTime - startTime
                
                // Check if maximum duration is reached
                if (elapsedTime >= maxDurationMs) {
                    onMaxDurationReached?.invoke()
                    break
                }
                
                val read = recorder?.read(data, 0, data.size) ?: 0
                if (read > 0) {
                    fos.write(data, 0, read)
                    totalAudioLen += read
                    // Process audio decibels
                    // Calculate sum of squares
                    val lastTime = System.currentTimeMillis()
                    if (lastTime - nowTime > 60) {
                        var sum: Long = 0
                        for (i in 0..<read) {
                            sum += data[i] * data[i]
                        }
                        // Calculate average and convert to decibels
                        val mean = sum / read.toDouble()
                        val volumeDb = 10 * log10(mean)
                        onVolumeDbChangeListener.onVolumeDbChange(volumeDb)
                        nowTime = lastTime
                    }
                }
            }
            // Fill back header
            updateWavHeader(outputFile, totalAudioLen, sampleRate, 1, 16)
        }
    }

    private fun writeWavHeader(
        out: FileOutputStream, sampleRate: Int, channels: Int,
        bitsPerSample: Int, dataSize: Long
    ) {
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val totalDataLen = 36 + dataSize
        val header = ByteArray(44)

        fun putLE(value: Long, offset: Int) {
            header[offset] = (value and 0xff).toByte()
            header[offset + 1] = ((value shr 8) and 0xff).toByte()
            header[offset + 2] = ((value shr 16) and 0xff).toByte()
            header[offset + 3] = ((value shr 24) and 0xff).toByte()
        }

        // ChunkID "RIFF"
        header[0] = 'R'.code.toByte(); header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte(); header[3] = 'F'.code.toByte()
        putLE(totalDataLen, 4)

        // Format "WAVE"
        header[8] = 'W'.code.toByte(); header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte(); header[11] = 'E'.code.toByte()

        // Subchunk1 "fmt "
        header[12] = 'f'.code.toByte(); header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte(); header[15] = ' '.code.toByte()
        putLE(16, 16) // Subchunk1 size
        header[20] = 1; header[21] = 0 // PCM
        header[22] = channels.toByte(); header[23] = 0
        putLE(sampleRate.toLong(), 24)
        putLE(byteRate.toLong(), 28)
        header[32] = (channels * bitsPerSample / 8).toByte(); header[33] = 0
        header[34] = bitsPerSample.toByte(); header[35] = 0

        // Subchunk2 "data"
        header[36] = 'd'.code.toByte(); header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte(); header[39] = 'a'.code.toByte()
        putLE(dataSize, 40)

        out.write(header, 0, 44)
    }

    private fun updateWavHeader(
        wavFile: File, dataSize: Long,
        sampleRate: Int, channels: Int, bitsPerSample: Int
    ) {
        val totalDataLen = 36 + dataSize
        val byteRate = sampleRate * channels * bitsPerSample / 8

        RandomAccessFile(wavFile, "rw").use { raf ->
            fun putLE(offset: Long, value: Long) {
                raf.seek(offset)
                raf.write(
                    byteArrayOf(
                        (value and 0xff).toByte(),
                        ((value shr 8) and 0xff).toByte(),
                        ((value shr 16) and 0xff).toByte(),
                        ((value shr 24) and 0xff).toByte()
                    )
                )
            }
            putLE(4, totalDataLen)
            putLE(40, dataSize)
        }
    }

    @Deprecated("Has issues, temporarily using decibel solution")
    private fun initVisualizer(audioSessionId: Int) {
        val captureSize = Visualizer.getCaptureSizeRange()[1]
        visualizer = Visualizer(audioSessionId)
        visualizer!!.captureSize = captureSize
        visualizer!!.setDataCaptureListener(
            object : OnDataCaptureListener {
                override fun onWaveFormDataCapture(
                    visualizer: Visualizer?,
                    waveform: ByteArray?,
                    samplingRate: Int
                ) {
                    // Process waveform data
                    waveform?.let {
                        val size = it.size.coerceAtMost(100)
                        for (i in 0 until size) {
                            Log.d("nfl", "$i : ${it[i]}")
                        }
                    }
                }

                override fun onFftDataCapture(
                    visualizer: Visualizer?,
                    fft: ByteArray?,
                    samplingRate: Int
                ) {
                    // Process FFT data
                }
            },
            Visualizer.getMaxCaptureRate() / 2,
            true,
            true
        )
        visualizer!!.enabled = true
    }

    interface OnVolumeDbChangeListener {
        /**
         * Current decibel value, typically 30~90
         */
        fun onVolumeDbChange(volumeDb: Double)
    }
}