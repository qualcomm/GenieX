// SPDX-License-Identifier: BSD-3-Clause
package com.geniex.sdk.jni

/** Raw additive TTS FFI. Initialize the SDK first; destroy before SDK deinit.
 * Select the shared QAIRT runtime before SDK init; runtimePath only asserts it.
 * Synthesis is synchronous. Callback receives copied mono PCM, and may return
 * false or call cancel. Never destroy a handle during synthesis.
 * Android hardware support requires an Android-compiled bundle and text library.
 */
internal class Tts {
    fun interface AudioCallback {
        fun onAudio(pcm: FloatArray, sampleRate: Int): Boolean
    }

    data class Result(
        val status: Int, val sampleRate: Int, val samples: Long,
        val firstAudioMs: Double, val generationMs: Double,
        val prefillNpuMs: Double, val stepNpuMs: Double,
        val prefillCalls: Int, val stepCalls: Int
    )

    external fun create(
        pluginId: String, modelPath: String, device: String,
        runtimePath: String, textLibrary: String, language: String,
        socModel: String, verifyNpu: Boolean
    ): Long
    external fun synthesize(handle: Long, text: String, voice: String, callback: AudioCallback): Result
    external fun cancel(handle: Long): Int
    external fun destroy(handle: Long): Int
}
