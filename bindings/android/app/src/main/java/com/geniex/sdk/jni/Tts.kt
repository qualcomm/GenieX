package com.geniex.sdk.jni

import com.geniex.sdk.bean.TtsCreateInput
import com.geniex.sdk.bean.TtsSynthesizeInput
import com.geniex.sdk.bean.TtsSynthesizeOutput

internal class Tts {
    external fun create(
        ttsCreateInput: TtsCreateInput
    ): Long

    external fun destroy(handle: Long): Int

    external fun synthesize(
        handle: Long,
        input: TtsSynthesizeInput
    ): TtsSynthesizeOutput

    external fun listAvailableVoices(
        handle: Long
    ): List<String>?
}
