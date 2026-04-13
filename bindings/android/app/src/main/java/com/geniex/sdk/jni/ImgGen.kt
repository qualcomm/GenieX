package com.geniex.sdk.jni

import com.geniex.sdk.bean.ImageGenCreateInput
import com.geniex.sdk.bean.ImageGenImg2ImgInput
import com.geniex.sdk.bean.ImageGenTxt2ImgInput

internal class ImgGen {
    external fun create(imageGenCreateInput: ImageGenCreateInput): Long
    external fun txt2Img(imageGenTxt2ImgInput: ImageGenTxt2ImgInput, handle: Long): Int
    external fun img2Img(imageGenImg2ImgInput: ImageGenImg2ImgInput, handle: Long): Int
    external fun destroy(handle: Long): Int
}