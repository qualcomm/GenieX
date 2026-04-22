package com.geniex.demo.bean

import android.annotation.SuppressLint
import kotlinx.serialization.Serializable

@SuppressLint("UnsafeOptInUsageError")
@Serializable
data class DownloadFileConfig (
    val name:String,
    /**
     * Path relative to model dir
     */
    val path:String? = "",
    val url:String? = ""
)