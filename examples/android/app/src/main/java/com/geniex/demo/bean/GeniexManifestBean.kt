package com.geniex.demo.bean

import android.annotation.SuppressLint
import kotlinx.serialization.Serializable

@SuppressLint("UnsafeOptInUsageError")
@Serializable
data class GeniexManifestBean(
    val ModelName: String? = null,
    val ModelType: String? = null,
    val PluginId: String? = null
)