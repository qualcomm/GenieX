package com.geniex.demo.bean


import java.io.File

data class DownloadableFile(
    val file: File,
    val url: String
)