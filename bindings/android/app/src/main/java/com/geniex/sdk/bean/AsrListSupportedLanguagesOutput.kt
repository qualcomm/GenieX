package com.geniex.sdk.bean

/**
 * Output bean for ASR listing supported languages.
 * @property languageCodes List of supported language codes (e.g., "en-US", "zh-CN").
 */
data class AsrListSupportedLanguagesOutput(
    val languageCodes: List<String>,
    val code: Int
)

