package com.geniex.demo.utils

class SharePreferenceKeys {
    companion object {
        const val KEY_LAST_LOAD_MODEL_ID = "last_load_model_id"
        /**
         * When navigation pops, since it will load the initial modelId, cannot directly load model,
         * should let MainChatScreen load it
         */
        const val KEY_PREPARE_LOAD_MODEL_ID = "prepare_load_model_id"
        const val KEY_SHOW_OPERATION_POPUP = "show_operation_popup"
    }
    enum class FileName(val fileName: String) {
        ModelDownloadState("mode_download_state"),
        CommonConfig("common_config")
    }
}