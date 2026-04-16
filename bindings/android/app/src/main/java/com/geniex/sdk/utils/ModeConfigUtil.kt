package com.geniex.sdk.utils

import android.system.Os
import android.text.TextUtils
import com.geniex.sdk.GeniexSdk.Companion.KEY_NPU_LIB_FOLDER_PATH
import com.geniex.sdk.GeniexSdk.Companion.PLUGIN_ID_NPU
import com.geniex.sdk.bean.CVModelConfig
import com.geniex.sdk.bean.ModelConfig
import java.io.File

class ModeConfigUtil {
    companion object {
        fun getNpuModelFolderPath(
            pluginId: String?,
            modelPath: String,
            modelConfig: ModelConfig
        ): String? {
            return if (PLUGIN_ID_NPU == pluginId) {
                if (TextUtils.isEmpty(modelConfig.npu_model_folder_path)) {
                    if (TextUtils.isEmpty(modelPath)) {
                        throw IllegalArgumentException("modelPath required")
                    } else {
                        File(modelPath).parentFile!!.absolutePath
                    }
                } else {
                    modelConfig.npu_model_folder_path
                }
            } else {
                modelConfig.npu_model_folder_path
            }
        }

        fun getNpuLibFolderPath(modelConfig: ModelConfig): String? {
            return if (TextUtils.isEmpty(modelConfig.npu_lib_folder_path)) {
                Os.getenv(KEY_NPU_LIB_FOLDER_PATH)
            } else {
                modelConfig.npu_lib_folder_path
            }
        }

        fun getNpuModelFolderPath(
            pluginId: String?,
            modelConfig: CVModelConfig
        ): String? {
            return if (PLUGIN_ID_NPU == pluginId) {
                val modelPath = modelConfig.rec_model_path
                if (TextUtils.isEmpty(modelConfig.npu_model_folder_path)) {
                    if (TextUtils.isEmpty(modelPath)) {
                        throw IllegalArgumentException("modelPath required")
                    } else {
                        File(modelPath).parentFile!!.absolutePath
                    }
                } else {
                    modelConfig.npu_model_folder_path
                }
            } else {
                modelConfig.npu_model_folder_path
            }
        }

        fun getNpuLibFolderPath(modelConfig: CVModelConfig): String? {
            return if (TextUtils.isEmpty(modelConfig.npu_lib_folder_path)) {
                Os.getenv(KEY_NPU_LIB_FOLDER_PATH)
            } else {
                modelConfig.npu_lib_folder_path
            }
        }

    }
}