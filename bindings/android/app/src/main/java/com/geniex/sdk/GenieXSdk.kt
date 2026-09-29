package com.geniex.sdk

import android.content.Context
import android.util.Log
import com.geniex.sdk.bean.RuntimeIdValue
import com.geniex.sdk.jni.ModelManager
import java.io.File


class GenieXSdk private constructor() {

    interface InitCallback {
        fun onSuccess()
        fun onFailure(reason: String)
    }

    external fun registerPlugin(pluginLibPath: String): Int

    external fun getPluginVersion(pluginId: String): String

    /**
     * The GenieX SDK bridge version (build_config::kBridgeVersion) -- the same
     * string [getPluginVersion] callers compare against when caching plugin
     * versions, exposed here for the SDK itself.
     */
    external fun getSdkVersion(): String

    /**
     * Load the QAIRT runtime from [path] instead of the one bundled in the APK, for
     * running against another QAIRT version. [path] is either a QAIRT SDK root or a
     * flat folder of QNN libraries, pushed somewhere the app can read; "" restores the
     * bundled runtime. Ignored by other plugins.
     *
     * Call before [init]: the QNN libraries load once per process and are never
     * unloaded, so this fails afterwards. An unusable path is reported when the model
     * is created, not here.
     */
    external fun setQairtRuntimePath(path: String): Int

    external fun getQairtRuntimePath(): String

    // Idempotent across Activity recreation. Plugin registration is
    // safe to re-attempt (it logs and moves on); model-manager init is
    // not — the FFI rejects re-init — so we guard it here.
    @Volatile
    private var pluginsRegistered = false

    @Volatile
    private var modelManagerInited = false

    /**
     * @param callback Use for Checking the context environment of GenieXSdk.When an exception occurs,
     * the [InitCallback.onFailure] will be invoked.
     */
    fun init(context: Context, callback: InitCallback? = null) {
        val nativeLibPath = context.applicationInfo.nativeLibraryDir

        val exceptionResult = StringBuilder()
        synchronized(this) {
            if (!pluginsRegistered) {
                arrayOf(
                    RuntimeIdValue.LLAMA_CPP.value,
                    RuntimeIdValue.QAIRT.value
                ).forEach { pluginName ->
                    File(
                        nativeLibPath,
                        "libgeniex_plugin_${pluginName}.so"
                    ).let { pluginSoFile ->
                        if (pluginSoFile.exists()) {
                            pluginSoFile.absolutePath.let { pluginPath ->
                                Log.d(TAG, "Loading plugin: $pluginPath")
                                if (registerPlugin(pluginPath) != 0) {
                                    exceptionResult.append("Cannot registerPlugin $pluginName\n")
                                }
                            }
                        } else {
                            exceptionResult.append("Cannot find ${pluginSoFile.name} in $nativeLibPath\n")
                        }
                    }
                }
                pluginsRegistered = true
            }

            if (!modelManagerInited) {
                val dataDir = File(context.filesDir, "geniex").apply { mkdirs() }
                val rc = ModelManager().init(dataDir.absolutePath)
                if (rc == 0 || rc == GENIEX_ERROR_ALREADY_INITIALIZED) {
                    modelManagerInited = true
                } else {
                    exceptionResult.append("geniex_model_init failed (rc=$rc)\n")
                }
            }
        }

        if (exceptionResult.isEmpty()) {
            callback?.onSuccess()
        } else {
            callback?.onFailure(exceptionResult.toString())
        }
    }

    companion object {
        private const val TAG = "GenieXSdk"
        const val PLUGIN_ID_QAIRT = "qairt"
        const val PLUGIN_ID_LLAMA_CPP = "llama_cpp"
        // Mirror of GENIEX_ERROR_COMMON_ALREADY_INITIALIZED from sdk/model-manager/crates/ffi/src/types.rs.
        private const val GENIEX_ERROR_ALREADY_INITIALIZED = -100008

        init {
            System.loadLibrary("npu_jni")
        }

        @Volatile
        private var instance: GenieXSdk? = null

        fun getInstance() = instance ?: synchronized(this) {
            instance ?: GenieXSdk().also { instance = it }
        }
    }
}
