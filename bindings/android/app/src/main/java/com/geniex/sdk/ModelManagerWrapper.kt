package com.geniex.sdk

import com.geniex.sdk.bean.FileProgress
import com.geniex.sdk.bean.ModelPaths
import com.geniex.sdk.bean.ModelPullInput
import com.geniex.sdk.bean.ModelType
import com.geniex.sdk.callback.DownloadProgressCallback
import com.geniex.sdk.jni.ModelManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

/**
 * High-level Kotlin API over the `geniex_model_*` C FFI. Single source
 * of truth for download / list / resolve-paths; UI code should not
 * reach into [ModelManager] directly.
 */
object ModelManagerWrapper {
    private val native = ModelManager()

    @Volatile
    private var initialized = false
    private val initLock = Any()

    /**
     * Idempotent. Calling after a successful init returns [Result.success]
     * without touching the FFI. Native `ALREADY_INITIALIZED` (-100008) is
     * also treated as success — the FFI's `OnceLock` never resets, and
     * `deinit` is a no-op, so the only sane contract is "first successful
     * init wins, further calls are benign".
     */
    suspend fun init(dataDir: String): Result<Unit> = withContext(Dispatchers.IO) {
        synchronized(initLock) {
            if (initialized) return@withContext Result.success(Unit)
            val rc = native.init(dataDir)
            if (rc == GENIEX_SUCCESS || rc == GENIEX_ERROR_ALREADY_INITIALIZED) {
                initialized = true
                Result.success(Unit)
            } else {
                Result.failure(GenieXModelError(rc, "geniex_model_init failed"))
            }
        }
    }

    /**
     * Pulls a model in a background coroutine. Progress updates stream on
     * the collecting scope. Cancelling the flow (or returning `false`
     * from the progress callback) requests cancellation; the Rust side
     * unwinds and returns `CANCELLED`, leaving partial files on disk for
     * a later resumed pull.
     *
     * Terminates with one of [PullEvent.Completed] or [PullEvent.Error].
     */
    fun pullFlow(input: ModelPullInput): Flow<PullEvent> = callbackFlow {
        val cancelled = AtomicBoolean(false)
        val cb = object : DownloadProgressCallback {
            override fun onProgress(files: Array<FileProgress>): Boolean {
                if (cancelled.get() || isClosedForSend) return false
                trySend(PullEvent.Progress(files.toList()))
                return true
            }
        }
        val rc = try {
            native.pull(input, cb)
        } catch (t: Throwable) {
            trySend(PullEvent.Error(-1, t.message ?: t.toString()))
            close()
            return@callbackFlow
        }

        if (rc == GENIEX_SUCCESS) {
            trySend(PullEvent.Completed)
        } else if (rc == GENIEX_ERROR_CANCELLED) {
            // Flow consumer cancelled. Don't emit an Error.
        } else {
            trySend(PullEvent.Error(rc, "geniex_model_pull failed (rc=$rc)"))
        }
        close()

        awaitClose {
            cancelled.set(true)
        }
    }.flowOn(Dispatchers.IO)

    suspend fun list(): List<String> = withContext(Dispatchers.IO) {
        native.list().toList()
    }

    suspend fun getPaths(modelName: String): ModelPaths? = withContext(Dispatchers.IO) {
        native.getPaths(modelName)
    }

    suspend fun remove(modelName: String): Int = withContext(Dispatchers.IO) {
        native.remove(modelName)
    }

    suspend fun clean(): Int = withContext(Dispatchers.IO) {
        native.clean()
    }

    suspend fun getType(modelName: String): ModelType? = withContext(Dispatchers.IO) {
        val v = native.getType(modelName)
        if (v < 0) null else ModelType.fromValue(v)
    }

    suspend fun resolveAlias(alias: String): String? = withContext(Dispatchers.IO) {
        native.resolveAlias(alias)
    }

    sealed class PullEvent {
        data class Progress(val files: List<FileProgress>) : PullEvent()
        object Completed : PullEvent()
        data class Error(val code: Int, val message: String) : PullEvent()
    }

    class GenieXModelError(val code: Int, message: String) : RuntimeException("$message (rc=$code)")

    private const val GENIEX_SUCCESS = 0
    // From sdk/model-manager/crates/ffi/src/types.rs — keep in sync.
    private const val GENIEX_ERROR_CANCELLED = -100006
    private const val GENIEX_ERROR_ALREADY_INITIALIZED = -100008
}
