package com.geniex.sdk.bean

/** Mirrors `geniex_ModelType` from geniex_model.h. */
enum class ModelType(val value: Int) {
    LLM(0),
    VLM(1);

    companion object {
        // @JvmStatic exposes this as a real static method on `ModelType`,
        // matching what model_manager_jni.cpp looks up via GetStaticMethodID.
        // Without it, Kotlin only emits `ModelType.Companion.fromValue(...)`.
        @JvmStatic
        fun fromValue(v: Int): ModelType? = values().firstOrNull { it.value == v }
    }
}
