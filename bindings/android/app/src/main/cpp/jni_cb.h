#pragma once
#include <jni.h>

#include <atomic>

struct JavaCallbackCtx {
    JavaVM*            vm             = nullptr;  // for cross-thread JNIEnv*
    jobject            cb_global      = nullptr;  // GlobalRef to callback object
    jmethodID          onToken_mid    = nullptr;  // (Ljava/lang/String;)Z
    jmethodID          onComplete_mid = nullptr;  // (Lcom/.../LlmGenerateResult;)V (可選)
    std::atomic<bool>* stop_flag      = nullptr;  // optional external stop flag
};

bool jni_cb_init(JNIEnv* env, jobject callback, const char* onTokenName, const char* onTokenSig,
    const char* onCompleteName, const char* onCompleteSig, std::atomic<bool>* stopFlag, JavaCallbackCtx* out);

bool jni_cb_emit_token(JavaCallbackCtx* ctx, const char* token_utf8);

void jni_cb_call_complete(JavaCallbackCtx* ctx, jobject result);

void jni_cb_dispose(JNIEnv* env, JavaCallbackCtx* ctx);