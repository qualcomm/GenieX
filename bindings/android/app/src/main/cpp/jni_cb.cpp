#include "jni_cb.h"

#include "android_utils.h"

static JNIEnv* get_env(JavaVM* vm, bool& attached) {
    attached    = false;
    JNIEnv* env = nullptr;
    if (!vm) return nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK || !env) {
        if (vm->AttachCurrentThread(&env, nullptr) != 0) return nullptr;
        attached = true;
    }
    return env;
}

bool jni_cb_init(JNIEnv* env, jobject callback, const char* onTokenName, const char* onTokenSig,
    const char* onCompleteName, const char* onCompleteSig, std::atomic<bool>* stopFlag, JavaCallbackCtx* out) {
    if (!env || !callback || !onTokenName || !onTokenSig || !out) return false;

    JavaVM* vm = nullptr;
    if (env->GetJavaVM(&vm) != 0 || !vm) return false;

    jclass cb_cls = env->GetObjectClass(callback);
    if (!cb_cls) return false;

    jmethodID onToken = env->GetMethodID(cb_cls, onTokenName, onTokenSig);
    if (!onToken) return false;

    jmethodID onComplete = nullptr;
    if (onCompleteName && onCompleteSig) {
        onComplete = env->GetMethodID(cb_cls, onCompleteName, onCompleteSig);
    }

    jobject cb_global = env->NewGlobalRef(callback);
    if (!cb_global) return false;

    out->vm             = vm;
    out->cb_global      = cb_global;
    out->onToken_mid    = onToken;
    out->onComplete_mid = onComplete;
    out->stop_flag      = stopFlag;
    return true;
}

bool isValidModifiedUTF8(const char* str) {
    if (str == NULL) {
        return false;
    }
    const unsigned char* p = (const unsigned char*)str;

    while (*p != 0) {
        if (*p == 0xC0 && *(p + 1) == 0x80) {
            p += 2;
            continue;
        }

        if ((*p & 0x80) == 0x00) {
            if (*p == 0x00) {
                return false;
            }
            p++;
            continue;
        }

        if ((*p & 0xE0) == 0xC0) {
            if (*(p + 1) == 0) return false;
            if ((*(p + 1) & 0xC0) != 0x80) return false;
            unsigned int codepoint = ((*p & 0x1F) << 6) | (*(p + 1) & 0x3F);
            if (codepoint < 0x0080 || codepoint > 0x07FF) {
                return false;
            }

            p += 2;
            continue;
        }

        if ((*p & 0xF0) == 0xE0) {
            if (*(p + 1) == 0 || *(p + 2) == 0) return false;
            if ((*(p + 1) & 0xC0) != 0x80 || (*(p + 2) & 0xC0) != 0x80) {
                return false;
            }

            unsigned int codepoint = ((*p & 0x0F) << 12) | ((*(p + 1) & 0x3F) << 6) | (*(p + 2) & 0x3F);

            if (codepoint >= 0xD800 && codepoint <= 0xDFFF) {
                if (codepoint >= 0xDC00) {
                    return false;
                }

                if ((*(p + 3) & 0xF0) != 0xE0) {
                    return false;
                }

                unsigned int low_surrogate = ((*(p + 3) & 0x0F) << 12) | ((*(p + 4) & 0x3F) << 6) | (*(p + 5) & 0x3F);

                if (low_surrogate < 0xDC00 || low_surrogate > 0xDFFF) {
                    return false;
                }

                p += 6;
            } else {
                if (codepoint < 0x0800 || codepoint > 0xFFFF) {
                    return false;
                }
                p += 3;
            }
            continue;
        }

        if ((*p & 0xF8) == 0xF0) {
            return false;
        }
        return false;
    }
    return true;
}

bool jni_cb_emit_token(JavaCallbackCtx* ctx, const char* token_utf8) {
    if (!ctx || !ctx->cb_global || !ctx->onToken_mid) return false;
    if (ctx->stop_flag && ctx->stop_flag->load()) return false;

    bool    attached = false;
    JNIEnv* env      = get_env(ctx->vm, attached);
    if (!env) return false;

    jstring jtoken;
    if (isValidModifiedUTF8(token_utf8)) {
        jtoken = env->NewStringUTF(token_utf8 ? token_utf8 : "");
    } else {
        jtoken = env->NewStringUTF("");
    }

    jboolean cont = env->CallBooleanMethod(ctx->cb_global, ctx->onToken_mid, jtoken);

    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
        cont = JNI_FALSE;
    }

    env->DeleteLocalRef(jtoken);
    if (attached) ctx->vm->DetachCurrentThread();
    return cont == JNI_TRUE;
}

void jni_cb_call_complete(JavaCallbackCtx* ctx, jobject result) {
    if (!ctx || !ctx->cb_global || !ctx->onComplete_mid) return;

    bool    attached = false;
    JNIEnv* env      = get_env(ctx->vm, attached);
    if (!env) return;

    env->CallVoidMethod(ctx->cb_global, ctx->onComplete_mid, result);
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    if (attached) ctx->vm->DetachCurrentThread();
}

void jni_cb_dispose(JNIEnv* env, JavaCallbackCtx* ctx) {
    if (!env || !ctx) return;
    if (ctx->cb_global) {
        env->DeleteGlobalRef(ctx->cb_global);
        ctx->cb_global = nullptr;
    }
    ctx->vm             = nullptr;
    ctx->onToken_mid    = nullptr;
    ctx->onComplete_mid = nullptr;
}
