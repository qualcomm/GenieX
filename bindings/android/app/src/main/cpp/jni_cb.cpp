#include "jni_cb.h"

#include <cstdint>
#include <vector>

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

// Decode standard UTF-8 to UTF-16 and build a jstring via NewString.
// The SDK already buffers partial multi-byte sequences across token
// boundaries, so the input is valid standard UTF-8 (including 4-byte emoji).
// NewStringUTF expects JNI modified UTF-8 and mangles supplementary
// characters, so we decode to UTF-16 (surrogate pairs) ourselves.
static jstring utf8_to_jstring(JNIEnv* env, const char* str) {
    std::vector<jchar>   u16;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(str ? str : "");

    while (*p) {
        unsigned char c = *p;
        uint32_t      cp;
        int           n;
        if (c < 0x80) {
            cp = c;
            n  = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            n  = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            n  = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07;
            n  = 4;
        } else {
            p++;  // invalid leading byte, skip
            continue;
        }

        for (int i = 1; i < n; i++) {
            if ((p[i] & 0xC0) != 0x80) {  // truncated sequence
                cp = 0xFFFD;
                n  = i;
                break;
            }
            cp = (cp << 6) | (p[i] & 0x3F);
        }
        p += n;

        if (cp <= 0xFFFF) {
            u16.push_back(static_cast<jchar>(cp));
        } else {  // supplementary plane -> surrogate pair
            cp -= 0x10000;
            u16.push_back(static_cast<jchar>(0xD800 + (cp >> 10)));
            u16.push_back(static_cast<jchar>(0xDC00 + (cp & 0x3FF)));
        }
    }

    return env->NewString(u16.data(), static_cast<jsize>(u16.size()));
}

bool jni_cb_emit_token(JavaCallbackCtx* ctx, const char* token_utf8) {
    if (!ctx || !ctx->cb_global || !ctx->onToken_mid) return false;
    if (ctx->stop_flag && ctx->stop_flag->load()) return false;

    bool    attached = false;
    JNIEnv* env      = get_env(ctx->vm, attached);
    if (!env) return false;

    jstring jtoken = utf8_to_jstring(env, token_utf8);

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
