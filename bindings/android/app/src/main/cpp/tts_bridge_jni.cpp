// SPDX-License-Identifier: BSD-3-Clause
#include <jni.h>

#include <string>

#include "geniex_tts.h"
#include "jniutils.h"

namespace {
struct AudioContext {
    JNIEnv*   env;
    jobject   callback;
    jmethodID method;
};

bool on_audio(const float* pcm, uint32_t count, uint32_t rate, void* data) {
    auto* ctx = static_cast<AudioContext*>(data);
    if (ctx->env->ExceptionCheck()) return false;
    auto samples = ctx->env->NewFloatArray(count);
    if (!samples) return false;
    ctx->env->SetFloatArrayRegion(samples, 0, count, pcm);
    bool more = false;
    if (!ctx->env->ExceptionCheck()) more = ctx->env->CallBooleanMethod(ctx->callback, ctx->method, samples, rate);
    ctx->env->DeleteLocalRef(samples);
    return more && !ctx->env->ExceptionCheck();
}
}  // namespace

extern "C" JNIEXPORT jlong JNICALL Java_com_geniex_sdk_jni_Tts_create(JNIEnv* env, jobject, jstring plugin,
    jstring model, jstring device, jstring runtime, jstring text_lib, jstring language, jstring soc, jboolean verify) {
    for (auto value : {plugin, model, device, runtime, text_lib, language, soc}) {
        if (!value) {
            env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "TTS create strings cannot be null");
            return 0;
        }
    }
    auto                  p = jniutils::jstring2str(env, plugin), m = jniutils::jstring2str(env, model);
    auto                  d = jniutils::jstring2str(env, device), r = jniutils::jstring2str(env, runtime);
    auto                  t = jniutils::jstring2str(env, text_lib), l = jniutils::jstring2str(env, language);
    auto                  s = jniutils::jstring2str(env, soc);
    geniex_TtsCreateInput input{sizeof(input),
        verify ? GENIEX_TTS_VERIFY_NPU : 0u,
        p.c_str(),
        m.c_str(),
        d.c_str(),
        r.c_str(),
        t.c_str(),
        l.c_str(),
        s.c_str()};
    geniex_TTS*           handle = nullptr;
    auto                  status = geniex_tts_create(&input, &handle);
    if (status != GENIEX_SUCCESS) {
        auto error = std::string("TTS create failed: ") + std::to_string(status);
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.c_str());
        return 0;
    }
    return reinterpret_cast<jlong>(handle);
}

extern "C" JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Tts_synthesize(
    JNIEnv* env, jobject, jlong handle, jstring text, jstring voice, jobject callback) {
    if (!text || !voice || !callback) {
        env->ThrowNew(
            env->FindClass("java/lang/IllegalArgumentException"), "TTS text, voice and callback cannot be null");
        return nullptr;
    }
    auto cb_class = env->GetObjectClass(callback);
    auto method   = env->GetMethodID(cb_class, "onAudio", "([FI)Z");
    env->DeleteLocalRef(cb_class);
    if (!method) return nullptr;
    auto                       t = jniutils::jstring2str(env, text), v = jniutils::jstring2str(env, voice);
    AudioContext               context{env, callback, method};
    geniex_TtsSynthesizeInput  input{sizeof(input), t.c_str(), v.c_str(), on_audio, &context};
    geniex_TtsSynthesizeOutput out{sizeof(out)};
    auto                       status = geniex_tts_synthesize(reinterpret_cast<geniex_TTS*>(handle), &input, &out);
    if (env->ExceptionCheck()) return nullptr;  // Preserve the callback's exception.
    auto cls = env->FindClass("com/geniex/sdk/jni/Tts$Result");
    if (!cls) return nullptr;
    auto ctor = env->GetMethodID(cls, "<init>", "(IIJDDDDII)V");
    if (!ctor) return nullptr;
    return env->NewObject(cls,
        ctor,
        status,
        static_cast<jint>(out.sample_rate),
        static_cast<jlong>(out.samples),
        out.first_audio_ms,
        out.generation_ms,
        out.prefill_npu_ms,
        out.step_npu_ms,
        static_cast<jint>(out.prefill_calls),
        static_cast<jint>(out.step_calls));
}

extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_Tts_cancel(JNIEnv*, jobject, jlong handle) {
    return geniex_tts_cancel(reinterpret_cast<geniex_TTS*>(handle));
}

extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_Tts_destroy(JNIEnv*, jobject, jlong handle) {
    return geniex_tts_destroy(reinterpret_cast<geniex_TTS*>(handle));
}
