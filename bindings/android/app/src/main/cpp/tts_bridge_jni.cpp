#include <android/log.h>
#include <jni.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "android_utils.h"
#include "geniex.h"
#include "jniutils.h"

using namespace geniex_android_sdk;

extern "C" {

geniex_TtsCreateInput extract_tts_create_input(JNIEnv* env, jobject inputObj) {
    geniex_TtsCreateInput out = {};

    if (!inputObj) {
        LOGe("extract_tts_create_input: inputObj is null");
        return out;
    }

    jclass cls = env->GetObjectClass(inputObj);
    if (!cls) {
        LOGe("[JNI] extract_tts_create_input: GetObjectClass returned null");
        return out;
    }

    out.model_name   = getStringField(env, cls, inputObj, "model_name");
    out.model_path   = getStringField(env, cls, inputObj, "model_path");
    out.vocoder_path = getStringField(env, cls, inputObj, "vocoder_path");

    jobject configObj = getObjectField(env, cls, inputObj, "config", "Lcom/geniex/sdk/bean/ModelConfig;");
    if (!configObj) {
        out.config = {};
    } else {
        out.config = jniutils::extract_model_config(env, configObj);
    }

    out.plugin_id = getStringField(env, cls, inputObj, "plugin_id");
    // Translate user-friendly device_id to internal device string
    const char* raw_device_id = getStringField(env, cls, inputObj, "device_id");
    if (raw_device_id) {
        std::string translated = jniutils::translate_device_id(raw_device_id);
        out.device_id          = jniutils::hold_c_str(translated);
        LOGd("device_id translated: %s -> %s", raw_device_id, translated.c_str());
    } else {
        out.device_id = nullptr;
    }

    env->DeleteLocalRef(cls);
    return out;
}

geniex_TTSConfig extract_tts_config(JNIEnv* env, jobject inputObj) {
    geniex_TTSConfig out = {};

    if (!inputObj) {
        LOGe("extract_tts_config: inputObj is null");
        return out;
    }

    jclass cls = env->GetObjectClass(inputObj);
    if (!cls) {
        LOGe("extract_tts_config: GetObjectClass returned null");
        return out;
    }

    out.voice       = getStringField(env, cls, inputObj, "voice");
    out.speed       = getFloatField(env, cls, inputObj, "speed");
    out.seed        = getIntField(env, cls, inputObj, "seed");
    out.sample_rate = getIntField(env, cls, inputObj, "sampleRate");

    env->DeleteLocalRef(cls);
    return out;
}

geniex_TtsSynthesizeInput extract_tts_synthesize_input(
    JNIEnv* env, jclass cls, jobject inputObj, geniex_TTSConfig* config) {
    geniex_TtsSynthesizeInput out = {};

    out.text_utf8   = getStringField(env, cls, inputObj, "textUtf8");
    out.output_path = getStringField(env, cls, inputObj, "outputPath");

    jobject configObj = getObjectField(env, cls, inputObj, "config", "Lcom/geniex/sdk/bean/TtsConfig;");
    if (!configObj) {
        out.config = nullptr;
    } else {
        *config    = extract_tts_config(env, configObj);
        out.config = config;
    }

    return out;
}

// TTS create - Initialize TTS with configuration
JNIEXPORT jlong JNICALL Java_com_geniex_sdk_jni_Tts_create(JNIEnv* env, jobject /*thiz*/, jobject ttsCreateInputObj) {
    try {
        geniex_TtsCreateInput input = extract_tts_create_input(env, ttsCreateInputObj);

        geniex_TTS* handle;
        LOGd("[JNI] create() geniex_tts_create called");
        int32_t err = geniex_tts_create(&input, &handle);
        if (err != GENIEX_SUCCESS || !handle) {
            LOGe("[JNI] geniex_tts_create failed, error code: %d", err);
            throw_runtime_exception(env, "TTS create failed, error code: %d", err);
            return 0;
        }
        LOGd("[JNI] create() geniex_tts_create returned handle=%p", handle);
        return reinterpret_cast<jlong>(handle);
    } catch (const std::exception& e) {
        LOGe("[JNI] create() exception: %s", e.what());
        return 0;
    }
}

// TTS destroy - Clean up TTS resources
JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_Tts_destroy(JNIEnv*, jobject, jlong handle) {
    LOGd("[JNI] geniex_tts_destroy called, handle=%p", (void*)handle);
    if (handle) {
        int32_t result = geniex_tts_destroy(reinterpret_cast<geniex_TTS*>(handle));
        if (result != GENIEX_SUCCESS) {
            LOGe("[JNI] geniex_tts_destroy failed, error code: %d", result);
        }
        return result;
    }
    return 0;
}

// TTS synthesize
JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Tts_synthesize(
    JNIEnv* env, jobject, jlong handle, jobject synthesizeInputObj) {
    if (!synthesizeInputObj) {
        LOGe("synthesizeInputObj is null");
        return nullptr;
    }
    if (!handle) {
        LOGe("TTS has been destroyed");
        return nullptr;
    }
    jclass cls = env->GetObjectClass(synthesizeInputObj);
    if (!cls) {
        LOGe("[JNI] synthesizeInputObj: GetObjectClass returned null");
        return nullptr;
    }

    geniex_TTSConfig          config;
    geniex_TtsSynthesizeInput input = extract_tts_synthesize_input(env, cls, synthesizeInputObj, &config);
    env->DeleteLocalRef(cls);

    geniex_TtsSynthesizeOutput output = {};
    int32_t                    err    = geniex_tts_synthesize((geniex_TTS*)handle, &input, &output);

    if (err < 0 || !output.result.audio_path) {
        throw_runtime_exception(env, "TTS synthesize failed, error code: %d", err);
        return nullptr;
    }

    // result
    jclass    resultClsOut = env->FindClass("com/geniex/sdk/bean/TtsResult");
    jmethodID resultCtor   = env->GetMethodID(resultClsOut, "<init>", "(Ljava/lang/String;FIII)V");
    LOGi("JNI TTS Synthesize result: audio_path='%s', duration=%.2f, sample_rate=%d",
        output.result.audio_path ? output.result.audio_path : "",
        output.result.duration_seconds,
        output.result.sample_rate);

    jobject resultObj = env->NewObject(resultClsOut,
        resultCtor,
        env->NewStringUTF(output.result.audio_path),
        output.result.duration_seconds,
        output.result.sample_rate,
        output.result.channels,
        output.result.num_samples);

    geniex_free((void*)output.result.audio_path);

    // profile_data
    jobject profileDataObj = jniutils::extract_profiling_data(env, output.profile_data);

    // output
    jclass    clsOut = env->FindClass("com/geniex/sdk/bean/TtsSynthesizeOutput");
    jmethodID ctor =
        env->GetMethodID(clsOut, "<init>", "(Lcom/geniex/sdk/bean/TtsResult;Lcom/geniex/sdk/bean/ProfilingData;)V");
    jobject result = env->NewObject(clsOut, ctor, resultObj, profileDataObj);
    return result;
}

// TTS listAvailableVoices
JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Tts_listAvailableVoices(JNIEnv* env, jobject, jlong handle) {
    geniex_TtsListAvailableVoicesInput  input = {};
    geniex_TtsListAvailableVoicesOutput out   = {};
    int32_t result = geniex_tts_list_available_voices(reinterpret_cast<const geniex_TTS*>(handle), &input, &out);
    if (result == GENIEX_SUCCESS) {
        return create_string_list(env, out.voice_ids, out.voice_count);
    } else {
        LOGe("get available voices failed and error code: %d", result);
    }
    return nullptr;
}

}  // extern "C"
