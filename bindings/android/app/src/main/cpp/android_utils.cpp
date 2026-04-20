//
// Created by echonfl on 2025/11/3.
//

#include "android_utils.h"

#include <jni.h>
#include <stdio.h>

#include "jniutils.h"

#define MAX_PATH_LEN 512
namespace geniex_android_sdk {

void throw_runtime_exception(JNIEnv *env, const char *_Nonnull format, ...) {
    va_list va;
    jclass  excCls = env->FindClass("java/lang/RuntimeException");
    if (excCls) {
        char errMsg[128];
        snprintf(errMsg, sizeof(errMsg), format, va);
        env->ThrowNew(excCls, errMsg);
    }
}

jobject create_java_profile_data(JNIEnv *env, geniex_ProfileData data) {
    jclass cls = env->FindClass("com/geniex/sdk/bean/ProfilingData");
    if (!cls) return nullptr;

    // (DDDJJJDDDLjava/lang/String;)V
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(DDDJJJDDDLjava/lang/String;)V");
    if (!ctor) return nullptr;

    const auto ttft_ms   = static_cast<jdouble>(data.ttft / 1000.0);
    const auto prompt_ms = static_cast<jdouble>(data.prompt_time / 1000.0);
    const auto decode_ms = static_cast<jdouble>(data.decode_time / 1000.0);

    const auto prompt_tokens = static_cast<jlong>(data.prompt_tokens);
    const auto gen_tokens    = static_cast<jlong>(data.generated_tokens);
    const auto audio_ms      = static_cast<jlong>(data.audio_duration / 1000);

    // ---- compute speeds/rtf locally (tokens/sec; RTF: audio/proc) ----
    auto tok_per_s = [](int64_t tokens, int64_t time_us) -> jdouble {
        if (time_us <= 0) return 0.0;
        return static_cast<jdouble>(tokens) * 1e6 / static_cast<jdouble>(time_us);
    };

    const jdouble prefill_speed  = tok_per_s(data.prompt_tokens, data.prompt_time);
    const jdouble decoding_speed = tok_per_s(data.generated_tokens, data.decode_time);

    const int64_t proc_us =
        (data.prompt_time > 0 ? data.prompt_time : 0) + (data.decode_time > 0 ? data.decode_time : 0);
    const jdouble rtf = (proc_us > 0) ? static_cast<jdouble>(data.audio_duration) / static_cast<jdouble>(proc_us) : 0.0;

    LOGd("prefill_speed=%.6f tok/s, decoding_speed=%.6f tok/s, rtf=%.4f", prefill_speed, decoding_speed, rtf);

    jstring jStopReason = env->NewStringUTF(data.stop_reason ? data.stop_reason : "");

    jobject obj = env->NewObject(cls,
        ctor,
        ttft_ms,
        prompt_ms,
        decode_ms,  // D D D
        prompt_tokens,
        gen_tokens,
        audio_ms,  // J J J
        prefill_speed,
        decoding_speed,
        rtf,         // D D D
        jStopReason  // String
    );

    env->DeleteLocalRef(jStopReason);
    return obj;
}

bool check_jni_exception(JNIEnv *env, const char *where) {
    if (env->ExceptionCheck()) {
        LOGe("Exception at %s", where);
        env->ExceptionDescribe();
        env->ExceptionClear();
        return true;
    }
    return false;
}

/**
 * env->DeleteLocalRef(cls); after extract all
 * @param env
 * @param cls
 * @param inputObj
 * @param fieldName
 * @return
 */
const char *getStringField(JNIEnv *env, jclass cls, jobject inputObj, const char *fieldName) {
    jfieldID fid = env->GetFieldID(cls, fieldName, "Ljava/lang/String;");
    if (check_jni_exception(env, "GetFieldID failed") || !fid) {
        LOGe("field '%s' not found", fieldName);
        return nullptr;
    }
    jstring jstr = (jstring)env->GetObjectField(inputObj, fid);
    if (check_jni_exception(env, "GetObjectField failed") || !jstr) {
        LOGd("'%s' is null", fieldName);
        return nullptr;
    }
    std::string s = jniutils::jstring2str(env, jstr);
    env->DeleteLocalRef(jstr);
    const char *c = jniutils::hold_c_str(s);
    LOGd("%s = %s", fieldName, c);
    return c;
}

/**
 * env->DeleteLocalRef(cls); after extract all
 * @param env
 * @param cls
 * @param obj
 * @param fieldName
 * @return
 */
jint getIntField(JNIEnv *env, jclass cls, jobject obj, const char *fieldName) {
    jfieldID fieldId = env->GetFieldID(cls, fieldName, "Ljava/lang/Integer;");
    if (check_jni_exception(env, "GetFieldID failed") || !fieldId) {
        LOGe("field '%s' not found", fieldName);
        return 0;
    }

    jobject intObj = env->GetObjectField(obj, fieldId);
    if (check_jni_exception(env, "GetObjectField failed") || !intObj) {
        LOGd("'%s' is null", fieldName);
        return 0;
    }

    jclass    integerClass   = env->FindClass("java/lang/Integer");
    jmethodID intValueMethod = env->GetMethodID(integerClass, "intValue", "()I");
    jint      result         = env->CallIntMethod(intObj, intValueMethod);

    env->DeleteLocalRef(intObj);
    env->DeleteLocalRef(integerClass);

    return result;
}

/**
 * env->DeleteLocalRef(cls); after extract all
 * @param env
 * @param cls
 * @param obj
 * @param fieldName
 * @return
 */
jfloat getFloatField(JNIEnv *env, jclass cls, jobject obj, const char *fieldName) {
    jfieldID fieldId = env->GetFieldID(cls, fieldName, "Ljava/lang/Float;");
    if (!fieldId) {
        LOGe("field '%s' not found", fieldName);
        return 0.0f;
    }

    jobject floatObj = env->GetObjectField(obj, fieldId);
    if (check_jni_exception(env, "GetObjectField failed") || !floatObj) {
        LOGd("'%s' is null", fieldName);
        return 0.0f;
    }

    jclass    floatClass       = env->FindClass("java/lang/Float");
    jmethodID floatValueMethod = env->GetMethodID(floatClass, "floatValue", "()F");
    jfloat    result           = env->CallFloatMethod(floatObj, floatValueMethod);

    env->DeleteLocalRef(floatObj);
    env->DeleteLocalRef(floatClass);

    return result;
}

jobject getObjectField(JNIEnv *env, jclass cls, jobject obj, const char *name, const char *sig) {
    jfieldID configId = env->GetFieldID(cls, name, sig);
    if (check_jni_exception(env, "GetFieldID failed") || !configId) {
        LOGe("field '%s' not found", name);
        return nullptr;
    } else {
        jobject configObj = env->GetObjectField(obj, configId);
        if (check_jni_exception(env, "GetObjectField failed") || !configObj) {
            LOGe("field '%s' is null", name);
            return nullptr;
        } else {
            return configObj;
        }
    }
}

jboolean getBoolField(JNIEnv *env, jclass cls, jobject obj, const char *fieldName) {
    jfieldID fieldId = env->GetFieldID(cls, fieldName, "Z");
    if (check_jni_exception(env, "GetFieldID failed") || !fieldId) {
        LOGe("field '%s' not found", fieldName);
        return 0;
    }

    jboolean boolObj = env->GetBooleanField(obj, fieldId);
    if (check_jni_exception(env, "GetObjectField failed")) {
        LOGd("'%s' is null", fieldName);
        return 0;
    }
    return boolObj;
}

/**
 * Get primitive float field (for non-nullable Kotlin Float)
 * JNI signature: "F"
 */
jfloat getPrimitiveFloatField(JNIEnv *env, jclass cls, jobject obj, const char *fieldName) {
    jfieldID fieldId = env->GetFieldID(cls, fieldName, "F");
    if (check_jni_exception(env, "GetFieldID failed") || !fieldId) {
        LOGe("primitive float field '%s' not found", fieldName);
        return 0.0f;
    }

    jfloat result = env->GetFloatField(obj, fieldId);
    if (check_jni_exception(env, "GetFloatField failed")) {
        LOGd("'%s' read failed", fieldName);
        return 0.0f;
    }
    return result;
}

/**
 * Get primitive int field (for non-nullable Kotlin Int)
 * JNI signature: "I"
 */
jint getPrimitiveIntField(JNIEnv *env, jclass cls, jobject obj, const char *fieldName) {
    jfieldID fieldId = env->GetFieldID(cls, fieldName, "I");
    if (check_jni_exception(env, "GetFieldID failed") || !fieldId) {
        LOGe("primitive int field '%s' not found", fieldName);
        return 0;
    }

    jint result = env->GetIntField(obj, fieldId);
    if (check_jni_exception(env, "GetIntField failed")) {
        LOGd("'%s' read failed", fieldName);
        return 0;
    }
    return result;
}

jobject create_string_list(JNIEnv *env, const char **strings, int count) {
    jclass    array_list_class       = env->FindClass("java/util/ArrayList");
    jmethodID array_list_constructor = env->GetMethodID(array_list_class, "<init>", "()V");
    jmethodID array_list_add         = env->GetMethodID(array_list_class, "add", "(Ljava/lang/Object;)Z");

    jobject list = env->NewObject(array_list_class, array_list_constructor);

    for (int i = 0; i < count; i++) {
        if (strings[i]) {
            jstring str = env->NewStringUTF(strings[i]);
            env->CallBooleanMethod(list, array_list_add, str);
            env->DeleteLocalRef(str);
        }
    }

    return list;
}

jobject create_float_list(JNIEnv *env, float *floats, int count) {
    jclass    array_list_class       = env->FindClass("java/util/ArrayList");
    jmethodID array_list_constructor = env->GetMethodID(array_list_class, "<init>", "()V");
    jmethodID array_list_add         = env->GetMethodID(array_list_class, "add", "(Ljava/lang/Object;)Z");

    jclass    float_class       = env->FindClass("java/lang/Float");
    jmethodID float_constructor = env->GetMethodID(float_class, "<init>", "(F)V");

    jobject list = env->NewObject(array_list_class, array_list_constructor);

    for (int i = 0; i < count; i++) {
        jobject float_obj = env->NewObject(float_class, float_constructor, floats[i]);
        env->CallBooleanMethod(list, array_list_add, float_obj);
        env->DeleteLocalRef(float_obj);
    }

    return list;
}

}  // namespace geniex_android_sdk
