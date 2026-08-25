// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//
// Created by echonfl on 2025/11/3.
//

#include "android_utils.h"

#include <jni.h>
#include <stdarg.h>
#include <stdio.h>

#include "jniutils.h"

#define MAX_PATH_LEN 512
namespace geniex_android_sdk {

void throw_runtime_exception(JNIEnv *env, const char *_Nonnull format, ...) {
    va_list va;
    va_start(va, format);
    char errMsg[128];
    vsnprintf(errMsg, sizeof(errMsg), format, va);
    va_end(va);

    jclass excCls = env->FindClass("java/lang/RuntimeException");
    if (excCls) {
        env->ThrowNew(excCls, errMsg);
    }
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
