//
// Created by echonfl on 2025/11/3.
//

#ifndef GENIEXSDK_ANDROID_UTILS_H
#define GENIEXSDK_ANDROID_UTILS_H

#include <android/log.h>
#include <jni.h>

#include "geniex.h"

namespace geniex_android_sdk {
#define TAG "GeniexSdk"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGd(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

void throw_runtime_exception(JNIEnv* env, const char* _Nonnull format, ...);

jobject create_java_profile_data(JNIEnv* env, geniex_ProfileData data);

bool check_jni_exception(JNIEnv* env, const char* where);

const char* getStringField(JNIEnv* env, jclass cls, jobject inputObj, const char* fieldName);

jint getIntField(JNIEnv* env, jclass cls, jobject obj, const char* fieldName);

jfloat getFloatField(JNIEnv* env, jclass cls, jobject obj, const char* fieldName);

jobject getObjectField(JNIEnv* env, jclass cls, jobject obj, const char* name, const char* sig);

jboolean getBoolField(JNIEnv* env, jclass cls, jobject obj, const char* fieldName);

// Primitive field getters (for non-nullable Kotlin types)
jfloat getPrimitiveFloatField(JNIEnv* env, jclass cls, jobject obj, const char* fieldName);
jint   getPrimitiveIntField(JNIEnv* env, jclass cls, jobject obj, const char* fieldName);

jobject create_string_list(JNIEnv* env, const char** strings, int count);

jobject create_float_list(JNIEnv* env, float* floats, int count);

}  // namespace geniex_android_sdk

#endif  // GENIEXSDK_ANDROID_UTILS_H
