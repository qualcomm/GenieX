#include <jni.h>

#include <string>
#include <vector>

#include "android_utils.h"
#include "geniex.h"
#include "jniutils.h"

using namespace jniutils;
using namespace geniex_android_sdk;

// JNI: create - Initialize Embedder with configuration
extern "C" JNIEXPORT jlong JNICALL Java_com_geniex_sdk_jni_Embedder_create(
    JNIEnv* env, jobject, jobject embedderCreateInputObj) {
    try {
        // Extract EmbedderCreateInput from Java object
        geniex_EmbedderCreateInput input = extract_embedder_create_input(env, embedderCreateInputObj);
        geniex_Embedder*           h     = nullptr;
        LOGd("[JNI] create() geniex_embedder_create called");

        int32_t result = geniex_embedder_create(&input, &h);

        if (result != GENIEX_SUCCESS || !h) {
            LOGe("[JNI] create() failed, error code: %d", result);
            throw_runtime_exception(env, "Embedder create failed, error code: %d", result);
            return 0;
        }

        LOGd("[JNI] create() geniex_embedder_create returned handle=%p", h);
        return (jlong)h;
    } catch (const std::exception& e) {
        LOGe("[JNI] create() exception: %s", e.what());
        return 0;
    }
}

// JNI: destroy - Clean up Embedder resources
extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_Embedder_destroy(JNIEnv*, jobject, jlong handle) {
    LOGd("[JNI] destroy() called, handle=%p", (void*)handle);
    if (handle) {
        int32_t result = geniex_embedder_destroy((geniex_Embedder*)handle);
        if (result != GENIEX_SUCCESS) {
            LOGe("[JNI] destroy() failed, error code: %d", result);
        }
        return result;
    }
    return 0;
}

// JNI: embed
extern "C" JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Embedder_embed(
    JNIEnv* env, jobject, jlong handle, jobjectArray jtexts, jobject configObj) {
    std::vector<std::string> texts = jstringArray2vec(env, jtexts);
    std::vector<const char*> c_texts;
    for (auto& s : texts) c_texts.push_back(s.c_str());

    geniex_EmbeddingConfig cfg = extract_embedding_config(env, configObj);

    geniex_EmbedderEmbedInput input = {};
    input.texts                     = c_texts.data();
    input.text_count                = c_texts.size();
    input.config                    = &cfg;

    geniex_EmbedderEmbedOutput output = {};
    int32_t                    result = geniex_embedder_embed((geniex_Embedder*)handle, &input, &output);

    if (result < 0 || !output.embeddings || output.embedding_count <= 0) {
        throw_runtime_exception(env, "Embedder embed failed, error code: %d", result);
        return nullptr;
    }

    // Get embedding dimension from the embedder
    geniex_EmbedderDimOutput dim_output = {};
    int32_t                  dim_result = geniex_embedder_embedding_dim((const geniex_Embedder*)handle, &dim_output);
    if (dim_result < 0 || dim_output.dimension <= 0) {
        geniex_free(output.embeddings);
        return nullptr;
    }

    // Calculate total floats: embedding_count * dimension
    jsize       total = output.embedding_count * dim_output.dimension;
    jfloatArray jarr  = env->NewFloatArray(total);
    env->SetFloatArrayRegion(jarr, 0, total, output.embeddings);

    // profileData
    jobject profileDataObj = extract_profiling_data(env, output.profile_data);

    jclass    resultCls = env->FindClass("com/geniex/sdk/bean/EmbedResult");
    jmethodID ctor      = env->GetMethodID(resultCls, "<init>", "([FLcom/geniex/sdk/bean/ProfilingData;)V");
    jobject   resultObj = env->NewObject(resultCls, ctor, jarr, profileDataObj);

    geniex_free(output.embeddings);
    return resultObj;
}

// JNI: embeddingDim
extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_Embedder_embeddingDim(JNIEnv*, jobject, jlong handle) {
    geniex_EmbedderDimOutput output = {};
    int32_t                  result = geniex_embedder_embedding_dim((const geniex_Embedder*)handle, &output);
    if (result < 0) return -1;
    return output.dimension;
}

// JNI: setLora - Not supported in current API
extern "C" JNIEXPORT void JNICALL Java_com_geniex_sdk_jni_Embedder_setLora(
    JNIEnv*, jobject, jlong handle, jint loraId) {
    // Not supported in current API
}

// JNI: addLora - Not supported in current API
extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_Embedder_addLora(
    JNIEnv* env, jobject, jlong handle, jstring loraPath) {
    // Not supported in current API
    return -1;
}

// JNI: removeLora - Not supported in current API
extern "C" JNIEXPORT void JNICALL Java_com_geniex_sdk_jni_Embedder_removeLora(
    JNIEnv*, jobject, jlong handle, jint loraId) {
    // Not supported in current API
}

// JNI: listLoras - Not supported in current API
extern "C" JNIEXPORT jintArray JNICALL Java_com_geniex_sdk_jni_Embedder_listLoras(JNIEnv* env, jobject, jlong handle) {
    // Not supported in current API
    return nullptr;
}

// JNI: getProfilingData - Get profiling data for Embedder
extern "C" JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Embedder_getProfilingData(
    JNIEnv* env, jobject, jlong handle) {
    // Embedder API doesn't provide separate get_profiling_data function
    // Profiling data comes from embed() output
    geniex_ProfileData data = {};
    return extract_profiling_data(env, data);
}