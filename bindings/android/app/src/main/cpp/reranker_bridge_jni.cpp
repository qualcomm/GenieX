#include <jni.h>

#include <string>
#include <vector>

#include "android_utils.h"
#include "jniutils.h"
#include "ml.h"

using namespace jniutils;
using namespace geniex_android_sdk;

extern "C" {

// create - Initialize Reranker with configuration
JNIEXPORT jlong JNICALL Java_com_geniex_sdk_jni_Reranker_create(JNIEnv* env, jobject, jobject rerankerCreateInputObj) {
    try {
        // Extract RerankerCreateInput from Java object
        ml_RerankerCreateInput input = extract_reranker_create_input(env, rerankerCreateInputObj);
        ml_Reranker*           h     = nullptr;
        LOGd("[JNI] create() ml_reranker_create called");

        int32_t result = ml_reranker_create(&input, &h);

        if (result != ML_SUCCESS || !h) {
            LOGe("[JNI] create() failed, error code: %d", result);
            throw_runtime_exception(env, "Reranker create failed, error code: %d", result);
            return 0;
        }

        LOGd("[JNI] create() ml_reranker_create returned handle=%p", h);
        return (jlong)h;
    } catch (const std::exception& e) {
        LOGe("[JNI] create() exception: %s", e.what());
        return 0;
    }
}

// destroy - Clean up Reranker resources
JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_Reranker_destroy(JNIEnv*, jobject, jlong handle) {
    LOGd("[JNI] destroy() called, handle=%p", (void*)handle);
    if (handle) {
        int32_t result = ml_reranker_destroy((ml_Reranker*)handle);
        if (result != ML_SUCCESS) {
            LOGe("[JNI] destroy() failed, error code: %d", result);
        }
        return result;
    }
    return 0;
}

// rerank
JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Reranker_rerank(
    JNIEnv* env, jobject, jlong handle, jstring query, jobjectArray jdocs, jobject configObj) {
    std::string              c_query = jstring2str(env, query);
    std::vector<std::string> docVec  = jstringArray2vec(env, jdocs);
    std::vector<const char*> c_docs;
    for (auto& s : docVec) c_docs.push_back(s.c_str());

    ml_RerankConfig cfg = extract_rerank_config(env, configObj);

    ml_RerankerRerankInput input = {};
    input.query                  = c_query.c_str();
    input.documents              = c_docs.data();
    input.documents_count        = c_docs.size();
    input.config                 = &cfg;

    ml_RerankerRerankOutput output = {};
    int32_t                 status = ml_reranker_rerank((ml_Reranker*)handle, &input, &output);

    if (status < 0 || !output.scores || output.score_count <= 0) {
        throw_runtime_exception(env, "Reranker rerank failed, error code: %d", status);
        return nullptr;
    }

    jfloatArray jarr = env->NewFloatArray(output.score_count);
    env->SetFloatArrayRegion(jarr, 0, output.score_count, output.scores);
    ml_free(output.scores);

    //
    jint      scoreCount     = output.score_count;
    jobject   profileDataObj = create_java_profile_data(env, output.profile_data);
    jclass    resultCls      = env->FindClass("com/geniex/sdk/bean/RerankerOutputResult");
    jmethodID ctor           = env->GetMethodID(resultCls, "<init>", "([FILcom/geniex/sdk/bean/ProfilingData;)V");
    jobject   result         = env->NewObject(resultCls, ctor, jarr, scoreCount, profileDataObj);
    return result;
}

}  // extern "C"