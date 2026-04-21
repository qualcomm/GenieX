#include <android/log.h>
#include <jni.h>
#include <pthread.h>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "android_utils.h"
#include "geniex.h"
#include "jni_cb.h"
#include "jniutils.h"

static std::unordered_map<void*, std::atomic<bool>*> g_stopFlags;
static std::mutex                                    g_stopFlagsMutex;

using namespace jniutils;
using namespace geniex_android_sdk;

// JNI: create
extern "C" JNIEXPORT jlong JNICALL Java_com_geniex_sdk_jni_Vlm_create(JNIEnv* env, jobject, jobject vlmCreateInputObj) {
    try {
        LOGi("[JNI] [create] Java_com_geniex_sdk_jni_Vlm_create ");
        geniex_VlmCreateInput create_input = extract_vlm_create_input(env, vlmCreateInputObj);
        LOGi("[JNI] [create] model_name = %s", create_input.model_name ? create_input.model_name : "(null)");
        LOGi("[JNI] [create] plugin_id = %s", create_input.plugin_id ? create_input.plugin_id : "(null)");
        geniex_VLM* handle = nullptr;
        int32_t     result = geniex_vlm_create(&create_input, &handle);

        if (result != GENIEX_SUCCESS || !handle) {
            LOGe("[JNI] create() failed, error code: %d", result);
            throw_runtime_exception(env, "Model create() failed, error code: %d", result);
            return 0;
        }

        LOGi("[JNI] create() geniex_vlm_create returned handle=%p", handle);
        return reinterpret_cast<jlong>(handle);

    } catch (const std::exception& e) {
        LOGe("[JNI] create() exception: %s", e.what());
        return 0;
    }
}

// JNI: destroy - Clean up VLM resources
extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_Vlm_destroy(JNIEnv*, jobject, jlong handle) {
    LOGd("[JNI] destroy() called, handle=%p", (void*)handle);
    if (handle) {
        int32_t result = geniex_vlm_destroy(reinterpret_cast<geniex_VLM*>(handle));
        if (result != GENIEX_SUCCESS) {
            LOGe("[JNI] destroy() failed, error code: %d", result);
        }
        return result;
    }
    return 0;
}

// JNI: reset - Reset VLM internal state
extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_Vlm_reset(JNIEnv* env, jobject, jlong handle) {
    LOGd("[JNI] reset() called, handle=%p", (void*)handle);
    if (!handle) {
        throw_runtime_exception(env, "VLM reset failed: invalid handle");
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }
    int32_t result = geniex_vlm_reset(reinterpret_cast<geniex_VLM*>(handle));
    if (result != GENIEX_SUCCESS) {
        LOGe("[JNI] reset() failed, error code: %d", result);
    }
    return result;
}

// JNI: encode
// extern "C"
// JNIEXPORT jint JNICALL
// Java_com_geniex_sdk_jni_Vlm_encode(JNIEnv *env, jobject, jlong handle, jstring text,
//                                 jintArray outTokens) {
//    std::string c_text = jstring2str(env, text);
//    int32_t *tokens = nullptr;
//    int32_t num_tokens = geniex_vlm_encode(reinterpret_cast<geniex_VLM *>(handle), c_text.c_str(), &tokens);
//    if (num_tokens > 0 && tokens && outTokens) {
//        env->SetIntArrayRegion(outTokens, 0, num_tokens, tokens);
//    }
//    free(tokens);
//    return num_tokens;
//}

// JNI: decode
// extern "C"
// JNIEXPORT jstring JNICALL
// Java_com_geniex_sdk_jni_Vlm_decode(JNIEnv *env, jobject, jlong handle, jintArray tokens,
//                                 jint length) {
//    std::vector<int32_t> c_tokens = jintArray2vec(env, tokens);
//    char *out_text = nullptr;
//    int32_t ret = geniex_vlm_decode(reinterpret_cast<geniex_VLM *>(handle), c_tokens.data(), length,
//                                &out_text);
//    if (ret <= 0 || !out_text) return nullptr;
//    jstring jresult = env->NewStringUTF(out_text);
//    free(out_text);
//    return jresult;
//}

// JNI: generate
extern "C" JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Vlm_generate(
    JNIEnv* env, jobject /*thiz*/, jlong handle, jstring prompt, jobject configObj, jobject callback) {
    try {
        void* h = (void*)handle;

        std::atomic<bool>* stop_flag = nullptr;
        if (callback) {
            std::lock_guard<std::mutex> lock(g_stopFlagsMutex);
            if (g_stopFlags.count(h)) delete g_stopFlags[h];
            g_stopFlags[h] = new std::atomic<bool>(false);
            stop_flag      = g_stopFlags[h];
        }

        std::string             cprompt = jstring2str(env, prompt);
        geniex_GenerationConfig cfg     = extract_generation_config(env, configObj);

        geniex_VlmGenerateInput  input  = {};
        geniex_VlmGenerateOutput output = {};
        input.prompt_utf8               = cprompt.c_str();
        input.config                    = &cfg;

        JavaCallbackCtx cbCtx{};
        if (callback) {
            if (!jni_cb_init(env,
                    callback,
                    /* onToken   */ "onToken",
                    "(Ljava/lang/String;)Z",
                    /* onComplete*/ "onComplete",
                    "(Lcom/geniex/sdk/bean/LlmGenerateResult;)V",
                    stop_flag,
                    &cbCtx)) {
                std::lock_guard<std::mutex> lock(g_stopFlagsMutex);
                delete g_stopFlags[h];
                g_stopFlags.erase(h);
                return nullptr;
            }
            input.on_token = [](const char* token, void* user) -> bool {
                return jni_cb_emit_token(reinterpret_cast<JavaCallbackCtx*>(user), token);
            };
            input.user_data = &cbCtx;
        }

        int32_t ret = geniex_vlm_generate(reinterpret_cast<geniex_VLM*>(handle), &input, &output);

        if (ret < 0 || !output.full_text) {
            if (callback) {
                std::lock_guard<std::mutex> lock(g_stopFlagsMutex);
                delete g_stopFlags[h];
                g_stopFlags.erase(h);
            }
            throw_runtime_exception(env, "VLM generate failed, error code: %d", ret);
            return nullptr;
        }

        jstring   fullText       = env->NewStringUTF(output.full_text);
        jobject   profileDataObj = extract_profiling_data(env, output.profile_data);
        jclass    resultCls      = env->FindClass("com/geniex/sdk/bean/LlmGenerateResult");
        jmethodID ctor =
            env->GetMethodID(resultCls, "<init>", "(Ljava/lang/String;Lcom/geniex/sdk/bean/ProfilingData;)V");
        jobject result = env->NewObject(resultCls, ctor, fullText, profileDataObj);

        if (callback) {
            jni_cb_call_complete(&cbCtx, result);

            if (env->ExceptionCheck()) {
                env->ExceptionDescribe();
                env->ExceptionClear();
            }

            jni_cb_dispose(env, &cbCtx);

            {
                std::lock_guard<std::mutex> lock(g_stopFlagsMutex);
                delete g_stopFlags[h];
                g_stopFlags.erase(h);
            }

            free(output.full_text);
            return nullptr;
        }

        free(output.full_text);
        return result;

    } catch (...) {
        return nullptr;
    }
}

// JNI: embed
// extern "C"
// JNIEXPORT jfloatArray JNICALL
// Java_com_geniex_sdk_jni_Vlm_embed(JNIEnv *env, jobject, jlong handle, jobjectArray jtexts) {
//    std::vector<std::string> texts = jstringArray2vec(env, jtexts);
//    std::vector<const char *> c_texts;
//    for (auto &s: texts) c_texts.push_back(s.c_str());
//    float *embeddings = nullptr;
//    int32_t dim = geniex_vlm_embed(reinterpret_cast<geniex_VLM *>(handle), c_texts.data(), c_texts.size(),
//                               &embeddings);
//    if (!embeddings || dim <= 0) return nullptr;
//    jfloatArray jarr = env->NewFloatArray(dim);
//    env->SetFloatArrayRegion(jarr, 0, dim, embeddings);
//    free(embeddings);
//    return jarr;
//}

// setSampler
// extern "C"
// JNIEXPORT void JNICALL
// Java_com_geniex_sdk_jni_Vlm_setSampler(JNIEnv *env, jobject, jlong handle, jobject cfgObj) {
//    geniex_SamplerConfig cfg = extract_sampler_config(env, cfgObj);
//    geniex_vlm_set_sampler(reinterpret_cast<geniex_VLM *>(handle), &cfg);
//}
//
//// resetSampler
// extern "C"
// JNIEXPORT void JNICALL Java_com_geniex_sdk_jni_Vlm_resetSampler(JNIEnv *, jobject, jlong handle) {
//     geniex_vlm_reset_sampler(reinterpret_cast<geniex_VLM *>(handle));
// }

extern "C" JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Vlm_applyChatTemplate(
    JNIEnv* env, jobject /*thiz*/, jlong handle, jobjectArray jmessages, jstring jtools, jboolean jEnableThinking) {
    clear_jni_cstr_pool();
    auto msgs = extract_vlm_chat_messages(env, jmessages);

    LOGe("[applyChatTemplate] message_count=%d", (int)msgs.size());

    const char* tools_cstr = nullptr;
    if (jtools != nullptr) {
        tools_cstr = env->GetStringUTFChars(jtools, nullptr);
    }

    geniex_VlmApplyChatTemplateInput  input{.messages = msgs.data(),
         .message_count                               = static_cast<int32_t>(msgs.size()),
         .tools                                       = tools_cstr,
         .enable_thinking                             = (jEnableThinking == JNI_TRUE)};
    geniex_VlmApplyChatTemplateOutput output{};

    int32_t ret = geniex_vlm_apply_chat_template(reinterpret_cast<geniex_VLM*>(handle), &input, &output);

    if (tools_cstr) {
        env->ReleaseStringUTFChars(jtools, tools_cstr);
    }

    if (ret < 0 || !output.formatted_text) {
        LOGe("[applyChatTemplate] failed! ret=%d, formatted_text=%p", ret, (void*)output.formatted_text);

        jclass    cls    = env->FindClass("com/geniex/sdk/bean/LlmApplyChatTemplateOutput");
        jmethodID ctor   = env->GetMethodID(cls, "<init>", "(Ljava/lang/String;)V");
        jobject   result = env->NewObject(cls, ctor, env->NewStringUTF(""));

        clear_jni_cstr_pool();
        return result;
    }

    jstring formatted = env->NewStringUTF(output.formatted_text);

    jclass    cls    = env->FindClass("com/geniex/sdk/bean/LlmApplyChatTemplateOutput");
    jmethodID ctor   = env->GetMethodID(cls, "<init>", "(Ljava/lang/String;)V");
    jobject   result = env->NewObject(cls, ctor, formatted);

    free(output.formatted_text);
    clear_jni_cstr_pool();
    return result;
}

extern "C" JNIEXPORT void JNICALL Java_com_geniex_sdk_jni_Vlm_stopStream(JNIEnv*, jobject, jlong handle) {
    void*                       h = (void*)handle;
    std::lock_guard<std::mutex> lock(g_stopFlagsMutex);
    if (g_stopFlags.count(h)) g_stopFlags[h]->store(true);
}

// Extract media paths from VlmChatMessage contents
extern "C" JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Vlm_extractMediaPaths(
    JNIEnv* env, jobject /*thiz*/, jobjectArray jmessages) {
    try {
        std::vector<std::string> image_paths;
        std::vector<std::string> audio_paths;

        extract_media_paths_from_messages(env, jmessages, image_paths, audio_paths);

        // Convert vectors to Java String arrays
        jobjectArray jImagePaths = env->NewObjectArray(image_paths.size(), env->FindClass("java/lang/String"), nullptr);
        for (size_t i = 0; i < image_paths.size(); ++i) {
            jstring jstr = env->NewStringUTF(image_paths[i].c_str());
            env->SetObjectArrayElement(jImagePaths, i, jstr);
            env->DeleteLocalRef(jstr);
        }

        jobjectArray jAudioPaths = env->NewObjectArray(audio_paths.size(), env->FindClass("java/lang/String"), nullptr);
        for (size_t i = 0; i < audio_paths.size(); ++i) {
            jstring jstr = env->NewStringUTF(audio_paths[i].c_str());
            env->SetObjectArrayElement(jAudioPaths, i, jstr);
            env->DeleteLocalRef(jstr);
        }

        // Create Pair<Array<String>, Array<String>>
        jclass    pairCls  = env->FindClass("kotlin/Pair");
        jmethodID pairCtor = env->GetMethodID(pairCls, "<init>", "(Ljava/lang/Object;Ljava/lang/Object;)V");
        jobject   pair     = env->NewObject(pairCls, pairCtor, jImagePaths, jAudioPaths);

        return pair;
    } catch (const std::exception& e) {
        LOGe("[JNI] extractMediaPaths() exception: %s", e.what());
        return nullptr;
    }
}