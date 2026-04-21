#include <dlfcn.h>
#include <jni.h>
#include <pthread.h>
#include <sys/stat.h>  // For chmod()
#include <unistd.h>    // For access()
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "android_utils.h"
#include "geniex.h"
#include "jni_cb.h"
#include "jniutils.h"

using namespace jniutils;

// Global map to track stop flags for each LLM handle
static std::unordered_map<void*, std::atomic<bool>*> g_stopFlags;
static std::mutex                                    g_stopFlagsMutex;

using namespace jniutils;
using namespace geniex_android_sdk;

// JNI: create - Initialize LLM with configuration
extern "C" JNIEXPORT jlong JNICALL Java_com_geniex_sdk_jni_Llm_create(
    JNIEnv* env, jobject thiz, jobject llm_create_input_obj) {
    try {
        geniex_LlmCreateInput create_input = extract_llm_create_input(env, llm_create_input_obj);
        geniex_LLM*           handle       = nullptr;
        LOGd("[JNI] create() geniex_llm_create called with:");
        LOGd("  model_name: %s", create_input.model_name);
        LOGd("  model_path: %s", create_input.model_path ? create_input.model_path : "(null)");
        LOGd("  tokenizer_path: %s", create_input.tokenizer_path ? create_input.tokenizer_path : "(null)");
        LOGd("  config.npu_model_folder_path (qnn): %s",
            create_input.config.qnn_model_folder_path ? create_input.config.qnn_model_folder_path : "(null)");
        LOGd("  config.npu_lib_folder_path (qnn): %s",
            create_input.config.qnn_lib_folder_path ? create_input.config.qnn_lib_folder_path : "(null)");
        LOGd("  config.max_tokens: %d", create_input.config.max_tokens);
        LOGd("  config.enable_thinking: %s", create_input.config.enable_thinking ? "true" : "false");
        LOGd("  config.verbose: %s", create_input.config.verbose ? "true" : "false");
        LOGd("  plugin_id: %s", create_input.plugin_id ? create_input.plugin_id : "(null)");

        int32_t result = geniex_llm_create(&create_input, &handle);

        if (result != GENIEX_SUCCESS || !handle) {
            LOGe("[JNI] create() failed, error code: %d", result);
            throw_runtime_exception(env, "Llm create failed, error code: %d", result);
            return 0;
        }
        LOGd("[JNI] create() geniex_llm_create returned handle=%p", handle);
        return reinterpret_cast<jlong>(handle);

    } catch (const std::exception& e) {
        LOGe("[JNI] create() exception: %s", e.what());
        return 0;
    }
}

// JNI: destroy - Clean up LLM resources
extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_Llm_destroy(JNIEnv*, jobject, jlong handle) {
    LOGd("[JNI] destroy() called, handle=%p", (void*)handle);
    if (handle) {
        int32_t result = geniex_llm_destroy((geniex_LLM*)handle);
        if (result != GENIEX_SUCCESS) {
            LOGe("[JNI] destroy() failed, error code: %d", result);
        }
        return result;
    }
    return 0;
}

// JNI: generate - Generate text with streaming support
extern "C" JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Llm_generate(
    JNIEnv* env, jobject /*thiz*/, jlong handle, jstring prompt, jobject configObj, jobject callback) {
    try {
        void* h = (void*)handle;

        // Setup stop flag for streaming control
        std::atomic<bool>* stop_flag = nullptr;
        if (callback) {
            std::lock_guard<std::mutex> lock(g_stopFlagsMutex);
            if (g_stopFlags.count(h)) delete g_stopFlags[h];
            g_stopFlags[h] = new std::atomic<bool>(false);
            stop_flag      = g_stopFlags[h];
        }

        std::string             cprompt = jstring2str(env, prompt);
        geniex_GenerationConfig cfg     = extract_generation_config(env, configObj);

        geniex_LlmGenerateInput  input  = {};
        geniex_LlmGenerateOutput output = {};
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

        int32_t ret = geniex_llm_generate(reinterpret_cast<geniex_LLM*>(handle), &input, &output);
        if (ret < 0 || !output.full_text) {
            if (callback) {
                std::lock_guard<std::mutex> lock(g_stopFlagsMutex);
                delete g_stopFlags[h];
                g_stopFlags.erase(h);
            }
            throw_runtime_exception(env, "%d", ret);
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

    } catch (const std::exception& e) {
        LOGe("[Llm JNI] generate() exception: %s", e.what());
        return nullptr;
    } catch (...) {
        LOGe("[Llm JNI] generate() unknown exception");
        return nullptr;
    }
}

// JNI: stopStream - Stop ongoing generation
extern "C" JNIEXPORT void JNICALL Java_com_geniex_sdk_jni_Llm_stopStream(JNIEnv*, jobject, jlong handle) {
    void*                       h = (void*)handle;
    std::lock_guard<std::mutex> lock(g_stopFlagsMutex);
    if (g_stopFlags.count(h)) g_stopFlags[h]->store(true);
}

// JNI: applyChatTemplate - Format messages with chat template
extern "C" JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Llm_applyChatTemplate(JNIEnv* env, jobject thiz,
    jlong handle, jobjectArray jmessages, jstring jtools, jboolean jEnableThinking, jboolean jAddGenerationPrompt) {
    static thread_local std::vector<std::string> str_buf;
    auto                                         msgs = extract_llm_chat_messages(env, jmessages, str_buf);

    const char* tools_cstr = nullptr;
    if (jtools != nullptr) {
        tools_cstr = env->GetStringUTFChars(jtools, nullptr);
    }

    geniex_LlmApplyChatTemplateInput  input{.messages = msgs.data(),
         .message_count                               = static_cast<int32_t>(msgs.size()),
         .tools                                       = tools_cstr,
         .enable_thinking                             = (jEnableThinking == JNI_TRUE),
         .add_generation_prompt                       = (jAddGenerationPrompt == JNI_TRUE)};
    geniex_LlmApplyChatTemplateOutput output{};

    int32_t ret = geniex_llm_apply_chat_template(reinterpret_cast<geniex_LLM*>(handle), &input, &output);

    if (tools_cstr) {
        env->ReleaseStringUTFChars(jtools, tools_cstr);
    }

    jclass    cls  = env->FindClass("com/geniex/sdk/bean/LlmApplyChatTemplateOutput");
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(Ljava/lang/String;)V");

    if (ret < 0 || !output.formatted_text) {
        jobject result = env->NewObject(cls, ctor, env->NewStringUTF(""));
        return result;
    }

    jstring formatted = env->NewStringUTF(output.formatted_text);
    jobject result    = env->NewObject(cls, ctor, formatted);
    free(output.formatted_text);
    return result;
}

extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_Llm_reset(JNIEnv* env, jobject thiz, jlong handle) {
    return geniex_llm_reset(reinterpret_cast<geniex_LLM*>(handle));
}