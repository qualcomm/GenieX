// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

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
        LOGd("  model_path: %s", create_input.model_path ? create_input.model_path : "(null)");
        LOGd("  tokenizer_path: %s", create_input.tokenizer_path ? create_input.tokenizer_path : "(null)");
        LOGd("  plugin_id: %s", create_input.plugin_id ? create_input.plugin_id : "(null)");

        int32_t result = geniex_llm_create(&create_input, &handle);

        if (result != GENIEX_SUCCESS || !handle) {
            LOGe("[JNI] create() failed, error code: %d", result);
            throw_runtime_exception(
                env, "Llm create failed: %s", geniex_get_error_message(static_cast<geniex_ErrorCode>(result)));
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

// JNI: generate - Generate text with streaming support. At most one of
// inputIds / inputEmbd should be non-null; when both are null, prompt is used.
// Priority when several are set: inputEmbd > inputIds > prompt (matches the C
// header's geniex_LlmGenerateInput contract).
extern "C" JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Llm_generate(JNIEnv* env, jobject /*thiz*/, jlong handle,
    jstring prompt, jintArray inputIds, jfloatArray inputEmbd, jint inputEmbdDim, jobject configObj, jobject callback) {
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

        std::string              cprompt        = prompt ? jstring2str(env, prompt) : std::string();
        auto                     config_storage = extract_generation_config(env, configObj);
        geniex_GenerationConfig* cfg            = config_storage.get();

        std::vector<int32_t> ids;
        std::vector<float>   embd;
        if (inputEmbd) {
            embd = jfloatArray2vec(env, inputEmbd);
        } else if (inputIds) {
            ids = jintArray2vec(env, inputIds);
        }

        geniex_LlmGenerateInput  input  = {};
        geniex_LlmGenerateOutput output = {};
        if (!embd.empty()) {
            input.input_embd       = embd.data();
            input.input_embd_count = static_cast<int32_t>(inputEmbdDim > 0 ? embd.size() / inputEmbdDim : 0);
            input.input_embd_dim   = inputEmbdDim;
        } else if (!ids.empty()) {
            input.input_ids       = ids.data();
            input.input_ids_count = static_cast<int32_t>(ids.size());
        } else {
            input.prompt_utf8 = cprompt.c_str();
        }
        input.config = cfg;

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
            throw_runtime_exception(
                env, "LLM generate failed: %s", geniex_get_error_message(static_cast<geniex_ErrorCode>(ret)));
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
// JNIEXPORT/JNICALL make the rule misread this void function as non-void.
// nosemgrep: reusable-workflows.semgrep_rules.cxx.funcret.gen.non-void-function-no-return
extern "C" JNIEXPORT void JNICALL Java_com_geniex_sdk_jni_Llm_stopStream(JNIEnv*, jobject, jlong handle) {
    void*                       h = (void*)handle;
    std::lock_guard<std::mutex> lock(g_stopFlagsMutex);
    if (g_stopFlags.count(h)) g_stopFlags[h]->store(true);
}

// JNI: applyChatTemplate - Format messages with chat template
extern "C" JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Llm_applyChatTemplate(JNIEnv* env, jobject thiz,
    jlong handle, jobjectArray jmessages, jstring jtools, jboolean jEnableThinking, jboolean jAddGenerationPrompt,
    jboolean jParallelToolCalls) {
    clear_jni_cstr_pool();
    auto msgs = extract_llm_chat_messages(env, jmessages);

    const char* tools_cstr = nullptr;
    if (jtools != nullptr) {
        tools_cstr = env->GetStringUTFChars(jtools, nullptr);
    }

    geniex_LlmApplyChatTemplateInput  input{.messages = msgs.data(),
         .message_count                               = static_cast<int32_t>(msgs.size()),
         .tools                                       = tools_cstr,
         .enable_thinking                             = (jEnableThinking == JNI_TRUE),
         .add_generation_prompt                       = (jAddGenerationPrompt == JNI_TRUE),
         .parallel_tool_calls                         = (jParallelToolCalls == JNI_TRUE)};
    geniex_LlmApplyChatTemplateOutput output{};

    int32_t ret = geniex_llm_apply_chat_template(reinterpret_cast<geniex_LLM*>(handle), &input, &output);

    if (tools_cstr) {
        env->ReleaseStringUTFChars(jtools, tools_cstr);
    }

    free_llm_chat_messages(msgs);
    clear_jni_cstr_pool();

    if (ret < 0 || !output.formatted_text) {
        LOGe("[applyChatTemplate] failed! ret=%d", ret);
        const char* reason =
            ret < 0 ? geniex_get_error_message(static_cast<geniex_ErrorCode>(ret)) : "formatted text is null";
        throw_runtime_exception(env, "applyChatTemplate failed: %s", reason);
        geniex_free(output.formatted_text);
        free_chat_template_grammar(output.chat_template_grammar);
        return nullptr;
    }

    jclass  cls       = nullptr;
    jstring formatted = nullptr;
    jobject grammar   = nullptr;
    auto    cleanup   = [&]() {
        geniex_free(output.formatted_text);
        free_chat_template_grammar(output.chat_template_grammar);
        if (formatted) env->DeleteLocalRef(formatted);
        if (grammar) env->DeleteLocalRef(grammar);
        if (cls) env->DeleteLocalRef(cls);
    };

    cls = env->FindClass("com/geniex/sdk/bean/LlmApplyChatTemplateOutput");
    if (!cls || env->ExceptionCheck()) {
        cleanup();
        return nullptr;
    }
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(Ljava/lang/String;Lcom/geniex/sdk/bean/ChatTemplateGrammar;)V");
    if (!ctor || env->ExceptionCheck()) {
        cleanup();
        return nullptr;
    }
    formatted = env->NewStringUTF(output.formatted_text);
    if (!formatted || env->ExceptionCheck()) {
        cleanup();
        return nullptr;
    }
    grammar = chat_template_grammar_to_java(env, output.chat_template_grammar);
    if (output.chat_template_grammar.grammar_type != GENIEX_GRAMMAR_TYPE_NONE && (!grammar || env->ExceptionCheck())) {
        if (!env->ExceptionCheck()) throw_runtime_exception(env, "failed to convert chat-template grammar");
        cleanup();
        return nullptr;
    }
    jobject result = env->NewObject(cls, ctor, formatted, grammar);
    if (!result || env->ExceptionCheck()) {
        if (!env->ExceptionCheck()) throw_runtime_exception(env, "failed to create chat-template output");
        if (result) env->DeleteLocalRef(result);
        cleanup();
        return nullptr;
    }

    cleanup();
    return result;
}

extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_Llm_reset(JNIEnv* env, jobject thiz, jlong handle) {
    return geniex_llm_reset(reinterpret_cast<geniex_LLM*>(handle));
}