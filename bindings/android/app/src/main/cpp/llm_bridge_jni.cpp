// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <dlfcn.h>
#include <jni.h>
#include <pthread.h>
#include <sys/stat.h>  // For chmod()
#include <unistd.h>    // For access()
#include <unistd.h>

#include <atomic>
#include <memory>
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

struct HandleState {
    std::mutex                         operation_mutex;
    std::mutex                         stop_mutex;
    std::shared_ptr<std::atomic<bool>> stop_flag;
    std::atomic<bool>                  closing{false};
};

static std::unordered_map<void*, std::shared_ptr<HandleState>> g_handleStates;
static std::mutex                                              g_handleStatesMutex;

static std::shared_ptr<HandleState> get_handle_state(void* handle) {
    std::lock_guard<std::mutex> lock(g_handleStatesMutex);
    auto                        it = g_handleStates.find(handle);
    return it == g_handleStates.end() ? nullptr : it->second;
}

static void request_stop(const std::shared_ptr<HandleState>& state) {
    if (!state) return;
    std::lock_guard<std::mutex> lock(state->stop_mutex);
    if (state->stop_flag) state->stop_flag->store(true);
}

using namespace jniutils;
using namespace geniex_android_sdk;

// JNI: create - Initialize LLM with configuration
extern "C" JNIEXPORT jlong JNICALL Java_com_geniex_sdk_jni_Llm_create(
    JNIEnv* env, jobject thiz, jobject llm_create_input_obj) {
    try {
        clear_jni_cstr_pool();
        geniex_LlmCreateInput create_input = extract_llm_create_input(env, llm_create_input_obj);
        geniex_LLM*           handle       = nullptr;
        LOGd("[JNI] create() geniex_llm_create called with:");
        LOGd("  model_path: %s", create_input.model_path ? create_input.model_path : "(null)");
        LOGd("  tokenizer_path: %s", create_input.tokenizer_path ? create_input.tokenizer_path : "(null)");
        LOGd("  plugin_id: %s", create_input.plugin_id ? create_input.plugin_id : "(null)");

        int32_t result = geniex_llm_create(&create_input, &handle);
        clear_jni_cstr_pool();

        if (result != GENIEX_SUCCESS || !handle) {
            LOGe("[JNI] create() failed, error code: %d", result);
            throw_runtime_exception(
                env, "Llm create failed: %s", geniex_get_error_message(static_cast<geniex_ErrorCode>(result)));
            return 0;
        }
        {
            std::lock_guard<std::mutex> lock(g_handleStatesMutex);
            g_handleStates[handle] = std::make_shared<HandleState>();
        }
        LOGd("[JNI] create() geniex_llm_create returned handle=%p", handle);
        return reinterpret_cast<jlong>(handle);

    } catch (const std::exception& e) {
        clear_jni_cstr_pool();
        LOGe("[JNI] create() exception: %s", e.what());
        return 0;
    }
}

// JNI: destroy - Clean up LLM resources
extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_Llm_destroy(JNIEnv*, jobject, jlong handle) {
    LOGd("[JNI] destroy() called, handle=%p", (void*)handle);
    if (handle) {
        void* h     = reinterpret_cast<void*>(handle);
        auto  state = get_handle_state(h);
        if (state) {
            state->closing.store(true);
            request_stop(state);
        }
        std::unique_lock<std::mutex> operation_lock;
        if (state) operation_lock = std::unique_lock<std::mutex>(state->operation_mutex);
        int32_t result = geniex_llm_destroy((geniex_LLM*)handle);
        {
            std::lock_guard<std::mutex> lock(g_handleStatesMutex);
            g_handleStates.erase(h);
        }
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
        void* h     = (void*)handle;
        auto  state = get_handle_state(h);
        if (!state) {
            throw_runtime_exception(env, "LLM generate failed: invalid handle");
            return nullptr;
        }

        std::unique_lock<std::mutex> operation_lock(state->operation_mutex);
        if (state->closing.load()) {
            throw_runtime_exception(env, "LLM generate failed: handle is closing");
            return nullptr;
        }

        // Setup stop flag for streaming control
        std::shared_ptr<std::atomic<bool>> stop_flag;
        if (callback) {
            stop_flag = std::make_shared<std::atomic<bool>>(false);
            std::lock_guard<std::mutex> lock(state->stop_mutex);
            state->stop_flag = stop_flag;
        }

        std::string                           cprompt = jstring2str(env, prompt);
        geniex_GenerationConfig               cfg     = extract_generation_config(env, configObj);
        std::unique_ptr<geniex_SamplerConfig> sampler_config(cfg.sampler_config);

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
                    stop_flag.get(),
                    &cbCtx)) {
                std::lock_guard<std::mutex> lock(state->stop_mutex);
                state->stop_flag.reset();
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
                jni_cb_dispose(env, &cbCtx);
                std::lock_guard<std::mutex> lock(state->stop_mutex);
                if (state->stop_flag == stop_flag) state->stop_flag.reset();
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
                std::lock_guard<std::mutex> lock(state->stop_mutex);
                if (state->stop_flag == stop_flag) state->stop_flag.reset();
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
    request_stop(get_handle_state((void*)handle));
}

// JNI: applyChatTemplate - Format messages with chat template
extern "C" JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Llm_applyChatTemplate(JNIEnv* env, jobject thiz,
    jlong handle, jobjectArray jmessages, jstring jtools, jboolean jEnableThinking, jboolean jAddGenerationPrompt) {
    auto state = get_handle_state((void*)handle);
    if (!state) {
        throw_runtime_exception(env, "LLM applyChatTemplate failed: invalid handle");
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(state->operation_mutex);
    if (state->closing.load()) {
        throw_runtime_exception(env, "LLM applyChatTemplate failed: handle is closing");
        return nullptr;
    }

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
    auto state = get_handle_state((void*)handle);
    if (!state || state->closing.load()) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
    std::lock_guard<std::mutex> lock(state->operation_mutex);
    if (state->closing.load()) return GENIEX_ERROR_COMMON_NOT_INITIALIZED;
    return geniex_llm_reset(reinterpret_cast<geniex_LLM*>(handle));
}
