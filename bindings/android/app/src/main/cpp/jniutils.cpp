// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "jniutils.h"

#include <android/log.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

#include <string>
#include <thread>
#include <vector>
#define TAG "GenieXSdk"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#define MAX_PATH_LEN 512
namespace jniutils {
static thread_local std::vector<std::unique_ptr<char[]>> jni_cstr_pool;

const char* hold_c_str(const std::string& s) {
    auto buf = std::make_unique<char[]>(s.size() + 1);
    memcpy(buf.get(), s.c_str(), s.size() + 1);
    const char* p = buf.get();
    jni_cstr_pool.emplace_back(std::move(buf));
    return p;
}

void clear_jni_cstr_pool() { jni_cstr_pool.clear(); }

std::string jstring2str(JNIEnv* env, jstring jstr) {
    if (!jstr) return {};
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    std::string result(chars ? chars : "");
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}

// Thin wrapper over the SDK's geniex_resolve_device. The SDK owns the
// alias table — we just marshal strings and translate the output into
// the Android-local ResolvedDevice struct so the 8 JNI call sites keep
// their existing shape.
ResolvedDevice resolve_device(const char* plugin_id, const char* model_name, const std::string& raw) {
    ResolvedDevice r;

    geniex_ResolveDeviceInput in{};
    in.plugin_id   = plugin_id;
    in.model_name  = model_name;
    in.mode        = raw.empty() ? nullptr : raw.c_str();
    in.ngl_default = -1;  // sentinel: "no override" unless the SDK forces one

    geniex_ResolveDeviceOutput out{};
    int32_t                    rc = geniex_resolve_device(&in, &out);
    if (rc != GENIEX_SUCCESS) {
        LOGe("[JNI] geniex_resolve_device failed (rc=%d), falling back to empty device", rc);
        return r;
    }

    if (out.device_id) {
        r.device_id = out.device_id;
        geniex_free(out.device_id);
    }
    if (out.warning) {
        r.warning = out.warning;
        LOGi("[JNI] resolve_device: %s", r.warning.c_str());
        geniex_free(out.warning);
    }
    // Preserve the "-1 means no override" contract downstream relies on.
    if (out.ngl != -1) r.ngl_override = out.ngl;
    return r;
}

std::vector<std::string> jstringArray2vec(JNIEnv* env, jobjectArray arr) {
    std::vector<std::string> vec;
    if (!arr) return vec;
    jsize len = env->GetArrayLength(arr);
    vec.reserve(len);
    for (jsize i = 0; i < len; ++i) {
        jstring jstr = (jstring)env->GetObjectArrayElement(arr, i);
        vec.push_back(jstring2str(env, jstr));
        env->DeleteLocalRef(jstr);
    }
    return vec;
}

std::vector<int32_t> jintArray2vec(JNIEnv* env, jintArray arr) {
    jsize                len = env->GetArrayLength(arr);
    std::vector<int32_t> result(len);
    env->GetIntArrayRegion(arr, 0, len, result.data());
    return result;
}

void getStringArrayField(JNIEnv* env, jobject obj, jclass cls, const char* fieldName, std::vector<std::string>& storage,
    std::vector<const char*>& ptrs) {
    jfieldID     fieldId = env->GetFieldID(cls, fieldName, "[Ljava/lang/String;");
    jobjectArray arr     = fieldId ? (jobjectArray)env->GetObjectField(obj, fieldId) : nullptr;
    storage.clear();
    ptrs.clear();
    if (arr) {
        jsize count = env->GetArrayLength(arr);
        for (jsize i = 0; i < count; ++i) {
            jstring jstr = (jstring)env->GetObjectArrayElement(arr, i);
            storage.push_back(jstring2str(env, jstr));
            env->DeleteLocalRef(jstr);
        }
        for (auto& s : storage) ptrs.push_back(s.c_str());
    }
}

geniex_GenerationConfig extract_generation_config(JNIEnv* env, jobject configObj) {
    geniex_GenerationConfig cfg = {};
    if (!configObj) return cfg;
    jclass cls = env->GetObjectClass(configObj);

    // ----------- mas\x token -----------
    cfg.max_tokens = env->GetIntField(configObj, env->GetFieldID(cls, "maxTokens", "I"));

    jfieldID nPastId = env->GetFieldID(cls, "nPast", "I");
    cfg.n_past       = nPastId ? env->GetIntField(configObj, nPastId) : 0;

    // ----------- stopWords  -----------
    static thread_local std::vector<std::string> stopWordStorage;
    static thread_local std::vector<const char*> stopWordPtrs;
    getStringArrayField(env, configObj, cls, "stopWords", stopWordStorage, stopWordPtrs);
    if (!stopWordPtrs.empty()) {
        cfg.stop       = stopWordPtrs.data();
        cfg.stop_count = stopWordPtrs.size();
    } else {
        cfg.stop       = nullptr;
        cfg.stop_count = 0;
    }

    // ----------- sampler config -----------
    jfieldID samplerCfgId  = env->GetFieldID(cls, "samplerConfig", "Lcom/geniex/sdk/bean/SamplerConfig;");
    jobject  samplerCfgObj = samplerCfgId ? env->GetObjectField(configObj, samplerCfgId) : nullptr;
    if (samplerCfgObj) {
        cfg.sampler_config = new geniex_SamplerConfig(extract_sampler_config(env, samplerCfgObj));
    }

    // ----------- imagePaths  -----------
    static thread_local std::vector<std::string> imagePathStorage;
    static thread_local std::vector<const char*> imagePathPtrs;
    getStringArrayField(env, configObj, cls, "imagePaths", imagePathStorage, imagePathPtrs);
    cfg.image_paths = imagePathPtrs.empty() ? nullptr : (geniex_Path*)imagePathPtrs.data();
    cfg.image_count = imagePathPtrs.size();

    // ----------- audioPaths  -----------
    static thread_local std::vector<std::string> audioPathStorage;
    static thread_local std::vector<const char*> audioPathPtrs;
    getStringArrayField(env, configObj, cls, "audioPaths", audioPathStorage, audioPathPtrs);
    cfg.audio_paths = audioPathPtrs.empty() ? nullptr : (geniex_Path*)audioPathPtrs.data();
    cfg.audio_count = audioPathPtrs.size();

    // ----------- sliding window (qcom-ai-hub/geniex#1197) -----------
    jfieldID slidingWindowId = env->GetFieldID(cls, "slidingWindow", "Z");
    cfg.sliding_window       = slidingWindowId ? env->GetBooleanField(configObj, slidingWindowId) : false;

    jfieldID slidingWindowNKeepId = env->GetFieldID(cls, "slidingWindowNKeep", "I");
    cfg.sliding_window_n_keep     = slidingWindowNKeepId ? env->GetIntField(configObj, slidingWindowNKeepId) : 0;

    return cfg;
}

geniex_SamplerConfig extract_sampler_config(JNIEnv* env, jobject configObj) {
    geniex_SamplerConfig cfg = {};
    if (!configObj) return cfg;

    jclass cls = env->GetObjectClass(configObj);

    // Scalars
    cfg.temperature = env->GetFloatField(configObj, env->GetFieldID(cls, "temperature", "F"));
    cfg.top_p       = env->GetFloatField(configObj, env->GetFieldID(cls, "topP", "F"));
    cfg.top_k       = env->GetIntField(configObj, env->GetFieldID(cls, "topK", "I"));

    // NEW: min_p
    cfg.min_p = env->GetFloatField(configObj, env->GetFieldID(cls, "minP", "F"));

    cfg.repetition_penalty = env->GetFloatField(configObj, env->GetFieldID(cls, "repetitionPenalty", "F"));
    cfg.presence_penalty   = env->GetFloatField(configObj, env->GetFieldID(cls, "presencePenalty", "F"));
    cfg.frequency_penalty  = env->GetFloatField(configObj, env->GetFieldID(cls, "frequencyPenalty", "F"));
    cfg.seed               = env->GetIntField(configObj, env->GetFieldID(cls, "seed", "I"));

    // Strings / paths use thread_local buffer to ensure pointer lifetime
    static thread_local std::string grammar_path_storage;
    static thread_local std::string grammar_str_storage;

    // grammarPath -> geniex_Path
    {
        jfieldID fid  = env->GetFieldID(cls, "grammarPath", "Ljava/lang/String;");
        jstring  jstr = (jstring)env->GetObjectField(configObj, fid);
        if (jstr) {
            grammar_path_storage = jstring2str(env, jstr);
            cfg.grammar_path     = grammar_path_storage.c_str();
            env->DeleteLocalRef(jstr);
        } else {
            cfg.grammar_path = nullptr;
        }
    }

    // grammarString -> const char*
    {
        jfieldID fid  = env->GetFieldID(cls, "grammarString", "Ljava/lang/String;");
        jstring  jstr = (jstring)env->GetObjectField(configObj, fid);
        if (jstr) {
            grammar_str_storage = jstring2str(env, jstr);
            cfg.grammar_string  = grammar_str_storage.c_str();
            env->DeleteLocalRef(jstr);
        } else {
            cfg.grammar_string = nullptr;
        }
    }

    // Optional: release the local ref to cls (not required, but cleaner).
    env->DeleteLocalRef(cls);
    return cfg;
}

geniex_ModelConfig extract_model_config(JNIEnv* env, jobject configObj) {
    geniex_ModelConfig config = {};
    if (!configObj) return config;

    jclass cls = env->GetObjectClass(configObj);

    config.n_ctx           = env->GetIntField(configObj, env->GetFieldID(cls, "nCtx", "I"));
    config.n_threads       = env->GetIntField(configObj, env->GetFieldID(cls, "nThreads", "I"));
    config.n_threads_batch = env->GetIntField(configObj, env->GetFieldID(cls, "nThreadsBatch", "I"));
    config.n_batch         = env->GetIntField(configObj, env->GetFieldID(cls, "nBatch", "I"));
    config.n_ubatch        = env->GetIntField(configObj, env->GetFieldID(cls, "nUBatch", "I"));
    config.n_seq_max       = env->GetIntField(configObj, env->GetFieldID(cls, "nSeqMax", "I"));

    config.n_gpu_layers = env->GetIntField(configObj, env->GetFieldID(cls, "nGpuLayers", "I"));

    // chat_template_path
    jfieldID fid              = env->GetFieldID(cls, "chat_template_path", "Ljava/lang/String;");
    jstring  jstr             = (jstring)env->GetObjectField(configObj, fid);
    config.chat_template_path = jstr ? hold_c_str(jstring2str(env, jstr)) : nullptr;

    // chat_template_content
    fid                          = env->GetFieldID(cls, "chat_template_content", "Ljava/lang/String;");
    jstr                         = (jstring)env->GetObjectField(configObj, fid);
    config.chat_template_content = jstr ? hold_c_str(jstring2str(env, jstr)) : nullptr;

    // Note: old fields like system_library_path, backend_library_path, etc. are removed
    // They don't exist in geniex_ModelConfig anymore (see include/ml.h)

    // max_tokens
    fid               = env->GetFieldID(cls, "max_tokens", "I");
    config.max_tokens = env->GetIntField(configObj, fid);

    // enable_thinking
    fid                    = env->GetFieldID(cls, "enable_thinking", "Z");
    config.enable_thinking = env->GetBooleanField(configObj, fid);

    // verbose
    fid            = env->GetFieldID(cls, "verbose", "Z");
    config.verbose = env->GetBooleanField(configObj, fid);

    return config;
}
jobject extract_profiling_data(JNIEnv* env, const geniex_ProfileData& data) {
    jclass cls = env->FindClass("com/geniex/sdk/bean/ProfilingData");
    if (!cls) return nullptr;

    // (DDDJJJDDDLjava/lang/String;)V
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(DDDJJJDDDLjava/lang/String;)V");
    if (!ctor) return nullptr;

    const jdouble ttft_ms   = static_cast<jdouble>(data.ttft / 1000.0);
    const jdouble prompt_ms = static_cast<jdouble>(data.prompt_time / 1000.0);
    const jdouble decode_ms = static_cast<jdouble>(data.decode_time / 1000.0);

    const jlong prompt_tokens = static_cast<jlong>(data.prompt_tokens);
    const jlong gen_tokens    = static_cast<jlong>(data.generated_tokens);
    const jlong audio_ms      = static_cast<jlong>(data.audio_duration / 1000);

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

    LOGe("prefill_speed=%.6f tok/s, decoding_speed=%.6f tok/s, rtf=%.4f", prefill_speed, decoding_speed, rtf);

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

static bool checkAndLogJniException(JNIEnv* env, const char* where) {
    if (env->ExceptionCheck()) {
        LOGe("[JNI] Exception at %s", where);
        env->ExceptionDescribe();
        env->ExceptionClear();
        return true;
    }
    return false;
}

geniex_LlmCreateInput extract_llm_create_input(JNIEnv* env, jobject inputObj) {
    geniex_LlmCreateInput out = {};
    if (!inputObj) {
        LOGe("[JNI] extract_llm_create_input: inputObj is null");
        return out;
    }

    jclass cls = env->GetObjectClass(inputObj);
    if (!cls) {
        LOGe("[JNI] extract_llm_create_input: GetObjectClass returned null");
        return out;
    }

    jfieldID fid;
    jstring  jstr;

    // === model_name (optional — QAIRT plugin reads metadata.json, only
    //     llama_cpp's gpt-oss device-default override consults it) ===
    LOGi("[JNI] [extract] locating field 'model_name' (Ljava/lang/String;)");
    fid = env->GetFieldID(cls, "model_name", "Ljava/lang/String;");
    if (checkAndLogJniException(env, "GetFieldID(model_name)") || !fid) {
        LOGi("[JNI] [extract] field 'model_name' not present (Kotlin property)");
    } else {
        jstr = (jstring)env->GetObjectField(inputObj, fid);
        if (checkAndLogJniException(env, "GetObjectField(model_name)") || !jstr) {
            LOGi("[JNI] [extract] model_name = (null)");
        } else {
            std::string s  = jstring2str(env, jstr);
            out.model_name = hold_c_str(s);
            LOGi("[JNI] [extract] model_name = %s", s.c_str());
            env->DeleteLocalRef(jstr);
        }
    }

    // === model_path ===
    LOGi("[JNI] [extract] locating field 'model_path' (Ljava/lang/String;)");
    fid = env->GetFieldID(cls, "model_path", "Ljava/lang/String;");
    if (checkAndLogJniException(env, "GetFieldID(model_path)") || !fid) {
        LOGe("[JNI] [extract] field 'model_path' not found");
    } else {
        jstr = (jstring)env->GetObjectField(inputObj, fid);
        if (checkAndLogJniException(env, "GetObjectField(model_path)") || !jstr) {
            LOGe("[JNI] [extract] 'model_path' is null");
        } else {
            std::string s  = jstring2str(env, jstr);
            out.model_path = hold_c_str(s);
            LOGi("[JNI] [extract] model_path = %s", s.c_str());
            env->DeleteLocalRef(jstr);
        }
    }

    // === tokenizer_path ===
    LOGi("[JNI] [extract] locating field 'tokenizer_path' (Ljava/lang/String;)");
    fid = env->GetFieldID(cls, "tokenizer_path", "Ljava/lang/String;");
    if (checkAndLogJniException(env, "GetFieldID(tokenizer_path)") || !fid) {
        LOGe("[JNI] [extract] field 'tokenizer_path' not found");
    } else {
        jstr = (jstring)env->GetObjectField(inputObj, fid);
        if (checkAndLogJniException(env, "GetObjectField(tokenizer_path)") || !jstr) {
            LOGi("[JNI] [extract] tokenizer_path = (null)");
        } else {
            std::string s      = jstring2str(env, jstr);
            out.tokenizer_path = hold_c_str(s);
            LOGi("[JNI] [extract] tokenizer_path = %s", s.c_str());
            env->DeleteLocalRef(jstr);
        }
    }

    // === config ===
    LOGi("[JNI] [extract] locating field 'config' (Lcom/geniex/sdk/bean/ModelConfig;)");
    fid = env->GetFieldID(cls, "config", "Lcom/geniex/sdk/bean/ModelConfig;");
    if (checkAndLogJniException(env, "GetFieldID(config)") || !fid) {
        LOGe("[JNI] [extract] field 'config' not found");
    } else {
        jobject configObj = env->GetObjectField(inputObj, fid);
        if (checkAndLogJniException(env, "GetObjectField(config)") || !configObj) {
            LOGe("[JNI] [extract] config is null");
        } else {
            out.config = extract_model_config(env, configObj);
            LOGi("[JNI] [extract] config extracted");
            env->DeleteLocalRef(configObj);
        }
    }

    // === runtime_id ===
    LOGi("[JNI] [extract] locating field 'runtime_id' (Ljava/lang/String;)");
    fid = env->GetFieldID(cls, "runtime_id", "Ljava/lang/String;");
    if (checkAndLogJniException(env, "GetFieldID(runtime_id)") || !fid) {
        LOGe("[JNI] [extract] field 'runtime_id' not found (Kotlin property likely)");
    } else {
        jstr = (jstring)env->GetObjectField(inputObj, fid);
        if (checkAndLogJniException(env, "GetObjectField(runtime_id)") || !jstr) {
            LOGi("[JNI] [extract] runtime_id = (null)");
        } else {
            std::string s = jstring2str(env, jstr);
            out.plugin_id = hold_c_str(s);
            LOGi("[JNI] [extract] runtime_id = %s", s.c_str());
            env->DeleteLocalRef(jstr);
        }
    }

    // === compute_unit ===
    std::string raw_dev;
    fid = env->GetFieldID(cls, "compute_unit", "Ljava/lang/String;");
    if (fid) {
        jstr = (jstring)env->GetObjectField(inputObj, fid);
        if (jstr) {
            raw_dev = jstring2str(env, jstr);
            env->DeleteLocalRef(jstr);
        }
    }
    {
        ResolvedDevice r = resolve_device(out.plugin_id, out.model_name, raw_dev);
        out.device_id    = r.device_id.empty() ? nullptr : hold_c_str(r.device_id);
        if (r.ngl_override >= 0) out.config.n_gpu_layers = r.ngl_override;
        LOGi("[JNI] [extract] compute_unit = %s, n_gpu_layers = %d (from raw='%s')",
            r.device_id.empty() ? "(null)" : r.device_id.c_str(),
            out.config.n_gpu_layers,
            raw_dev.c_str());
    }

    env->DeleteLocalRef(cls);
    return out;
}

geniex_VlmCreateInput extract_vlm_create_input(JNIEnv* env, jobject inputObj) {
    geniex_VlmCreateInput out{};  // zero-initialize all fields to 0 / nullptr
    if (!inputObj) return out;

    jclass cls = env->GetObjectClass(inputObj);
    if (!cls) return out;

    jfieldID fid;
    jstring  jstr;

    // model_name: String
    fid = env->GetFieldID(cls, "model_name", "Ljava/lang/String;");
    if (fid) {
        jstr = (jstring)env->GetObjectField(inputObj, fid);
        if (jstr) {
            std::string s  = jstring2str(env, jstr);
            out.model_name = hold_c_str(s);
            LOGi("[JNI] [extract_vlm] model_name = %s", s.c_str());
            env->DeleteLocalRef(jstr);
        }
    }

    // model_path : String
    fid = env->GetFieldID(cls, "model_path", "Ljava/lang/String;");
    if (fid) {
        jstr           = (jstring)env->GetObjectField(inputObj, fid);
        out.model_path = jstr ? hold_c_str(jstring2str(env, jstr)) : nullptr;
        if (jstr) env->DeleteLocalRef(jstr);
    }

    // mmproj_path : String
    fid = env->GetFieldID(cls, "mmproj_path", "Ljava/lang/String;");
    if (fid) {
        jstr            = (jstring)env->GetObjectField(inputObj, fid);
        out.mmproj_path = jstr ? hold_c_str(jstring2str(env, jstr)) : nullptr;
        if (jstr) env->DeleteLocalRef(jstr);
    }

    // config : Lcom/geniex/sdk/bean/ModelConfig;
    fid = env->GetFieldID(cls, "config", "Lcom/geniex/sdk/bean/ModelConfig;");
    if (fid) {
        jobject configObj = env->GetObjectField(inputObj, fid);
        out.config        = extract_model_config(env, configObj);  // existing helper
        if (configObj) env->DeleteLocalRef(configObj);
    }

    // runtime_id : String
    fid = env->GetFieldID(cls, "runtime_id", "Ljava/lang/String;");
    if (fid) {
        jstr          = (jstring)env->GetObjectField(inputObj, fid);
        out.plugin_id = jstr ? hold_c_str(jstring2str(env, jstr)) : nullptr;
        if (jstr) env->DeleteLocalRef(jstr);
    }

    // compute_unit : String
    {
        std::string raw_dev;
        fid = env->GetFieldID(cls, "compute_unit", "Ljava/lang/String;");
        if (fid) {
            jstr = (jstring)env->GetObjectField(inputObj, fid);
            if (jstr) {
                raw_dev = jstring2str(env, jstr);
                env->DeleteLocalRef(jstr);
            }
        }
        ResolvedDevice r = resolve_device(out.plugin_id, out.model_name, raw_dev);
        out.device_id    = r.device_id.empty() ? nullptr : hold_c_str(r.device_id);
        if (r.ngl_override >= 0) out.config.n_gpu_layers = r.ngl_override;
        LOGi("[JNI] [extract_vlm] compute_unit = %s, n_gpu_layers = %d (from raw='%s')",
            r.device_id.empty() ? "(null)" : r.device_id.c_str(),
            out.config.n_gpu_layers,
            raw_dev.c_str());
    }

    env->DeleteLocalRef(cls);
    return out;
}

std::vector<geniex_LlmChatMessage> extract_llm_chat_messages(
    JNIEnv* env, jobjectArray jmessages, std::vector<std::string>& str_buf) {
    std::vector<geniex_LlmChatMessage> msgs;
    str_buf.clear();
    jsize len = env->GetArrayLength(jmessages);
    msgs.reserve(len);

    for (jsize i = 0; i < len; ++i) {
        jobject jmsg   = env->GetObjectArrayElement(jmessages, i);
        jclass  msgCls = env->GetObjectClass(jmsg);

        jfieldID roleFid    = env->GetFieldID(msgCls, "role", "Ljava/lang/String;");
        jfieldID contentFid = env->GetFieldID(msgCls, "content", "Ljava/lang/String;");

        jstring jrole    = (jstring)env->GetObjectField(jmsg, roleFid);
        jstring jcontent = (jstring)env->GetObjectField(jmsg, contentFid);

        std::string sRole    = jstring2str(env, jrole);
        std::string sContent = jstring2str(env, jcontent);
        str_buf.push_back(sRole);
        str_buf.push_back(sContent);

        geniex_LlmChatMessage m;
        m.role    = str_buf[str_buf.size() - 2].c_str();
        m.content = str_buf[str_buf.size() - 1].c_str();
        msgs.push_back(m);

        env->DeleteLocalRef(jrole);
        env->DeleteLocalRef(jcontent);
        env->DeleteLocalRef(jmsg);
        env->DeleteLocalRef(msgCls);
    }
    return msgs;
}
// typedef struct {
//     const char* type;  // "text", "image", "audio", …
//     const char* text;  // payload: actual text, URL, or token
// } geniex_VlmContent;
//
// typedef struct {
//     const char*    role;           // "user", "assistant", "system", …
//     int64_t        content_count;  // number of elements in `contents`
//     geniex_VlmContent* contents;       // dynamically-allocated array (may be NULL)
// } geniex_VlmChatMessage;

static geniex_VlmContent extract_vlm_content(JNIEnv* env, jobject jcontent) {
    geniex_VlmContent out{};
    if (!jcontent) return out;

    jclass cls = env->GetObjectClass(jcontent);

    // type: String  -> getType()
    jmethodID midGetType = env->GetMethodID(cls, "getType", "()Ljava/lang/String;");
    if (!midGetType) {
        env->ExceptionClear();
    }  // in case there is no getter (obfuscated or named differently)
    if (midGetType) {
        jstring jtype = (jstring)env->CallObjectMethod(jcontent, midGetType);
        if (jtype) {
            out.type = hold_c_str(jstring2str(env, jtype));
            env->DeleteLocalRef(jtype);
        }
    } else {
        // Fallback: read the field directly (for non-data-class or public fields).
        jfieldID typeFid = env->GetFieldID(cls, "type", "Ljava/lang/String;");
        if (typeFid) {
            jstring jtype = (jstring)env->GetObjectField(jcontent, typeFid);
            if (jtype) {
                out.type = hold_c_str(jstring2str(env, jtype));
                env->DeleteLocalRef(jtype);
            }
        } else {
            env->ExceptionClear();
        }
    }

    jmethodID midGetText = env->GetMethodID(cls, "getText", "()Ljava/lang/String;");
    if (!midGetText) {
        env->ExceptionClear();
    }
    if (midGetText) {
        jstring jtext = (jstring)env->CallObjectMethod(jcontent, midGetText);
        if (jtext) {
            out.text = hold_c_str(jstring2str(env, jtext));
            env->DeleteLocalRef(jtext);
        }
    } else {
        jfieldID textFid = env->GetFieldID(cls, "text", "Ljava/lang/String;");
        if (textFid) {
            jstring jtext = (jstring)env->GetObjectField(jcontent, textFid);
            if (jtext) {
                out.text = hold_c_str(jstring2str(env, jtext));
                env->DeleteLocalRef(jtext);
            }
        } else {
            env->ExceptionClear();
        }
    }

    env->DeleteLocalRef(cls);
    return out;
}

// Prerequisites (already declared above):
// static thread_local std::vector<std::unique_ptr<char[]>> jni_cstr_pool;
// static const char* hold_c_str(const std::string& s);
// static inline void clear_jni_cstr_pool();

std::vector<geniex_VlmChatMessage> extract_vlm_chat_messages(JNIEnv* env, jobjectArray jmessages) {
    std::vector<geniex_VlmChatMessage> msgs;

    if (!jmessages) return msgs;

    const jsize len = env->GetArrayLength(jmessages);
    if (len <= 0) return msgs;

    // java.util.List reflection
    jclass    listCls     = env->FindClass("java/util/List");
    jmethodID midListSize = nullptr;
    jmethodID midListGet  = nullptr;
    if (listCls) {
        midListSize = env->GetMethodID(listCls, "size", "()I");
        midListGet  = env->GetMethodID(listCls, "get", "(I)Ljava/lang/Object;");
    }

    msgs.reserve(len);

    for (jsize i = 0; i < len; ++i) {
        jobject jmsg = env->GetObjectArrayElement(jmessages, i);
        if (!jmsg) {
            msgs.push_back(geniex_VlmChatMessage{});
            continue;
        }

        jclass msgCls = env->GetObjectClass(jmsg);

        // role -> getRole()
        const char* role_cstr  = nullptr;
        jmethodID   midGetRole = env->GetMethodID(msgCls, "getRole", "()Ljava/lang/String;");
        if (midGetRole) {
            jstring jrole = (jstring)env->CallObjectMethod(jmsg, midGetRole);
            if (jrole) {
                role_cstr = hold_c_str(jstring2str(env, jrole));
                env->DeleteLocalRef(jrole);
            }
        }

        // contents: List<VlmContent> -> getContents()
        geniex_VlmContent* contents_arr = nullptr;
        int64_t            count        = 0;

        jmethodID midGetContents = env->GetMethodID(msgCls, "getContents", "()Ljava/util/List;");
        if (midGetContents && listCls && midListSize && midListGet) {
            jobject jList = env->CallObjectMethod(jmsg, midGetContents);
            if (jList) {
                const jint cLen = env->CallIntMethod(jList, midListSize);
                if (cLen > 0) {
                    contents_arr = new geniex_VlmContent[cLen];
                    count        = cLen;
                    for (jint ci = 0; ci < cLen; ++ci) {
                        jobject jc = env->CallObjectMethod(jList, midListGet, ci);
                        if (jc) {
                            contents_arr[ci] = extract_vlm_content(env, jc);  // uses the hold_c_str-backed variant
                            env->DeleteLocalRef(jc);
                        } else {
                            contents_arr[ci] = geniex_VlmContent{};
                        }
                    }
                }
                env->DeleteLocalRef(jList);
            }
        }

        geniex_VlmChatMessage m{};
        m.role          = role_cstr;
        m.content_count = count;
        m.contents      = contents_arr;

        msgs.push_back(m);

        env->DeleteLocalRef(msgCls);
        env->DeleteLocalRef(jmsg);
    }

    if (listCls) env->DeleteLocalRef(listCls);
    return msgs;
}

void free_vlm_chat_messages(std::vector<geniex_VlmChatMessage>& msgs) {
    for (auto& m : msgs) {
        delete[] m.contents;  // deleting nullptr is safe
        m.contents      = nullptr;
        m.content_count = 0;
    }
}

// Extract image and audio paths from VlmChatMessage contents
void extract_media_paths_from_messages(
    JNIEnv* env, jobjectArray jmessages, std::vector<std::string>& image_paths, std::vector<std::string>& audio_paths) {
    image_paths.clear();
    audio_paths.clear();

    if (!jmessages) return;

    const jsize len = env->GetArrayLength(jmessages);
    if (len <= 0) return;

    // Get List class methods
    jclass    listCls     = env->FindClass("java/util/List");
    jmethodID midListSize = nullptr;
    jmethodID midListGet  = nullptr;
    if (listCls) {
        midListSize = env->GetMethodID(listCls, "size", "()I");
        midListGet  = env->GetMethodID(listCls, "get", "(I)Ljava/lang/Object;");
    }

    for (jsize i = 0; i < len; ++i) {
        jobject jmsg = env->GetObjectArrayElement(jmessages, i);
        if (!jmsg) continue;

        jclass msgCls = env->GetObjectClass(jmsg);

        // Get contents: List<VlmContent> -> getContents()
        jmethodID midGetContents = env->GetMethodID(msgCls, "getContents", "()Ljava/util/List;");
        if (midGetContents && listCls && midListSize && midListGet) {
            jobject jList = env->CallObjectMethod(jmsg, midGetContents);
            if (jList) {
                const jint cLen = env->CallIntMethod(jList, midListSize);
                for (jint ci = 0; ci < cLen; ++ci) {
                    jobject jContent = env->CallObjectMethod(jList, midListGet, ci);
                    if (!jContent) continue;

                    jclass contentCls = env->GetObjectClass(jContent);

                    // Get type
                    jmethodID midGetType = env->GetMethodID(contentCls, "getType", "()Ljava/lang/String;");
                    jstring   jType      = nullptr;
                    if (midGetType) {
                        jType = (jstring)env->CallObjectMethod(jContent, midGetType);
                    }

                    // Get text (the path for image/audio)
                    jmethodID midGetText = env->GetMethodID(contentCls, "getText", "()Ljava/lang/String;");
                    jstring   jText      = nullptr;
                    if (midGetText) {
                        jText = (jstring)env->CallObjectMethod(jContent, midGetText);
                    }

                    if (jType && jText) {
                        std::string type = jstring2str(env, jType);
                        std::string text = jstring2str(env, jText);

                        if (type == "image") {
                            image_paths.push_back(text);
                            LOGi("[extract_media_paths] Found image: %s", text.c_str());
                        } else if (type == "audio") {
                            audio_paths.push_back(text);
                            LOGi("[extract_media_paths] Found audio: %s", text.c_str());
                        }
                    }

                    if (jType) env->DeleteLocalRef(jType);
                    if (jText) env->DeleteLocalRef(jText);
                    env->DeleteLocalRef(contentCls);
                    env->DeleteLocalRef(jContent);
                }
                env->DeleteLocalRef(jList);
            }
        }

        env->DeleteLocalRef(msgCls);
        env->DeleteLocalRef(jmsg);
    }

    if (listCls) env->DeleteLocalRef(listCls);

    LOGi("[extract_media_paths] Total images: %zu, Total audios: %zu", image_paths.size(), audio_paths.size());
}

// Function to redirect pipe output to Android logcat
void redirect_output_to_logcat(const char* tag, int pipe_fd) {
    char    buffer[4096];  // Larger buffer for QNN debug logs
    ssize_t bytes_read;

    while ((bytes_read = read(pipe_fd, buffer, sizeof(buffer) - 1)) > 0) {
        buffer[bytes_read] = '\0';

        // Process line by line to ensure complete log messages
        char* line_start = buffer;
        char* line_end   = nullptr;

        while ((line_end = strchr(line_start, '\n')) != nullptr) {
            *line_end = '\0';  // Null-terminate the line

            if (strlen(line_start) > 0) {
                if (strcmp(tag, "STDERR") == 0) {
                    __android_log_print(ANDROID_LOG_ERROR, TAG, "[%s] %s", tag, line_start);
                } else {
                    __android_log_print(ANDROID_LOG_INFO, TAG, "[%s] %s", tag, line_start);
                }
            }

            line_start = line_end + 1;
        }

        // Handle remaining text without newline
        if (strlen(line_start) > 0) {
            if (strcmp(tag, "STDERR") == 0) {
                __android_log_print(ANDROID_LOG_ERROR, TAG, "[%s] %s", tag, line_start);
            } else {
                __android_log_print(ANDROID_LOG_INFO, TAG, "[%s] %s", tag, line_start);
            }
        }
    }

    close(pipe_fd);
}

void setup_redirect_stdout_stderr() {
    int stdout_pipe[2];
    int stderr_pipe[2];

    if (pipe(stdout_pipe) == -1) {
        LOGe("Failed to create stdout pipe");
        return;
    }

    if (pipe(stderr_pipe) == -1) {
        LOGe("Failed to create stderr pipe");
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        return;
    }

// Increase pipe buffer size for QNN debug logs (optional, but helpful)
#ifdef F_SETPIPE_SZ
    fcntl(stdout_pipe[0], F_SETPIPE_SZ, 1048576);  // 1MB
    fcntl(stderr_pipe[0], F_SETPIPE_SZ, 1048576);  // 1MB
#endif

    // Redirect stdout
    if (dup2(stdout_pipe[1], STDOUT_FILENO) == -1) {
        LOGe("Failed to redirect stdout");
    } else {
        close(stdout_pipe[1]);
        std::thread(redirect_output_to_logcat, "STDOUT", stdout_pipe[0]).detach();
    }

    // Redirect stderr
    if (dup2(stderr_pipe[1], STDERR_FILENO) == -1) {
        LOGe("Failed to redirect stderr");
    } else {
        close(stderr_pipe[1]);
        std::thread(redirect_output_to_logcat, "STDERR", stderr_pipe[0]).detach();
    }

    LOGi("stdout/stderr redirection to logcat initialized");
}
}  // namespace jniutils
