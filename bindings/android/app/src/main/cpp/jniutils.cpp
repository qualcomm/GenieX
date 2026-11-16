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

#include "power_mode_alias.h"
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

void free_chat_template_grammar(geniex_ChatTemplateGrammar& grammar) {
    geniex_free(grammar.grammar);
    geniex_free(grammar.generation_prompt);
    if (grammar.triggers) {
        for (int32_t i = 0; i < grammar.trigger_count; ++i) {
            geniex_free(grammar.triggers[i].value);
        }
        geniex_free(grammar.triggers);
    }
    geniex_free(grammar.preserved_tokens);
    grammar = {};
}

jobject chat_template_grammar_to_java(JNIEnv* env, const geniex_ChatTemplateGrammar& grammar) {
    if (grammar.grammar_type == GENIEX_GRAMMAR_TYPE_NONE) return nullptr;
    if (grammar.trigger_count < 0 || grammar.preserved_token_count < 0 ||
        (grammar.trigger_count > 0 && !grammar.triggers) ||
        (grammar.preserved_token_count > 0 && !grammar.preserved_tokens)) {
        return nullptr;
    }

    jclass       trigger_class     = nullptr;
    jobjectArray triggers          = nullptr;
    jintArray    preserved_tokens  = nullptr;
    jstring      grammar_text      = nullptr;
    jstring      generation_prompt = nullptr;
    jclass       grammar_class     = nullptr;
    auto         cleanup           = [&]() {
        if (grammar_text) env->DeleteLocalRef(grammar_text);
        if (generation_prompt) env->DeleteLocalRef(generation_prompt);
        if (grammar_class) env->DeleteLocalRef(grammar_class);
        if (preserved_tokens) env->DeleteLocalRef(preserved_tokens);
        if (triggers) env->DeleteLocalRef(triggers);
        if (trigger_class) env->DeleteLocalRef(trigger_class);
    };
    auto fail = [&]() -> jobject {
        cleanup();
        return nullptr;
    };

    trigger_class = env->FindClass("com/geniex/sdk/bean/GrammarTrigger");
    if (!trigger_class || env->ExceptionCheck()) return fail();
    jmethodID trigger_ctor = env->GetMethodID(trigger_class, "<init>", "(ILjava/lang/String;I)V");
    if (!trigger_ctor || env->ExceptionCheck()) return fail();

    triggers = env->NewObjectArray(grammar.trigger_count, trigger_class, nullptr);
    if (!triggers || env->ExceptionCheck()) return fail();
    for (int32_t i = 0; i < grammar.trigger_count; ++i) {
        const auto& trigger = grammar.triggers[i];
        jstring     value   = trigger.value ? env->NewStringUTF(trigger.value) : nullptr;
        if ((trigger.value && !value) || env->ExceptionCheck()) {
            if (value) env->DeleteLocalRef(value);
            return fail();
        }
        jobject item = env->NewObject(trigger_class,
            trigger_ctor,
            static_cast<jint>(trigger.trigger_type),
            value,
            static_cast<jint>(trigger.token));
        if (value) env->DeleteLocalRef(value);
        if (!item || env->ExceptionCheck()) {
            if (item) env->DeleteLocalRef(item);
            return fail();
        }
        env->SetObjectArrayElement(triggers, i, item);
        env->DeleteLocalRef(item);
        if (env->ExceptionCheck()) return fail();
    }

    preserved_tokens = env->NewIntArray(grammar.preserved_token_count);
    if (!preserved_tokens || env->ExceptionCheck()) return fail();
    if (grammar.preserved_token_count > 0) {
        env->SetIntArrayRegion(preserved_tokens,
            0,
            grammar.preserved_token_count,
            reinterpret_cast<const jint*>(grammar.preserved_tokens));
        if (env->ExceptionCheck()) return fail();
    }

    if (grammar.grammar) {
        grammar_text = env->NewStringUTF(grammar.grammar);
        if (!grammar_text || env->ExceptionCheck()) return fail();
    }
    if (grammar.generation_prompt) {
        generation_prompt = env->NewStringUTF(grammar.generation_prompt);
        if (!generation_prompt || env->ExceptionCheck()) return fail();
    }
    grammar_class = env->FindClass("com/geniex/sdk/bean/ChatTemplateGrammar");
    if (!grammar_class || env->ExceptionCheck()) return fail();
    jmethodID grammar_ctor = env->GetMethodID(
        grammar_class, "<init>", "(ILjava/lang/String;ZLjava/lang/String;[Lcom/geniex/sdk/bean/GrammarTrigger;[I)V");
    if (!grammar_ctor || env->ExceptionCheck()) return fail();
    jobject result = env->NewObject(grammar_class,
        grammar_ctor,
        static_cast<jint>(grammar.grammar_type),
        grammar_text,
        static_cast<jboolean>(grammar.lazy),
        generation_prompt,
        triggers,
        preserved_tokens);
    if (!result || env->ExceptionCheck()) {
        if (result) env->DeleteLocalRef(result);
        return fail();
    }

    cleanup();
    return result;
}

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
// their existing shape. `ngl_default` is the caller's n_gpu_layers, which
// the SDK passes through for gpu/npu/hybrid (-1 = all layers) and forces
// to 0 for cpu/qairt.
ResolvedDevice resolve_device(
    const char* plugin_id, const char* model_name, const std::string& raw, int32_t ngl_default) {
    ResolvedDevice r;

    geniex_ResolveDeviceInput in{};
    in.plugin_id   = plugin_id;
    in.model_name  = model_name;
    in.mode        = raw.empty() ? nullptr : raw.c_str();
    in.ngl_default = ngl_default;

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
    r.ngl = out.ngl;
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

std::vector<float> jfloatArray2vec(JNIEnv* env, jfloatArray arr) {
    jsize              len = env->GetArrayLength(arr);
    std::vector<float> result(len);
    env->GetFloatArrayRegion(arr, 0, len, result.data());
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

static void extract_sampler_config(
    JNIEnv* env, jobject configObj, geniex_SamplerConfig& cfg, GenerationConfigStorage& storage);

GenerationConfigStorage extract_generation_config(JNIEnv* env, jobject configObj) {
    GenerationConfigStorage storage;
    auto&                   cfg = storage.value;
    if (!configObj) return storage;
    jclass cls = env->GetObjectClass(configObj);

    // ----------- max token -----------
    cfg.max_tokens = env->GetIntField(configObj, env->GetFieldID(cls, "maxTokens", "I"));

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
        storage.has_sampler = true;
        extract_sampler_config(env, samplerCfgObj, storage.sampler, storage);
        env->DeleteLocalRef(samplerCfgObj);
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

    return storage;
}

static void extract_sampler_config(
    JNIEnv* env, jobject configObj, geniex_SamplerConfig& cfg, GenerationConfigStorage& storage) {
    cfg = {};
    if (!configObj) return;

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

    // chatTemplateGrammar borrows buffers owned by GenerationConfigStorage.
    {
        jfieldID fid         = env->GetFieldID(cls, "chatTemplateGrammar", "Lcom/geniex/sdk/bean/ChatTemplateGrammar;");
        jobject  grammar_obj = fid ? env->GetObjectField(configObj, fid) : nullptr;
        if (grammar_obj) {
            jclass grammar_cls   = env->GetObjectClass(grammar_obj);
            auto&  grammar       = cfg.chat_template_grammar;
            grammar.grammar_type = static_cast<geniex_GrammarType>(
                env->GetIntField(grammar_obj, env->GetFieldID(grammar_cls, "grammarType", "I")));
            grammar.lazy = env->GetBooleanField(grammar_obj, env->GetFieldID(grammar_cls, "lazy", "Z"));

            auto copy_string = [&](const char* name, char** dst) {
                jfieldID string_id = env->GetFieldID(grammar_cls, name, "Ljava/lang/String;");
                jstring  string =
                    string_id ? static_cast<jstring>(env->GetObjectField(grammar_obj, string_id)) : nullptr;
                if (string) {
                    storage.grammar_strings.push_back(jstring2str(env, string));
                    *dst = storage.grammar_strings.back().data();
                    env->DeleteLocalRef(string);
                }
            };
            copy_string("grammar", &grammar.grammar);
            copy_string("generationPrompt", &grammar.generation_prompt);

            jfieldID triggers_id = env->GetFieldID(grammar_cls, "triggers", "[Lcom/geniex/sdk/bean/GrammarTrigger;");
            jobjectArray triggers =
                triggers_id ? static_cast<jobjectArray>(env->GetObjectField(grammar_obj, triggers_id)) : nullptr;
            if (triggers) {
                jsize count = env->GetArrayLength(triggers);
                storage.grammar_triggers.resize(count);
                jclass   trigger_cls = env->FindClass("com/geniex/sdk/bean/GrammarTrigger");
                jfieldID type_id     = env->GetFieldID(trigger_cls, "triggerType", "I");
                jfieldID value_id    = env->GetFieldID(trigger_cls, "value", "Ljava/lang/String;");
                jfieldID token_id    = env->GetFieldID(trigger_cls, "token", "I");
                for (jsize i = 0; i < count; ++i) {
                    jobject item        = env->GetObjectArrayElement(triggers, i);
                    auto&   target      = storage.grammar_triggers[i];
                    target.trigger_type = static_cast<geniex_GrammarTriggerType>(env->GetIntField(item, type_id));
                    target.token        = env->GetIntField(item, token_id);
                    jstring value       = static_cast<jstring>(env->GetObjectField(item, value_id));
                    if (value) {
                        storage.grammar_strings.push_back(jstring2str(env, value));
                        target.value = storage.grammar_strings.back().data();
                        env->DeleteLocalRef(value);
                    }
                    env->DeleteLocalRef(item);
                }
                grammar.triggers      = storage.grammar_triggers.empty() ? nullptr : storage.grammar_triggers.data();
                grammar.trigger_count = count;
                env->DeleteLocalRef(trigger_cls);
                env->DeleteLocalRef(triggers);
            }

            jfieldID  preserved_id = env->GetFieldID(grammar_cls, "preservedTokens", "[I");
            jintArray preserved =
                preserved_id ? static_cast<jintArray>(env->GetObjectField(grammar_obj, preserved_id)) : nullptr;
            if (preserved) {
                jsize count = env->GetArrayLength(preserved);
                storage.grammar_preserved_tokens.resize(count);
                if (count > 0) {
                    env->GetIntArrayRegion(
                        preserved, 0, count, reinterpret_cast<jint*>(storage.grammar_preserved_tokens.data()));
                }
                grammar.preserved_tokens =
                    storage.grammar_preserved_tokens.empty() ? nullptr : storage.grammar_preserved_tokens.data();
                grammar.preserved_token_count = count;
                env->DeleteLocalRef(preserved);
            }
            env->DeleteLocalRef(grammar_cls);
            env->DeleteLocalRef(grammar_obj);
        }
    }

    env->DeleteLocalRef(cls);
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

    // spec_type
    fid              = env->GetFieldID(cls, "spec_type", "Ljava/lang/String;");
    jstr             = (jstring)env->GetObjectField(configObj, fid);
    config.spec_type = jstr ? hold_c_str(jstring2str(env, jstr)) : nullptr;

    // spec_draft_model
    fid                     = env->GetFieldID(cls, "spec_draft_model", "Ljava/lang/String;");
    jstr                    = (jstring)env->GetObjectField(configObj, fid);
    config.spec_draft_model = jstr ? hold_c_str(jstring2str(env, jstr)) : nullptr;

    // spec_n_max
    fid               = env->GetFieldID(cls, "spec_n_max", "I");
    config.spec_n_max = env->GetIntField(configObj, fid);

    // spec_n_min
    fid               = env->GetFieldID(cls, "spec_n_min", "I");
    config.spec_n_min = env->GetIntField(configObj, fid);

    // spec_p_min
    fid               = env->GetFieldID(cls, "spec_p_min", "F");
    config.spec_p_min = env->GetFloatField(configObj, fid);

    // power_mode
    fid  = env->GetFieldID(cls, "power_mode", "Ljava/lang/String;");
    jstr = (jstring)env->GetObjectField(configObj, fid);
    {
        const std::string power_mode_str = jstr ? jstring2str(env, jstr) : "";
        if (!geniex_power_mode_from_alias(power_mode_str.c_str(), &config.power_mode)) {
            LOGe("invalid power_mode '%s', falling back to burst", power_mode_str.c_str());
            config.power_mode = GENIEX_POWER_MODE_BURST;
        }
    }

    return config;
}
jobject extract_profiling_data(JNIEnv* env, const geniex_ProfileData& data) {
    jclass cls = env->FindClass("com/geniex/sdk/bean/ProfilingData");
    if (!cls) return nullptr;

    // (DDDDJJJDDJJLjava/lang/String;)V
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(DDDDJJJDDJJLjava/lang/String;)V");
    if (!ctor) return nullptr;

    const jdouble ttft_ms   = static_cast<jdouble>(data.ttft / 1000.0);
    const jdouble media_ms  = static_cast<jdouble>(data.media_time / 1000.0);
    const jdouble prompt_ms = static_cast<jdouble>(data.prompt_time / 1000.0);
    const jdouble decode_ms = static_cast<jdouble>(data.decode_time / 1000.0);

    const jlong prompt_tokens = static_cast<jlong>(data.prompt_tokens);
    const jlong total_tokens  = static_cast<jlong>(data.total_prompt_tokens);
    const jlong gen_tokens    = static_cast<jlong>(data.generated_tokens);

    // ---- compute speeds locally (tokens/sec) ----
    auto tok_per_s = [](int64_t tokens, int64_t time_us) -> jdouble {
        if (time_us <= 0) return 0.0;
        return static_cast<jdouble>(tokens) * 1e6 / static_cast<jdouble>(time_us);
    };

    const jdouble prefill_speed  = tok_per_s(data.prompt_tokens, data.prompt_time);
    const jdouble decoding_speed = tok_per_s(data.generated_tokens, data.decode_time);

    LOGe("prefill_speed=%.6f tok/s, decoding_speed=%.6f tok/s", prefill_speed, decoding_speed);

    const jlong draft_n_total    = static_cast<jlong>(data.draft_n_total);
    const jlong draft_n_accepted = static_cast<jlong>(data.draft_n_accepted);

    jstring jStopReason = env->NewStringUTF(data.stop_reason ? data.stop_reason : "");

    jobject obj = env->NewObject(cls,
        ctor,
        ttft_ms,
        media_ms,
        prompt_ms,
        decode_ms,  // D D D D
        prompt_tokens,
        total_tokens,
        gen_tokens,  // J J J
        prefill_speed,
        decoding_speed,  // D D
        draft_n_total,
        draft_n_accepted,  // J J
        jStopReason        // String
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
        ResolvedDevice r        = resolve_device(out.plugin_id, nullptr, raw_dev, out.config.n_gpu_layers);
        out.device_id           = r.device_id.empty() ? nullptr : hold_c_str(r.device_id);
        out.config.n_gpu_layers = r.ngl;
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

    // vit_device_id : String
    fid = env->GetFieldID(cls, "vit_device_id", "Ljava/lang/String;");
    if (fid) {
        jstr              = (jstring)env->GetObjectField(inputObj, fid);
        out.vit_device_id = jstr ? hold_c_str(jstring2str(env, jstr)) : nullptr;
        if (jstr) env->DeleteLocalRef(jstr);
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
        ResolvedDevice r        = resolve_device(out.plugin_id, nullptr, raw_dev, out.config.n_gpu_layers);
        out.device_id           = r.device_id.empty() ? nullptr : hold_c_str(r.device_id);
        out.config.n_gpu_layers = r.ngl;
        LOGi("[JNI] [extract_vlm] compute_unit = %s, n_gpu_layers = %d (from raw='%s')",
            r.device_id.empty() ? "(null)" : r.device_id.c_str(),
            out.config.n_gpu_layers,
            raw_dev.c_str());
    }

    env->DeleteLocalRef(cls);
    return out;
}

// Reads an optional String field, returning nullptr when absent or null.
static const char* opt_string_field(JNIEnv* env, jobject obj, jclass cls, const char* name) {
    jfieldID fid = env->GetFieldID(cls, name, "Ljava/lang/String;");
    if (!fid) {
        env->ExceptionClear();
        return nullptr;
    }
    jstring jval = (jstring)env->GetObjectField(obj, fid);
    if (!jval) return nullptr;
    const char* held = hold_c_str(jstring2str(env, jval));
    env->DeleteLocalRef(jval);
    return held;
}

// Reads a `List<ToolCall>` field into a fresh array. Returns nullptr with
// count 0 when the field is absent or the list is empty; the caller owns the
// array and frees it with free_*_chat_messages.
static geniex_ToolCall* extract_tool_calls(
    JNIEnv* env, jobject jmsg, jclass msgCls, const char* field, int32_t* out_count) {
    *out_count = 0;

    jfieldID fid = env->GetFieldID(msgCls, field, "Ljava/util/List;");
    if (!fid) {
        env->ExceptionClear();
        return nullptr;
    }
    jobject jList = env->GetObjectField(jmsg, fid);
    if (!jList) return nullptr;

    jclass    listCls = env->FindClass("java/util/List");
    jmethodID midSize = listCls ? env->GetMethodID(listCls, "size", "()I") : nullptr;
    jmethodID midGet  = listCls ? env->GetMethodID(listCls, "get", "(I)Ljava/lang/Object;") : nullptr;
    if (!midSize || !midGet) {
        env->ExceptionClear();
        if (listCls) env->DeleteLocalRef(listCls);
        env->DeleteLocalRef(jList);
        return nullptr;
    }

    const jint n = env->CallIntMethod(jList, midSize);
    if (n <= 0) {
        env->DeleteLocalRef(listCls);
        env->DeleteLocalRef(jList);
        return nullptr;
    }

    auto* calls = new geniex_ToolCall[n]();
    for (jint i = 0; i < n; ++i) {
        jobject jtc = env->CallObjectMethod(jList, midGet, i);
        if (!jtc) continue;
        jclass tcCls       = env->GetObjectClass(jtc);
        calls[i].id        = opt_string_field(env, jtc, tcCls, "id");
        calls[i].name      = opt_string_field(env, jtc, tcCls, "name");
        calls[i].arguments = opt_string_field(env, jtc, tcCls, "arguments");
        env->DeleteLocalRef(tcCls);
        env->DeleteLocalRef(jtc);
    }
    *out_count = static_cast<int32_t>(n);

    env->DeleteLocalRef(listCls);
    env->DeleteLocalRef(jList);
    return calls;
}

std::vector<geniex_LlmChatMessage> extract_llm_chat_messages(JNIEnv* env, jobjectArray jmessages) {
    std::vector<geniex_LlmChatMessage> msgs;
    jsize                              len = env->GetArrayLength(jmessages);
    msgs.reserve(len);

    for (jsize i = 0; i < len; ++i) {
        jobject jmsg   = env->GetObjectArrayElement(jmessages, i);
        jclass  msgCls = env->GetObjectClass(jmsg);

        geniex_LlmChatMessage m{};
        m.role         = opt_string_field(env, jmsg, msgCls, "role");
        m.content      = opt_string_field(env, jmsg, msgCls, "content");
        m.tool_call_id = opt_string_field(env, jmsg, msgCls, "toolCallId");
        m.tool_name    = opt_string_field(env, jmsg, msgCls, "toolName");
        m.tool_calls   = extract_tool_calls(env, jmsg, msgCls, "toolCalls", &m.tool_call_count);
        msgs.push_back(m);

        env->DeleteLocalRef(jmsg);
        env->DeleteLocalRef(msgCls);
    }
    return msgs;
}

void free_llm_chat_messages(std::vector<geniex_LlmChatMessage>& msgs) {
    for (auto& m : msgs) {
        delete[] m.tool_calls;  // deleting nullptr is safe
        m.tool_calls      = nullptr;
        m.tool_call_count = 0;
    }
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
        m.tool_call_id  = opt_string_field(env, jmsg, msgCls, "toolCallId");
        m.tool_name     = opt_string_field(env, jmsg, msgCls, "toolName");
        m.tool_calls    = extract_tool_calls(env, jmsg, msgCls, "toolCalls", &m.tool_call_count);

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
        delete[] m.tool_calls;
        m.tool_calls      = nullptr;
        m.tool_call_count = 0;
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
