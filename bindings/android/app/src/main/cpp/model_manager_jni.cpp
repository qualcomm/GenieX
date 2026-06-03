#include <jni.h>

#include <atomic>
#include <cstring>
#include <string>
#include <vector>

#include "android_utils.h"
#include "geniex.h"
#include "geniex_model.h"
#include "jniutils.h"

using namespace jniutils;
using namespace geniex_android_sdk;

namespace {

// Read a nullable String field from a Kotlin data class by name.
std::string read_opt_string(JNIEnv* env, jclass cls, jobject obj, const char* name) {
    jfieldID fid = env->GetFieldID(cls, name, "Ljava/lang/String;");
    if (!fid) {
        env->ExceptionClear();
        return {};
    }
    auto jstr = static_cast<jstring>(env->GetObjectField(obj, fid));
    if (!jstr) return {};
    std::string s = jstring2str(env, jstr);
    env->DeleteLocalRef(jstr);
    return s;
}

// Pull the HubSource value out of a Kotlin enum instance (the `value` field).
int read_hub_source_value(JNIEnv* env, jclass cls, jobject obj) {
    jfieldID hubFid = env->GetFieldID(cls, "hub", "Lcom/geniex/sdk/bean/HubSource;");
    if (!hubFid) return 0;
    jobject hubObj = env->GetObjectField(obj, hubFid);
    if (!hubObj) return 0;
    jclass    hubCls      = env->GetObjectClass(hubObj);
    jmethodID getValueMid = env->GetMethodID(hubCls, "getValue", "()I");
    int       v           = getValueMid ? env->CallIntMethod(hubObj, getValueMid) : 0;
    env->DeleteLocalRef(hubCls);
    env->DeleteLocalRef(hubObj);
    return v;
}

// Pull the optional ModelType value out of a Kotlin instance (the `value`
// field of the enum). Returns GENIEX_MODEL_TYPE_AUTO (-1) when the field is
// null or missing.
int read_model_type_value(JNIEnv* env, jclass cls, jobject obj) {
    jfieldID typeFid = env->GetFieldID(cls, "model_type", "Lcom/geniex/sdk/bean/ModelType;");
    if (!typeFid) {
        env->ExceptionClear();
        return GENIEX_MODEL_TYPE_AUTO;
    }
    jobject typeObj = env->GetObjectField(obj, typeFid);
    if (!typeObj) return GENIEX_MODEL_TYPE_AUTO;
    jclass    typeCls     = env->GetObjectClass(typeObj);
    jmethodID getValueMid = env->GetMethodID(typeCls, "getValue", "()I");
    int       v           = getValueMid ? env->CallIntMethod(typeObj, getValueMid) : GENIEX_MODEL_TYPE_AUTO;
    env->DeleteLocalRef(typeCls);
    env->DeleteLocalRef(typeObj);
    return v;
}

// Context passed through `geniex_download_progress_cb.user_data`.
struct ProgressCtx {
    JavaVM*           vm;
    jobject           cb_global;         // global ref to DownloadProgressCallback
    jmethodID         onProgress_mid;    // ([Lcom/geniex/sdk/bean/FileProgress;)Z
    jclass            fileProgressCls;   // global ref
    jmethodID         fileProgressCtor;  // (Ljava/lang/String;JJ)V
    std::atomic<bool> cancelled{false};
};

JNIEnv* attach_if_needed(JavaVM* vm, bool& attached) {
    attached    = false;
    JNIEnv* env = nullptr;
    if (!vm) return nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK || !env) {
        if (vm->AttachCurrentThread(&env, nullptr) != 0) return nullptr;
        attached = true;
    }
    return env;
}

// FFI trampoline — called from tokio worker threads. Must be C linkage.
// Thread model: we attach on first call and detach before returning so a
// tokio worker's long life doesn't keep the attachment indefinitely
// (attaching + never detaching aborts the process when the thread exits).
bool progress_trampoline(const geniex_FileProgress* files, int32_t file_count, void* user_data) {
    auto* ctx = static_cast<ProgressCtx*>(user_data);
    if (!ctx || !ctx->cb_global || !ctx->onProgress_mid) return true;
    if (ctx->cancelled.load()) return false;

    bool    attached = false;
    JNIEnv* env      = attach_if_needed(ctx->vm, attached);
    if (!env) return true;  // can't reach the JVM; don't cancel

    jobjectArray arr = env->NewObjectArray(file_count, ctx->fileProgressCls, nullptr);
    if (!arr) {
        env->ExceptionClear();
        if (attached) ctx->vm->DetachCurrentThread();
        return true;
    }

    for (int32_t i = 0; i < file_count; ++i) {
        jstring jName = env->NewStringUTF(files[i].file_name ? files[i].file_name : "");
        jobject item  = env->NewObject(ctx->fileProgressCls,
            ctx->fileProgressCtor,
            jName,
            static_cast<jlong>(files[i].downloaded_bytes),
            static_cast<jlong>(files[i].total_bytes));
        env->SetObjectArrayElement(arr, i, item);
        env->DeleteLocalRef(item);
        env->DeleteLocalRef(jName);
    }

    jboolean keep_going = env->CallBooleanMethod(ctx->cb_global, ctx->onProgress_mid, arr);
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
        keep_going = JNI_FALSE;
    }

    env->DeleteLocalRef(arr);
    if (attached) ctx->vm->DetachCurrentThread();

    bool go = keep_going == JNI_TRUE;
    if (!go) ctx->cancelled.store(true);
    return go;
}

// Build a com.geniex.sdk.bean.ModelPaths from a geniex_ModelPaths,
// returning a Java local ref (or null on error).
jobject build_model_paths(JNIEnv* env, const geniex_ModelPaths& paths) {
    jclass cls = env->FindClass("com/geniex/sdk/bean/ModelPaths");
    if (!cls) return nullptr;
    jclass    modelTypeCls = env->FindClass("com/geniex/sdk/bean/ModelType");
    jmethodID fromValueMid = env->GetStaticMethodID(modelTypeCls, "fromValue", "(I)Lcom/geniex/sdk/bean/ModelType;");
    jmethodID ctor         = env->GetMethodID(cls,
        "<init>",
        "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;"
                "Lcom/geniex/sdk/bean/ModelType;Ljava/lang/String;Ljava/lang/String;)V");
    if (!ctor) {
        env->DeleteLocalRef(cls);
        env->DeleteLocalRef(modelTypeCls);
        return nullptr;
    }
    auto    maybe      = [&](const char* s) -> jstring { return s ? env->NewStringUTF(s) : nullptr; };
    jstring jModelPath = maybe(paths.model_path);
    jstring jModelDir  = maybe(paths.model_dir);
    jstring jModelName = maybe(paths.model_name);
    jstring jPluginId  = maybe(paths.plugin_id);
    jobject jModelType = env->CallStaticObjectMethod(modelTypeCls, fromValueMid, static_cast<jint>(paths.model_type));
    jstring jMmproj    = maybe(paths.mmproj_path);
    jstring jTokenizer = maybe(paths.tokenizer_path);
    // Constructor order matches ModelPaths.kt:
    //   model_path, model_dir, model_name, plugin_id, model_type, mmproj_path?, tokenizer_path?
    jobject obj =
        env->NewObject(cls, ctor, jModelPath, jModelDir, jModelName, jPluginId, jModelType, jMmproj, jTokenizer);
    if (jModelPath) env->DeleteLocalRef(jModelPath);
    if (jModelDir) env->DeleteLocalRef(jModelDir);
    if (jModelName) env->DeleteLocalRef(jModelName);
    if (jPluginId) env->DeleteLocalRef(jPluginId);
    if (jModelType) env->DeleteLocalRef(jModelType);
    if (jMmproj) env->DeleteLocalRef(jMmproj);
    if (jTokenizer) env->DeleteLocalRef(jTokenizer);
    env->DeleteLocalRef(modelTypeCls);
    env->DeleteLocalRef(cls);
    return obj;
}

// Build a com.geniex.sdk.bean.ModelDetail from a geniex_ModelDetail.
jobject build_model_detail(JNIEnv* env, const geniex_ModelDetail& d) {
    jclass cls = env->FindClass("com/geniex/sdk/bean/ModelDetail");
    if (!cls) return nullptr;
    jclass    modelTypeCls = env->FindClass("com/geniex/sdk/bean/ModelType");
    jmethodID fromValueMid = env->GetStaticMethodID(modelTypeCls, "fromValue", "(I)Lcom/geniex/sdk/bean/ModelType;");
    jmethodID ctor         = env->GetMethodID(cls,
        "<init>",
        "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;"
                "Lcom/geniex/sdk/bean/ModelType;J[Ljava/lang/String;)V");
    if (!ctor) {
        env->DeleteLocalRef(cls);
        env->DeleteLocalRef(modelTypeCls);
        return nullptr;
    }
    auto    maybe      = [&](const char* s) -> jstring { return env->NewStringUTF(s ? s : ""); };
    jstring jName      = maybe(d.name);
    jstring jModelName = maybe(d.model_name);
    jstring jPluginId  = maybe(d.plugin_id);
    jobject jModelType = env->CallStaticObjectMethod(modelTypeCls, fromValueMid, static_cast<jint>(d.model_type));

    jclass       stringCls   = env->FindClass("java/lang/String");
    jobjectArray jPrecisions = env->NewObjectArray(d.precision_count, stringCls, nullptr);
    for (int32_t i = 0; i < d.precision_count; ++i) {
        jstring p = env->NewStringUTF(d.precisions[i] ? d.precisions[i] : "");
        env->SetObjectArrayElement(jPrecisions, i, p);
        env->DeleteLocalRef(p);
    }

    jobject obj = env->NewObject(
        cls, ctor, jName, jModelName, jPluginId, jModelType, static_cast<jlong>(d.total_size), jPrecisions);

    env->DeleteLocalRef(jName);
    env->DeleteLocalRef(jModelName);
    env->DeleteLocalRef(jPluginId);
    if (jModelType) env->DeleteLocalRef(jModelType);
    env->DeleteLocalRef(jPrecisions);
    env->DeleteLocalRef(stringCls);
    env->DeleteLocalRef(modelTypeCls);
    env->DeleteLocalRef(cls);
    return obj;
}

// Build a com.geniex.sdk.bean.ModelQuery from a geniex_ModelQueryOutput.
jobject build_model_query(JNIEnv* env, const geniex_ModelQueryOutput& out) {
    jclass       candCls     = env->FindClass("com/geniex/sdk/bean/QuantCandidate");
    jmethodID    candCtor    = env->GetMethodID(candCls, "<init>", "(Ljava/lang/String;JZ)V");
    jobjectArray jCandidates = env->NewObjectArray(out.candidate_count, candCls, nullptr);
    for (int32_t i = 0; i < out.candidate_count; ++i) {
        jstring jQuant = env->NewStringUTF(out.candidates[i].quant ? out.candidates[i].quant : "");
        jobject item   = env->NewObject(candCls,
            candCtor,
            jQuant,
            static_cast<jlong>(out.candidates[i].size),
            static_cast<jboolean>(out.candidates[i].is_default));
        env->SetObjectArrayElement(jCandidates, i, item);
        env->DeleteLocalRef(item);
        env->DeleteLocalRef(jQuant);
    }

    jclass    cls          = env->FindClass("com/geniex/sdk/bean/ModelQuery");
    jclass    modelTypeCls = env->FindClass("com/geniex/sdk/bean/ModelType");
    jmethodID fromValueMid = env->GetStaticMethodID(modelTypeCls, "fromValue", "(I)Lcom/geniex/sdk/bean/ModelType;");
    jmethodID ctor         = env->GetMethodID(cls,
        "<init>",
        "(Ljava/lang/String;Ljava/lang/String;Lcom/geniex/sdk/bean/ModelType;"
                "[Lcom/geniex/sdk/bean/QuantCandidate;)V");

    jstring jModelName = env->NewStringUTF(out.model_name ? out.model_name : "");
    jstring jPluginId  = env->NewStringUTF(out.plugin_id ? out.plugin_id : "");
    jobject jModelType = env->CallStaticObjectMethod(modelTypeCls, fromValueMid, static_cast<jint>(out.model_type));

    jobject obj = env->NewObject(cls, ctor, jModelName, jPluginId, jModelType, jCandidates);

    env->DeleteLocalRef(jModelName);
    env->DeleteLocalRef(jPluginId);
    if (jModelType) env->DeleteLocalRef(jModelType);
    env->DeleteLocalRef(jCandidates);
    env->DeleteLocalRef(candCls);
    env->DeleteLocalRef(modelTypeCls);
    env->DeleteLocalRef(cls);
    return obj;
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_ModelManager_init(
    JNIEnv* env, jobject /*thiz*/, jstring jDataDir) {
    std::string dataDir = jstring2str(env, jDataDir);
    return geniex_model_init(dataDir.empty() ? nullptr : dataDir.c_str());
}

extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_ModelManager_deinit(JNIEnv* /*env*/, jobject /*thiz*/) {
    return geniex_model_deinit();
}

extern "C" JNIEXPORT jstring JNICALL Java_com_geniex_sdk_jni_ModelManager_lastErrorMessage(
    JNIEnv* env, jobject /*thiz*/) {
    const char* msg = geniex_model_last_error_message();
    if (!msg) return nullptr;
    return env->NewStringUTF(msg);
}

extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_ModelManager_pull(
    JNIEnv* env, jobject /*thiz*/, jobject inputObj, jobject callback) {
    if (!inputObj) {
        throw_runtime_exception(env, "ModelPullInput is null");
        return -1;
    }

    jclass cls = env->GetObjectClass(inputObj);

    std::string modelName   = read_opt_string(env, cls, inputObj, "model_name");
    std::string quant       = read_opt_string(env, cls, inputObj, "quant");
    std::string localPath   = read_opt_string(env, cls, inputObj, "local_path");
    std::string hfToken     = read_opt_string(env, cls, inputObj, "hf_token");
    std::string chipset     = read_opt_string(env, cls, inputObj, "chipset");
    std::string displayName = read_opt_string(env, cls, inputObj, "display_name");
    int         hubValue    = read_hub_source_value(env, cls, inputObj);
    int         modelType   = read_model_type_value(env, cls, inputObj);
    env->DeleteLocalRef(cls);

    if (modelName.empty()) {
        throw_runtime_exception(env, "ModelPullInput.model_name is empty");
        return -1;
    }

    // Build progress context if a callback was supplied. `ProgressCtx` is
    // stack-local to this call — the FFI is blocking, so progress callbacks
    // can only fire while we're still on this frame.
    ProgressCtx ctx{};
    JavaVM*     vm = nullptr;
    env->GetJavaVM(&vm);
    jobject cb_global            = nullptr;
    jclass  fileProgressClsLocal = nullptr;
    if (callback) {
        cb_global            = env->NewGlobalRef(callback);
        jclass    cbCls      = env->GetObjectClass(callback);
        jmethodID onProgress = env->GetMethodID(cbCls, "onProgress", "([Lcom/geniex/sdk/bean/FileProgress;)Z");
        env->DeleteLocalRef(cbCls);

        fileProgressClsLocal = env->FindClass("com/geniex/sdk/bean/FileProgress");
        jclass    fpGlobal   = static_cast<jclass>(env->NewGlobalRef(fileProgressClsLocal));
        jmethodID fpCtor     = env->GetMethodID(fpGlobal, "<init>", "(Ljava/lang/String;JJ)V");

        ctx.vm               = vm;
        ctx.cb_global        = cb_global;
        ctx.onProgress_mid   = onProgress;
        ctx.fileProgressCls  = fpGlobal;
        ctx.fileProgressCtor = fpCtor;
    }

    geniex_ModelPullInput in{};
    in.struct_size  = sizeof(geniex_ModelPullInput);
    in.model_name   = modelName.c_str();
    in.quant        = quant.empty() ? nullptr : quant.c_str();
    in.hub          = static_cast<geniex_HubSource>(hubValue);
    in.local_path   = localPath.empty() ? nullptr : localPath.c_str();
    in.hf_token     = hfToken.empty() ? nullptr : hfToken.c_str();
    in.chipset      = chipset.empty() ? nullptr : chipset.c_str();
    in.display_name = displayName.empty() ? nullptr : displayName.c_str();
    in.model_type   = modelType;
    if (callback) {
        in.on_progress = progress_trampoline;
        in.user_data   = &ctx;
    }

    LOGi("[ModelManager JNI] pull() model=%s hub=%d quant=%s chipset=%s",
        in.model_name,
        static_cast<int>(in.hub),
        in.quant ? in.quant : "(auto)",
        in.chipset ? in.chipset : "(none)");

    int32_t rc = geniex_model_pull(&in);

    if (callback) {
        if (ctx.fileProgressCls) env->DeleteGlobalRef(ctx.fileProgressCls);
        if (cb_global) env->DeleteGlobalRef(cb_global);
        if (fileProgressClsLocal) env->DeleteLocalRef(fileProgressClsLocal);
    }

    LOGi("[ModelManager JNI] pull() returned rc=%d", rc);
    return rc;
}

extern "C" JNIEXPORT jobjectArray JNICALL Java_com_geniex_sdk_jni_ModelManager_list(JNIEnv* env, jobject /*thiz*/) {
    jclass stringCls = env->FindClass("java/lang/String");

    geniex_ModelListDetailedOutput out{};
    int32_t                        rc = geniex_model_list_detailed(&out);
    if (rc != GENIEX_SUCCESS) {
        LOGe("[ModelManager JNI] list() failed rc=%d", rc);
        jobjectArray empty = env->NewObjectArray(0, stringCls, nullptr);
        env->DeleteLocalRef(stringCls);
        return empty;
    }

    jobjectArray arr = env->NewObjectArray(out.count, stringCls, nullptr);
    for (int32_t i = 0; i < out.count; ++i) {
        jstring s = env->NewStringUTF(out.models[i].name ? out.models[i].name : "");
        env->SetObjectArrayElement(arr, i, s);
        env->DeleteLocalRef(s);
    }
    env->DeleteLocalRef(stringCls);
    geniex_model_list_detailed_free(&out);
    return arr;
}

extern "C" JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_ModelManager_getPaths(
    JNIEnv* env, jobject /*thiz*/, jstring jModelName) {
    std::string name = jstring2str(env, jModelName);
    if (name.empty()) return nullptr;

    geniex_ModelPaths paths{};
    int32_t           rc = geniex_model_get_paths(name.c_str(), &paths);
    if (rc != GENIEX_SUCCESS) {
        LOGe("[ModelManager JNI] getPaths(%s) failed rc=%d", name.c_str(), rc);
        return nullptr;
    }

    jobject obj = build_model_paths(env, paths);
    geniex_model_paths_free(&paths);
    return obj;
}

extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_ModelManager_remove(
    JNIEnv* env, jobject /*thiz*/, jstring jModelName) {
    std::string name = jstring2str(env, jModelName);
    if (name.empty()) return -1;
    return geniex_model_remove(name.c_str());
}

extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_ModelManager_clean(JNIEnv* /*env*/, jobject /*thiz*/) {
    int32_t count = 0;
    int32_t rc    = geniex_model_clean(&count);
    if (rc != GENIEX_SUCCESS) return rc;
    return count;
}

extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_ModelManager_getType(
    JNIEnv* env, jobject /*thiz*/, jstring jModelName) {
    std::string name = jstring2str(env, jModelName);
    if (name.empty()) return -1;
    geniex_ModelType t;
    int32_t          rc = geniex_model_get_type(name.c_str(), &t);
    if (rc != GENIEX_SUCCESS) return rc;
    return static_cast<jint>(t);
}

extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_ModelManager_setType(
    JNIEnv* env, jobject /*thiz*/, jstring jModelName, jint jModelType) {
    std::string name = jstring2str(env, jModelName);
    if (name.empty()) return -1;
    return geniex_model_set_type(name.c_str(), static_cast<geniex_ModelType>(jModelType));
}

extern "C" JNIEXPORT jobjectArray JNICALL Java_com_geniex_sdk_jni_ModelManager_listDetailed(
    JNIEnv* env, jobject /*thiz*/) {
    jclass detailCls = env->FindClass("com/geniex/sdk/bean/ModelDetail");

    geniex_ModelListDetailedOutput out{};
    int32_t                        rc = geniex_model_list_detailed(&out);
    if (rc != GENIEX_SUCCESS) {
        LOGe("[ModelManager JNI] listDetailed() failed rc=%d", rc);
        jobjectArray empty = env->NewObjectArray(0, detailCls, nullptr);
        env->DeleteLocalRef(detailCls);
        return empty;
    }

    jobjectArray arr = env->NewObjectArray(out.count, detailCls, nullptr);
    for (int32_t i = 0; i < out.count; ++i) {
        jobject item = build_model_detail(env, out.models[i]);
        env->SetObjectArrayElement(arr, i, item);
        if (item) env->DeleteLocalRef(item);
    }
    env->DeleteLocalRef(detailCls);
    geniex_model_list_detailed_free(&out);
    return arr;
}

extern "C" JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_ModelManager_query(
    JNIEnv* env, jobject /*thiz*/, jobject inputObj) {
    if (!inputObj) {
        throw_runtime_exception(env, "ModelPullInput is null");
        return nullptr;
    }

    jclass cls = env->GetObjectClass(inputObj);

    std::string modelName   = read_opt_string(env, cls, inputObj, "model_name");
    std::string localPath   = read_opt_string(env, cls, inputObj, "local_path");
    std::string hfToken     = read_opt_string(env, cls, inputObj, "hf_token");
    std::string chipset     = read_opt_string(env, cls, inputObj, "chipset");
    std::string displayName = read_opt_string(env, cls, inputObj, "display_name");
    int         hubValue    = read_hub_source_value(env, cls, inputObj);
    env->DeleteLocalRef(cls);

    if (modelName.empty()) {
        throw_runtime_exception(env, "ModelPullInput.model_name is empty");
        return nullptr;
    }

    geniex_ModelPullInput in{};
    in.struct_size  = sizeof(geniex_ModelPullInput);
    in.model_name   = modelName.c_str();
    in.hub          = static_cast<geniex_HubSource>(hubValue);
    in.local_path   = localPath.empty() ? nullptr : localPath.c_str();
    in.hf_token     = hfToken.empty() ? nullptr : hfToken.c_str();
    in.chipset      = chipset.empty() ? nullptr : chipset.c_str();
    in.display_name = displayName.empty() ? nullptr : displayName.c_str();
    in.model_type   = GENIEX_MODEL_TYPE_AUTO;

    geniex_ModelQueryOutput out{};
    int32_t                 rc = geniex_model_query(&in, &out);
    if (rc != GENIEX_SUCCESS) {
        LOGe("[ModelManager JNI] query(%s) failed rc=%d", modelName.c_str(), rc);
        return nullptr;
    }

    jobject obj = build_model_query(env, out);
    geniex_model_query_free(&out);
    return obj;
}

extern "C" JNIEXPORT jstring JNICALL Java_com_geniex_sdk_jni_ModelManager_resolveAlias(
    JNIEnv* env, jobject /*thiz*/, jstring jAlias) {
    std::string alias = jstring2str(env, jAlias);
    if (alias.empty()) return nullptr;
    char*   out = nullptr;
    int32_t rc  = geniex_model_resolve_alias(alias.c_str(), &out);
    if (rc != GENIEX_SUCCESS || !out) return nullptr;
    jstring result = env->NewStringUTF(out);
    geniex_free(out);
    return result;
}
