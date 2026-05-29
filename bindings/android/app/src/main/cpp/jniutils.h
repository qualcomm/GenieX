#pragma once

#include <jni.h>

#include <string>
#include <vector>

#include "geniex.h"

namespace jniutils {
geniex_GenerationConfig extract_generation_config(JNIEnv* env, jobject configObj);

geniex_SamplerConfig extract_sampler_config(JNIEnv* env, jobject configObj);

geniex_ModelConfig extract_model_config(JNIEnv* env, jobject configObj);

//    std::vector<geniex_ChatMessage> extract_chat_messages(JNIEnv* env, jobjectArray jmessages,
//    std::vector<std::string>& str_buf);
void getStringArrayField(JNIEnv* env, jobject obj, jclass cls, const char* fieldName, std::vector<std::string>& storage,
    std::vector<const char*>& ptrs);

jobject extract_profiling_data(JNIEnv* env, const geniex_ProfileData& data);

std::string jstring2str(JNIEnv* env, jstring jstr);

/**
 * Result of resolving a (plugin_id, device_id_alias) pair. When
 * `ngl_override` is non-negative it must be copied into the caller's
 * `geniex_ModelConfig.n_gpu_layers` to match the alias semantics
 * (e.g. "cpu" -> 0, "hybrid" -> 999).
 */
struct ResolvedDevice {
    std::string device_id;
    int32_t     ngl_override = -1;  // <0 = leave caller's value untouched
    std::string warning;            // non-empty when the alias was coerced
};

/**
 * Thin wrapper over the SDK's `geniex_resolve_device`. The alias table
 * (cpu / gpu / npu / hybrid, model-specific overrides, qairt coercion)
 * lives in `sdk/src/device.cpp` — this helper just marshals the result
 * into Android-local types so the 8 JNI call sites stay stable.
 *
 * `model_name` may be null when unknown; it's only consulted for
 * model-specific default overrides.
 */
ResolvedDevice resolve_device(const char* plugin_id, const char* model_name, const std::string& raw_device_id);

const char* hold_c_str(const std::string& s);

std::vector<std::string> jstringArray2vec(JNIEnv* env, jobjectArray arr);

std::vector<int32_t> jintArray2vec(JNIEnv* env, jintArray arr);

geniex_LlmCreateInput extract_llm_create_input(JNIEnv* env, jobject inputObj);

geniex_VlmCreateInput extract_vlm_create_input(JNIEnv* env, jobject inputObj);

void                               clear_jni_cstr_pool();
std::vector<geniex_LlmChatMessage> extract_llm_chat_messages(
    JNIEnv* env, jobjectArray jmessages, std::vector<std::string>& str_buf);

std::vector<geniex_VlmChatMessage> extract_vlm_chat_messages(JNIEnv* env, jobjectArray jmessages);

// Extract image and audio paths from VlmChatMessage contents
void extract_media_paths_from_messages(
    JNIEnv* env, jobjectArray jmessages, std::vector<std::string>& image_paths, std::vector<std::string>& audio_paths);

void setup_redirect_stdout_stderr();
}  // namespace jniutils
