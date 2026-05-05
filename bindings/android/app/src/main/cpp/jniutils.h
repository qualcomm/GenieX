#pragma once

#include <jni.h>

#include <string>
#include <vector>

#include "geniex.h"

namespace jniutils {
geniex_GenerationConfig extract_generation_config(JNIEnv* env, jobject configObj);

geniex_SamplerConfig extract_sampler_config(JNIEnv* env, jobject configObj);

geniex_EmbeddingConfig extract_embedding_config(JNIEnv* env, jobject configObj);

geniex_RerankConfig extract_rerank_config(JNIEnv* env, jobject configObj);

geniex_ModelConfig extract_model_config(JNIEnv* env, jobject configObj);

//    std::vector<geniex_ChatMessage> extract_chat_messages(JNIEnv* env, jobjectArray jmessages,
//    std::vector<std::string>& str_buf);
void getStringArrayField(JNIEnv* env, jobject obj, jclass cls, const char* fieldName, std::vector<std::string>& storage,
    std::vector<const char*>& ptrs);

jobject extract_profiling_data(JNIEnv* env, const geniex_ProfileData& data);

std::string jstring2str(JNIEnv* env, jstring jstr);

/**
 * Translate user-friendly device_id to the internal device string.
 * Aliases mirror bindings/go/device.go / bindings/python/geniex/auto.py:
 *   "cpu"    -> ""           (pure CPU; callers should also force ngl=0)
 *   "gpu"    -> "GPUOpenCL"
 *   "npu"    -> "HTP0"       (single-session HTP — pinned)
 *   "hybrid" -> ""           (empty; per-tensor HTP+CPU scheduler, caller
 *                             should also force ngl=999)
 *  anything else -> passed through unchanged so callers can still supply
 *  literal ids like "HTP0,HTP1,HTP2,HTP3".
 */
std::string translate_device_id(const std::string& device_id);

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
 * Map a (plugin_id, user-supplied device_id) pair to the concrete
 * (device_id, n_gpu_layers) the SDK plugins expect. Mirrors
 * geniex_sdk.ResolveDevice on the Go side.
 *
 * When `raw_device_id` is empty, the plugin's default alias kicks in
 * (hybrid for llama_cpp, npu for qairt). qairt coerces any non-NPU
 * alias to NPU with a warning.
 */
ResolvedDevice resolve_device(const char* plugin_id, const std::string& raw_device_id);

const char* hold_c_str(const std::string& s);

std::vector<std::string> jstringArray2vec(JNIEnv* env, jobjectArray arr);

std::vector<int32_t> jintArray2vec(JNIEnv* env, jintArray arr);

geniex_LlmCreateInput extract_llm_create_input(JNIEnv* env, jobject inputObj);

geniex_VlmCreateInput extract_vlm_create_input(JNIEnv* env, jobject inputObj);

geniex_EmbedderCreateInput extract_embedder_create_input(JNIEnv* env, jobject inputObj);

geniex_RerankerCreateInput extract_reranker_create_input(JNIEnv* env, jobject inputObj);

void                               clear_jni_cstr_pool();
std::vector<geniex_LlmChatMessage> extract_llm_chat_messages(
    JNIEnv* env, jobjectArray jmessages, std::vector<std::string>& str_buf);

std::vector<geniex_VlmChatMessage> extract_vlm_chat_messages(JNIEnv* env, jobjectArray jmessages);

// Extract image and audio paths from VlmChatMessage contents
void extract_media_paths_from_messages(
    JNIEnv* env, jobjectArray jmessages, std::vector<std::string>& image_paths, std::vector<std::string>& audio_paths);

void setup_redirect_stdout_stderr();
}  // namespace jniutils
