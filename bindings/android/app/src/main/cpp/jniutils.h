#pragma once

#include <jni.h>

#include <string>
#include <vector>

#include "ml.h"

namespace jniutils {
ml_GenerationConfig extract_generation_config(JNIEnv* env, jobject configObj);

ml_SamplerConfig extract_sampler_config(JNIEnv* env, jobject configObj);

ml_EmbeddingConfig extract_embedding_config(JNIEnv* env, jobject configObj);

ml_RerankConfig extract_rerank_config(JNIEnv* env, jobject configObj);

ml_ModelConfig extract_model_config(JNIEnv* env, jobject configObj);

//    std::vector<ml_ChatMessage> extract_chat_messages(JNIEnv* env, jobjectArray jmessages, std::vector<std::string>&
//    str_buf);
void getStringArrayField(JNIEnv* env, jobject obj, jclass cls, const char* fieldName, std::vector<std::string>& storage,
    std::vector<const char*>& ptrs);

jobject extract_profiling_data(JNIEnv* env, const ml_ProfileData& data);

std::string jstring2str(JNIEnv* env, jstring jstr);

/**
 * Translate user-friendly device_id to internal device string.
 * Mappings: "dev0" -> "HTP0,HTP1,HTP2,HTP3", "gpu" -> "GPUOpenCL"
 */
std::string translate_device_id(const std::string& device_id);

const char* hold_c_str(const std::string& s);

std::vector<std::string> jstringArray2vec(JNIEnv* env, jobjectArray arr);

std::vector<int32_t> jintArray2vec(JNIEnv* env, jintArray arr);

ml_LlmCreateInput extract_llm_create_input(JNIEnv* env, jobject inputObj);

ml_VlmCreateInput extract_vlm_create_input(JNIEnv* env, jobject inputObj);

ml_EmbedderCreateInput extract_embedder_create_input(JNIEnv* env, jobject inputObj);

ml_RerankerCreateInput extract_reranker_create_input(JNIEnv* env, jobject inputObj);

void                           clear_jni_cstr_pool();
std::vector<ml_LlmChatMessage> extract_llm_chat_messages(
    JNIEnv* env, jobjectArray jmessages, std::vector<std::string>& str_buf);

std::vector<ml_VlmChatMessage> extract_vlm_chat_messages(JNIEnv* env, jobjectArray jmessages);

// Extract image and audio paths from VlmChatMessage contents
void extract_media_paths_from_messages(
    JNIEnv* env, jobjectArray jmessages, std::vector<std::string>& image_paths, std::vector<std::string>& audio_paths);

void setup_redirect_stdout_stderr();
}  // namespace jniutils
