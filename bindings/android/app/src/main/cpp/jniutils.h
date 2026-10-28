// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <jni.h>

#include <deque>
#include <string>
#include <vector>

#include "geniex.h"

namespace jniutils {
struct GenerationConfigStorage {
    geniex_GenerationConfig            value{};
    geniex_SamplerConfig               sampler{};
    bool                               has_sampler = false;
    std::deque<std::string>            grammar_strings;
    std::vector<geniex_GrammarTrigger> grammar_triggers;
    std::vector<int32_t>               grammar_preserved_tokens;

    geniex_GenerationConfig* get() {
        value.sampler_config = has_sampler ? &sampler : nullptr;
        return &value;
    }
};

struct DecideInputStorage {
    geniex_DecideInput                            value{};
    std::deque<std::string>                       strings;
    std::vector<geniex_DecideQuestion>            questions;
    std::vector<std::vector<geniex_DecideOption>> options;
    std::vector<const char*>                      image_paths;
    std::vector<const char*>                      audio_paths;

    geniex_DecideInput* get();
};

GenerationConfigStorage extract_generation_config(JNIEnv* env, jobject configObj);

DecideInputStorage extract_decide_input(JNIEnv* env, jobject inputObj);

jobject decide_output_to_java(JNIEnv* env, const geniex_DecideOutput& output);

geniex_ModelConfig extract_model_config(JNIEnv* env, jobject configObj);

void getStringArrayField(JNIEnv* env, jobject obj, jclass cls, const char* fieldName, std::vector<std::string>& storage,
    std::vector<const char*>& ptrs);

jobject extract_profiling_data(JNIEnv* env, const geniex_ProfileData& data);

std::string jstring2str(JNIEnv* env, jstring jstr);

/**
 * Result of resolving a (plugin_id, device_id_alias) pair. `ngl` is the
 * resolved n_gpu_layers to copy into `geniex_ModelConfig.n_gpu_layers`
 * (cpu / qairt -> 0; gpu / npu / hybrid pass the caller's value through,
 * -1 = all layers).
 */
struct ResolvedDevice {
    std::string device_id;
    int32_t     ngl = -1;
    std::string warning;  // non-empty when the alias was coerced
};

/**
 * Thin wrapper over the SDK's `geniex_resolve_device`. The alias table
 * (cpu / gpu / npu / hybrid, qairt coercion) lives in
 * `sdk/src/device.cpp` — this helper just marshals the result into
 * Android-local types so the 8 JNI call sites stay stable.
 *
 * `model_name` may be null when unknown; it is currently unused by the
 * resolver and reserved for future model-specific defaults. `ngl_default`
 * is the caller's n_gpu_layers, passed through for gpu / npu / hybrid.
 */
ResolvedDevice resolve_device(
    const char* plugin_id, const char* model_name, const std::string& raw_device_id, int32_t ngl_default);

const char* hold_c_str(const std::string& s);

std::vector<std::string> jstringArray2vec(JNIEnv* env, jobjectArray arr);

std::vector<int32_t> jintArray2vec(JNIEnv* env, jintArray arr);

std::vector<float> jfloatArray2vec(JNIEnv* env, jfloatArray arr);

geniex_LlmCreateInput extract_llm_create_input(JNIEnv* env, jobject inputObj);

geniex_VlmCreateInput extract_vlm_create_input(JNIEnv* env, jobject inputObj);

void clear_jni_cstr_pool();

// The returned messages point into the hold_c_str pool and own their tool-call
// arrays: call free_*_chat_messages, then clear_jni_cstr_pool, once the SDK
// call that consumes them has returned.
std::vector<geniex_LlmChatMessage> extract_llm_chat_messages(JNIEnv* env, jobjectArray jmessages);
void                               free_llm_chat_messages(std::vector<geniex_LlmChatMessage>& msgs);
void                               free_chat_template_grammar(geniex_ChatTemplateGrammar& grammar);
jobject chat_template_grammar_to_java(JNIEnv* env, const geniex_ChatTemplateGrammar& grammar);

std::vector<geniex_VlmChatMessage> extract_vlm_chat_messages(JNIEnv* env, jobjectArray jmessages);
void                               free_vlm_chat_messages(std::vector<geniex_VlmChatMessage>& msgs);

// Extract image and audio paths from VlmChatMessage contents
void extract_media_paths_from_messages(
    JNIEnv* env, jobjectArray jmessages, std::vector<std::string>& image_paths, std::vector<std::string>& audio_paths);

void setup_redirect_stdout_stderr();
}  // namespace jniutils
