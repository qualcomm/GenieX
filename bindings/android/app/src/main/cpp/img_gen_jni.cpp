//
// Created by echonfl on 2025/12/8.
//
#include <android/log.h>
#include <jni.h>
#include <pthread.h>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "android_utils.h"
#include "geniex.h"
#include "jni_cb.h"
#include "jniutils.h"

static std::unordered_map<void *, std::atomic<bool> *> g_stopFlags;
static std::mutex                                      g_stopFlagsMutex;

using namespace jniutils;
using namespace geniex_android_sdk;

extern "C" {
geniex_ImageSamplerConfig extract_image_sampler_config(JNIEnv *env, jobject inputObj) {
    geniex_ImageSamplerConfig out = {};

    if (!inputObj) {
        LOGe("extract_image_sampler_config: inputObj is null");
        return out;
    }

    jclass cls = env->GetObjectClass(inputObj);
    if (!cls) {
        LOGe("extract_image_sampler_config: GetObjectClass returned null");
        return out;
    }
    out.method = getStringField(env, cls, inputObj, "method");
    LOGd("geniex_ImageGenTxt2ImgInput.config.sampler_config.method:%s", out.method);

    out.steps = getIntField(env, cls, inputObj, "steps");
    LOGd("geniex_ImageGenTxt2ImgInput.config.sampler_config.steps:%d", out.steps);

    out.guidance_scale = getFloatField(env, cls, inputObj, "guidance_scale");
    LOGd("geniex_ImageGenTxt2ImgInput.config.sampler_config.guidance_scale:%f", out.guidance_scale);

    out.eta = getFloatField(env, cls, inputObj, "eta");
    LOGd("geniex_ImageGenTxt2ImgInput.config.sampler_config.eta:%f", out.eta);

    out.seed = getIntField(env, cls, inputObj, "seed");
    LOGd("geniex_ImageGenTxt2ImgInput.config.sampler_config.seed:%d", out.seed);

    env->DeleteLocalRef(inputObj);
    return out;
}

geniex_SchedulerConfig extract_scheduler_config(JNIEnv *env, jobject inputObj) {
    geniex_SchedulerConfig out = {};

    if (!inputObj) {
        LOGe("extract_scheduler_config: inputObj is null");
        return out;
    }

    jclass cls = env->GetObjectClass(inputObj);
    if (!cls) {
        LOGe("extract_scheduler_config: GetObjectClass returned null");
        return out;
    }
    out.type                = getStringField(env, cls, inputObj, "type");
    out.num_train_timesteps = getIntField(env, cls, inputObj, "num_train_timesteps");
    out.steps_offset        = getIntField(env, cls, inputObj, "steps_offset");
    out.beta_start          = getFloatField(env, cls, inputObj, "beta_start");
    out.beta_end            = getFloatField(env, cls, inputObj, "beta_end");
    out.beta_schedule       = getStringField(env, cls, inputObj, "beta_schedule");
    out.prediction_type     = getStringField(env, cls, inputObj, "prediction_type");
    out.timestep_type       = getStringField(env, cls, inputObj, "timestep_type");
    out.timestep_spacing    = getStringField(env, cls, inputObj, "timestep_spacing");
    out.interpolation_type  = getStringField(env, cls, inputObj, "interpolation_type");
    out.config_path         = getStringField(env, cls, inputObj, "config_path");
    env->DeleteLocalRef(inputObj);
    return out;
}

geniex_ImageGenerationConfig extract_image_generation_config(JNIEnv *env, jobject inputObj) {
    geniex_ImageGenerationConfig out = {};

    if (!inputObj) {
        LOGe("extract_image_generation_config: inputObj is null");
        return out;
    }

    jclass cls = env->GetObjectClass(inputObj);
    if (!cls) {
        LOGe("extract_image_generation_config: GetObjectClass returned null");
        return out;
    }
    // ----------- prompts  -----------
    static thread_local std::vector<std::string>  promptsStorage;
    static thread_local std::vector<const char *> promptsPtrs;
    getStringArrayField(env, inputObj, cls, "prompts", promptsStorage, promptsPtrs);
    if (!promptsStorage.empty()) {
        out.prompts      = promptsPtrs.data();
        out.prompt_count = promptsPtrs.size();
    } else {
        out.prompts      = nullptr;
        out.prompt_count = 0;
    }
    for (int i = 0; i < out.prompt_count; ++i) {
        LOGd("geniex_ImageGenTxt2ImgInput.config.prompt_count:%d = %s", i, out.prompts[i]);
    }

    // ----------- negative_prompts  -----------
    static thread_local std::vector<std::string>  negativePromptsStorage;
    static thread_local std::vector<const char *> negativePromptsPtrs;
    getStringArrayField(env, inputObj, cls, "negative_prompts", negativePromptsStorage, negativePromptsPtrs);
    if (!promptsStorage.empty()) {
        out.negative_prompts      = negativePromptsPtrs.data();
        out.negative_prompt_count = negativePromptsPtrs.size();
    } else {
        out.negative_prompts      = nullptr;
        out.negative_prompt_count = 0;
    }
    for (int i = 0; i < out.negative_prompt_count; ++i) {
        LOGd("geniex_ImageGenTxt2ImgInput.config.negative_prompts:%d = %s", i, out.negative_prompts[i]);
    }

    out.height = getIntField(env, cls, inputObj, "height");
    LOGd("geniex_ImageGenTxt2ImgInput.config.height:%d", out.height);
    out.width = getIntField(env, cls, inputObj, "width");
    LOGd("geniex_ImageGenTxt2ImgInput.config.width:%d", out.width);

    jobject samplerConfigObj =
        getObjectField(env, cls, inputObj, "sampler_config", "Lcom/geniex/sdk/bean/ImageSamplerConfig;");
    if (!samplerConfigObj) {
        out.sampler_config = {};
    } else {
        out.sampler_config = extract_image_sampler_config(env, samplerConfigObj);
    }

    jobject schedulerConfigObj =
        getObjectField(env, cls, inputObj, "scheduler_config", "Lcom/geniex/sdk/bean/SchedulerConfig;");
    if (!schedulerConfigObj) {
        out.scheduler_config = {};
    } else {
        out.scheduler_config = extract_scheduler_config(env, schedulerConfigObj);
    }

    out.strength = getFloatField(env, cls, inputObj, "strength");
    LOGd("geniex_ImageGenTxt2ImgInput.config.strength:%f", out.strength);

    env->DeleteLocalRef(inputObj);
    return out;
}

geniex_ImageGenTxt2ImgInput extract_image_gen_txt_2_img_input(
    JNIEnv *env, jobject inputObj, geniex_ImageGenerationConfig *config) {
    geniex_ImageGenTxt2ImgInput out = {};

    if (!inputObj) {
        LOGe("extract_image_gen_txt_2_img_input: inputObj is null");
        return out;
    }

    jclass cls = env->GetObjectClass(inputObj);
    if (!cls) {
        LOGe("extract_image_gen_txt_2_img_input: GetObjectClass returned null");
        return out;
    }
    out.prompt_utf8 = getStringField(env, cls, inputObj, "prompt_utf8");
    LOGd("geniex_ImageGenTxt2ImgInput out.prompt_utf8:%s", out.prompt_utf8);

    jobject configObj = getObjectField(env, cls, inputObj, "config", "Lcom/geniex/sdk/bean/ImageGenerationConfig;");
    if (!configObj) {
        out.config = nullptr;
    } else {
        *config    = extract_image_generation_config(env, configObj);
        out.config = config;
    }

    out.output_path = getStringField(env, cls, inputObj, "output_path");
    LOGd("geniex_ImageGenTxt2ImgInput out.output_path:%s", out.output_path);
    return out;
}

geniex_ImageGenImg2ImgInput extract_image_gen_img_2_img_input(
    JNIEnv *env, jobject inputObj, geniex_ImageGenerationConfig *config) {
    geniex_ImageGenImg2ImgInput out = {};

    if (!inputObj) {
        LOGe("extract_image_gen_img_2_img_input: inputObj is null");
        return out;
    }

    jclass cls = env->GetObjectClass(inputObj);
    if (!cls) {
        LOGe("extract_image_gen_img_2_img_input: GetObjectClass returned null");
        return out;
    }

    out.init_image_path = getStringField(env, cls, inputObj, "init_image_path");
    out.prompt_utf8     = getStringField(env, cls, inputObj, "prompt_utf8");
    LOGd("geniex_ImageGenImg2ImgInput out.prompt_utf8:%s", out.prompt_utf8);

    jobject configObj = getObjectField(env, cls, inputObj, "config", "Lcom/geniex/sdk/bean/ImageGenerationConfig;");
    if (!configObj) {
        out.config = nullptr;
    } else {
        *config    = extract_image_generation_config(env, configObj);
        out.config = config;
    }

    out.output_path = getStringField(env, cls, inputObj, "output_path");
    LOGd("geniex_ImageGenImg2ImgInput out.output_path:%s", out.output_path);
    return out;
}

geniex_ImageGenCreateInput extract_image_gen_create_input(JNIEnv *env, jobject inputObj) {
    geniex_ImageGenCreateInput out = {};

    if (!inputObj) {
        LOGe("extract_image_gen_create_input: inputObj is null");
        return out;
    }

    jclass cls = env->GetObjectClass(inputObj);
    if (!cls) {
        LOGe("extract_image_gen_create_input: GetObjectClass returned null");
        return out;
    }

    out.model_name = getStringField(env, cls, inputObj, "model_name");
    out.model_path = getStringField(env, cls, inputObj, "model_path");

    jobject configObj = getObjectField(env, cls, inputObj, "config", "Lcom/geniex/sdk/bean/ModelConfig;");
    if (!configObj) {
        out.config = {};
    } else {
        out.config = extract_model_config(env, configObj);
    }
    env->DeleteLocalRef(configObj);

    out.scheduler_config_path = getStringField(env, cls, inputObj, "scheduler_config_path");
    out.plugin_id             = getStringField(env, cls, inputObj, "plugin_id");
    // Translate user-friendly device_id to internal device string
    const char *raw_device_id = getStringField(env, cls, inputObj, "device_id");
    if (raw_device_id) {
        std::string translated = jniutils::translate_device_id(raw_device_id);
        out.device_id          = jniutils::hold_c_str(translated);
        LOGd("device_id translated: %s -> %s", raw_device_id, translated.c_str());
    } else {
        out.device_id = nullptr;
    }

    env->DeleteLocalRef(inputObj);
    return out;
}
}

extern "C" JNIEXPORT jlong JNICALL Java_com_geniex_sdk_jni_ImgGen_create(
    JNIEnv *env, jobject thiz, jobject image_gen_create_input) {
    geniex_ImageGenCreateInput input = extract_image_gen_create_input(env, image_gen_create_input);
    geniex_ImageGen           *handle;
    int32_t                    result = geniex_imagegen_create(&input, &handle);
    if (result != GENIEX_SUCCESS || !handle) {
        LOGe("[JNI] geniex_img_gen_create failed, error code: %d", result);
        throw_runtime_exception(env, "geniex_img_gen_create failed, error code: %d", result);
        return 0;
    }
    LOGi("[JNI] create() geniex_img_gen_create returned handle=%p", handle);
    return reinterpret_cast<jlong>(handle);
}

extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_ImgGen_txt2Img(
    JNIEnv *env, jobject thiz, jobject image_gen_txt2_img_input, jlong handle) {
    geniex_ImageGenerationConfig config;
    geniex_ImageGenTxt2ImgInput  input  = extract_image_gen_txt_2_img_input(env, image_gen_txt2_img_input, &config);
    geniex_ImageGenOutput        output = {};
    int32_t result = geniex_imagegen_txt2img(reinterpret_cast<geniex_ImageGen *>(handle), &input, &output);
    if (result != GENIEX_SUCCESS) {
        LOGe("[JNI] txt2Img failed, error code: %d", result);
        return result;
    }
    return GENIEX_SUCCESS;
}

extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_ImgGen_destroy(JNIEnv *env, jobject thiz, jlong handle) {
    return geniex_imagegen_destroy(reinterpret_cast<geniex_ImageGen *>(handle));
}

extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_ImgGen_img2Img(
    JNIEnv *env, jobject thiz, jobject image_gen_img2_img_input, jlong handle) {
    geniex_ImageGenerationConfig config;
    geniex_ImageGenImg2ImgInput  input  = extract_image_gen_img_2_img_input(env, image_gen_img2_img_input, &config);
    geniex_ImageGenOutput        output = {};
    int32_t result = geniex_imagegen_img2img(reinterpret_cast<geniex_ImageGen *>(handle), &input, &output);
    if (result != GENIEX_SUCCESS) {
        LOGe("[JNI] img2Img failed, error code: %d", result);
        return result;
    }
    return GENIEX_SUCCESS;
}