#include <jni.h>

#include <string>
#include <vector>

#include "android_utils.h"
#include "geniex.h"
#include "jniutils.h"

using namespace jniutils;
using namespace geniex_android_sdk;

// JNI: create - Create CV model handle
extern "C" {
geniex_CVCapabilities extract_cv_capability(JNIEnv* env, jobject inputObj) {
    geniex_CVCapabilities out = GENIEX_CV_OCR;
    jclass                cls = env->GetObjectClass(inputObj);
    if (!cls) {
        LOGe("extract_cv_capability: GetObjectClass returned null");
        return out;
    }

    jfieldID fid = env->GetFieldID(cls, "capabilities", "Lcom/geniex/sdk/bean/CVCapability;");
    if (check_jni_exception(env, "GetFieldID(capabilities)") || !fid) {
        LOGe("extract_cv_capability field 'capabilities' not found");
    } else {
        jobject cap = (jobject)env->GetObjectField(inputObj, fid);
        if (check_jni_exception(env, "GetObjectField(capabilities)") || !cap) {
            LOGd("extract_cv_capability 'capabilities' is null");
        } else {
            jclass enum_class = env->GetObjectClass(cap);
            if (!enum_class) {
                LOGe("extract_cv_capability: enum_class returned null");
                return out;
            }
            // 获取 name() 方法
            jmethodID name_method = env->GetMethodID(enum_class, "name", "()Ljava/lang/String;");
            if (check_jni_exception(env, "GetFieldID(name)") || !name_method) {
                LOGe("extract_cv_capability field 'capabilities's name_method' not found");
            } else {
                // 调用 name() 获取名称
                jstring name = (jstring)env->CallObjectMethod(cap, name_method);
                if (check_jni_exception(env, "GetFieldID(name value)") || !name) {
                    LOGe("extract_cv_capability field 'capabilities's name_method value' not found");
                } else {
                    std::string name_string = jstring2str(env, name);
                    LOGd("extract_cv_capability capabilities = %s", hold_c_str(name_string));

                    if (strcmp("OCR", name_string.c_str()) == 0) {
                        return GENIEX_CV_OCR;
                    } else if (strcmp("CLASSIFICATION", name_string.c_str()) == 0) {
                        return GENIEX_CV_CLASSIFICATION;
                    } else if (strcmp("SEGMENTATION", name_string.c_str()) == 0) {
                        return GENIEX_CV_SEGMENTATION;
                    } else if (strcmp("CUSTOM", name_string.c_str()) == 0) {
                        return GENIEX_CV_CUSTOM;
                    } else {
                        return GENIEX_CV_OCR;
                    }
                }
            }
        }
    }

    return GENIEX_CV_OCR;
}

geniex_CVModelConfig extract_cv_model_config(JNIEnv* env, jobject inputObj) {
    geniex_CVModelConfig out = {};

    if (!inputObj) {
        LOGe("extract_cv_create_input: inputObj is null");
        return out;
    }

    jclass cls = env->GetObjectClass(inputObj);
    if (!cls) {
        LOGe("extract_cv_create_input: GetObjectClass returned null");
        return out;
    }
    out.capabilities = extract_cv_capability(env, inputObj);
    LOGd("extract_cv_create_input capabilities %d", out.capabilities);

    jfieldID fid;
    jstring  jstr;

    // === det_model_path ===
    fid = env->GetFieldID(cls, "det_model_path", "Ljava/lang/String;");
    if (check_jni_exception(env, "GetFieldID(det_model_path)") || !fid) {
        LOGe("extract_cv_model_config field 'det_model_path' not found");
    } else {
        jstr = (jstring)env->GetObjectField(inputObj, fid);
        if (check_jni_exception(env, "GetObjectField(det_model_path)") || !jstr) {
            LOGd("extract_cv_model_config 'det_model_path' is null");
        } else {
            std::string s      = jstring2str(env, jstr);
            out.det_model_path = hold_c_str(s);
            LOGd("extract_cv_model_config det_model_path = %s", s.c_str());
            env->DeleteLocalRef(jstr);
        }
    }

    // === rec_model_path ===
    fid = env->GetFieldID(cls, "rec_model_path", "Ljava/lang/String;");
    if (check_jni_exception(env, "GetFieldID(rec_model_path)") || !fid) {
        LOGe("extract_cv_model_config field 'rec_model_path' not found");
    } else {
        jstr = (jstring)env->GetObjectField(inputObj, fid);
        if (check_jni_exception(env, "GetObjectField(rec_model_path)") || !jstr) {
            LOGd("extract_cv_model_config 'rec_model_path' is null");
        } else {
            std::string s      = jstring2str(env, jstr);
            out.rec_model_path = hold_c_str(s);
            LOGd("extract_cv_model_config rec_model_path = %s", s.c_str());
            env->DeleteLocalRef(jstr);
        }
    }

    // === char_dict_path ===
    fid = env->GetFieldID(cls, "char_dict_path", "Ljava/lang/String;");
    if (check_jni_exception(env, "GetFieldID(char_dict_path)") || !fid) {
        LOGe("extract_cv_model_config field 'char_dict_path' not found");
    } else {
        jstr = (jstring)env->GetObjectField(inputObj, fid);
        if (check_jni_exception(env, "GetObjectField(char_dict_path)") || !jstr) {
            LOGd("extract_cv_model_config 'char_dict_path' is null");
        } else {
            std::string s      = jstring2str(env, jstr);
            out.char_dict_path = hold_c_str(s);
            LOGd("extract_cv_model_config char_dict_path = %s", s.c_str());
            env->DeleteLocalRef(jstr);
        }
    }

    return out;
}

geniex_CVCreateInput extract_cv_create_input(JNIEnv* env, jobject inputObj) {
    geniex_CVCreateInput out = {};
    if (!inputObj) {
        LOGe("extract_cv_create_input: inputObj is null");
        return out;
    }

    jclass cls = env->GetObjectClass(inputObj);
    if (!cls) {
        LOGe("extract_cv_create_input: GetObjectClass returned null");
        return out;
    }

    jfieldID fid;
    jstring  jstr;

    // === model_name ===
    fid = env->GetFieldID(cls, "model_name", "Ljava/lang/String;");
    if (check_jni_exception(env, "GetFieldID(model_name)") || !fid) {
        LOGe("extract_cv_create_input field 'model_name' not found");
    } else {
        jstr = (jstring)env->GetObjectField(inputObj, fid);
        if (check_jni_exception(env, "GetObjectField(model_name)") || !jstr) {
            LOGd("extract_cv_create_input 'model_name' is null");
        } else {
            std::string s  = jstring2str(env, jstr);
            out.model_name = hold_c_str(s);
            LOGd("extract_cv_create_input model_name = %s", s.c_str());
            env->DeleteLocalRef(jstr);
        }
    }

    // === plugin_id ===
    fid = env->GetFieldID(cls, "plugin_id", "Ljava/lang/String;");
    if (check_jni_exception(env, "GetFieldID(plugin_id)") || !fid) {
        LOGe("extract_cv_create_input field 'plugin_id' not found");
    } else {
        jstr = (jstring)env->GetObjectField(inputObj, fid);
        if (check_jni_exception(env, "GetObjectField(plugin_id)") || !jstr) {
            LOGd("extract_cv_create_input 'plugin_id' is null");
        } else {
            std::string s = jstring2str(env, jstr);
            out.plugin_id = hold_c_str(s);
            LOGd("extract_cv_create_input plugin_id = %s", s.c_str());
            env->DeleteLocalRef(jstr);
        }
    }

    // === device_id ===
    {
        std::string raw_dev;
        fid = env->GetFieldID(cls, "device_id", "Ljava/lang/String;");
        if (fid) {
            jstr = (jstring)env->GetObjectField(inputObj, fid);
            if (jstr) {
                raw_dev = jstring2str(env, jstr);
                env->DeleteLocalRef(jstr);
            }
        }
        ResolvedDevice rdev = resolve_device(out.plugin_id, out.model_name, raw_dev);
        out.device_id       = rdev.device_id.empty() ? nullptr : hold_c_str(rdev.device_id);
        LOGd("extract_cv_create_input device_id = %s (from raw='%s')",
            rdev.device_id.empty() ? "(null)" : rdev.device_id.c_str(),
            raw_dev.c_str());
    }

    // === license_id ===
    fid = env->GetFieldID(cls, "license_id", "Ljava/lang/String;");
    if (check_jni_exception(env, "GetFieldID(license_id)") || !fid) {
        LOGe("extract_cv_create_input field 'license_id' not found");
    } else {
        jstr = (jstring)env->GetObjectField(inputObj, fid);
        if (check_jni_exception(env, "GetObjectField(license_id)") || !jstr) {
            LOGd("extract_cv_create_input 'license_id' is null");
        } else {
            std::string s  = jstring2str(env, jstr);
            out.license_id = hold_c_str(s);
            LOGd("extract_cv_create_input license_id = %s", s.c_str());
            env->DeleteLocalRef(jstr);
        }
    }

    // === license_key ===
    fid = env->GetFieldID(cls, "license_key", "Ljava/lang/String;");
    if (check_jni_exception(env, "GetFieldID(license_key)") || !fid) {
        LOGe("extract_cv_create_input field 'license_key' not found");
    } else {
        jstr = (jstring)env->GetObjectField(inputObj, fid);
        if (check_jni_exception(env, "GetObjectField(license_key)") || !jstr) {
            LOGd("extract_cv_create_input 'license_key' is null");
        } else {
            std::string s   = jstring2str(env, jstr);
            out.license_key = hold_c_str(s);
            LOGd("extract_cv_create_input license_key = %s", s.c_str());
            env->DeleteLocalRef(jstr);
        }
    }

    // === config ===
    fid = env->GetFieldID(cls, "config", "Lcom/geniex/sdk/bean/CVModelConfig;");
    if (check_jni_exception(env, "GetFieldID(config)") || !fid) {
        LOGe("extract_cv_create_input field 'config' not found");
    } else {
        jobject config = (jobject)env->GetObjectField(inputObj, fid);
        if (check_jni_exception(env, "GetObjectField(config)") || !config) {
            LOGd("extract_cv_create_input 'config' is null");
        } else {
            out.config = extract_cv_model_config(env, config);
            env->DeleteLocalRef(config);
        }
    }
    return out;
}

//=============================================================================
// 创建 BoundingBox Java 对象
jobject create_bounding_box(JNIEnv* env, geniex_BoundingBox bbox) {
    jclass    bbox_class       = env->FindClass("com/geniex/sdk/bean/BoundingBox");
    jmethodID bbox_constructor = env->GetMethodID(bbox_class, "<init>", "(FFFF)V");

    jobject bbox_obj = env->NewObject(bbox_class, bbox_constructor, bbox.x, bbox.y, bbox.width, bbox.height);
    return bbox_obj;
}

// 创建单个 CVResult Java 对象
jobject create_cv_result(JNIEnv* env, geniex_CVResult result) {
    jclass    result_class       = env->FindClass("com/geniex/sdk/bean/CVResult");
    jmethodID result_constructor = env->GetMethodID(result_class, "<init>", "()V");

    jobject result_obj = env->NewObject(result_class, result_constructor);

    // 设置 imagePaths (List<String>)
    if (result.image_paths && result.image_count > 0) {
        jobject  image_paths_list  = create_string_list(env, result.image_paths, result.image_count);
        jfieldID image_paths_field = env->GetFieldID(result_class, "image_paths", "Ljava/util/ArrayList;");
        env->SetObjectField(result_obj, image_paths_field, image_paths_list);
        env->DeleteLocalRef(image_paths_list);
    }

    // 设置 classId
    jfieldID class_id_field = env->GetFieldID(result_class, "class_id", "I");
    env->SetIntField(result_obj, class_id_field, result.class_id);

    // 设置 confidence
    jfieldID confidence_field = env->GetFieldID(result_class, "confidence", "F");
    env->SetFloatField(result_obj, confidence_field, result.confidence);

    // 设置 bounding box
    jfieldID bbox_field = env->GetFieldID(result_class, "bbox", "Lcom/geniex/sdk/bean/BoundingBox;");
    jobject  bbox_obj   = create_bounding_box(env, result.bbox);
    env->SetObjectField(result_obj, bbox_field, bbox_obj);
    env->DeleteLocalRef(bbox_obj);

    // 设置 text
    if (result.text) {
        jfieldID text_field = env->GetFieldID(result_class, "text", "Ljava/lang/String;");
        jstring  text_str   = env->NewStringUTF(result.text);
        env->SetObjectField(result_obj, text_field, text_str);
        env->DeleteLocalRef(text_str);
    }

    // 设置 embedding (List<Float>)
    if (result.embedding && result.embedding_dim > 0) {
        jobject embedding_list = create_float_list(env, result.embedding, result.embedding_dim);

        jfieldID embedding_field = env->GetFieldID(result_class, "embedding", "Ljava/util/ArrayList;");
        env->SetObjectField(result_obj, embedding_field, embedding_list);
        env->DeleteLocalRef(embedding_list);
    }

    return result_obj;
}

// 主函数：创建 CVResult List
jobject create_cv_results(JNIEnv* env, geniex_CVInferOutput output) {
    if (output.results == NULL || output.result_count <= 0) {
        // 返回空的 ArrayList
        jclass    array_list_class       = env->FindClass("java/util/ArrayList");
        jmethodID array_list_constructor = env->GetMethodID(array_list_class, "<init>", "()V");
        return env->NewObject(array_list_class, array_list_constructor);
    }

    jclass    array_list_class       = env->FindClass("java/util/ArrayList");
    jmethodID array_list_constructor = env->GetMethodID(array_list_class, "<init>", "()V");
    jmethodID array_list_add         = env->GetMethodID(array_list_class, "add", "(Ljava/lang/Object;)Z");

    jobject results_list = env->NewObject(array_list_class, array_list_constructor);

    for (int i = 0; i < output.result_count; i++) {
        jobject result_obj = create_cv_result(env, output.results[i]);
        env->CallBooleanMethod(results_list, array_list_add, result_obj);
        env->DeleteLocalRef(result_obj);
    }

    return results_list;
}

// 资源清理函数（保持不变）
void free_cv_infer_output(geniex_CVInferOutput output) {
    if (output.results == NULL) return;

    for (int i = 0; i < output.result_count; i++) {
        geniex_CVResult result = output.results[i];

        if (result.image_paths) {
            for (int j = 0; j < result.image_count; j++) {
                geniex_free((void*)result.image_paths[j]);
            }
            geniex_free(result.image_paths);
        }

        if (result.text) {
            geniex_free((void*)result.text);
        }

        if (result.embedding) {
            geniex_free(result.embedding);
        }
    }

    geniex_free(output.results);
}
}

// JNI: destroy - Clean up CV resources
extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_Cv_destroy(JNIEnv*, jobject, jlong handle) {
    LOGd("[JNI] destroy() called, handle=%p", (void*)handle);
    if (handle) {
        int32_t result = geniex_cv_destroy((geniex_CV*)handle);
        if (result != GENIEX_SUCCESS) {
            LOGe("[JNI] destroy() failed, error code: %d", result);
        }
        return result;
    }
    return 0;
}

// JNI: infer - Perform CV inference on image
// extern "C"
// JNIEXPORT jstring JNICALL
// Java_com_geniex_sdk_jni_Cv_infer(JNIEnv *env, jobject, jlong handle, jstring imagePath) {

//}

// JNI: getProfilingData - Get profiling data for CV model
extern "C" JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Cv_getProfilingData(JNIEnv* env, jobject, jlong handle) {
    // CV API doesn't provide get_profiling_data function
    // Return empty profiling data
    geniex_ProfileData data = {};
    return extract_profiling_data(env, data);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_geniex_sdk_jni_Cv_create(JNIEnv* env, jobject thiz, jobject input) {
    try {
        geniex_CVCreateInput in = extract_cv_create_input(env, input);
        geniex_CV*           h  = nullptr;
        LOGd("[JNI] create() geniex_cv_create called");

        int32_t result = geniex_cv_create(&in, &h);

        if (result != GENIEX_SUCCESS || !h) {
            LOGe("[JNI] create() failed, error code: %d", result);
            throw_runtime_exception(env, "CV create failed, error code: %d", result);
            return 0;
        }

        LOGd("[JNI] create() geniex_cv_create returned handle=%p", h);
        return (jlong)h;
    } catch (const std::exception& e) {
        LOGe("[JNI] create() exception: %s", e.what());
        return 0;
    }
}

extern "C" JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Cv_infer(
    JNIEnv* env, jobject thiz, jlong handle, jstring image_path) {
    if (!handle || !image_path) return nullptr;

    std::string c_image_path = jstring2str(env, image_path);

    geniex_CVInferInput input = {};
    input.input_image_path    = c_image_path.c_str();

    geniex_CVInferOutput output = {};
    int32_t              result = geniex_cv_infer((geniex_CV*)handle, &input, &output);

    if (result < 0 || !output.results || output.result_count <= 0) {
        throw_runtime_exception(env, "CV infer failed, error code: %d", result);
        return nullptr;
    }

    jobject cv_results = create_cv_results(env, output);
    free_cv_infer_output(output);
    return cv_results;
}