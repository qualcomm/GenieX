#include <android/log.h>
#include <jni.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>  // for strdup if needed
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "android_utils.h"
#include "geniex.h"
#include "jniutils.h"

using namespace geniex_android_sdk;

// =============================================================================
// ASR Streaming Callback Context
// =============================================================================

/**
 * Context for ASR streaming callback - holds JNI references needed to call back into Java
 */
struct AsrStreamingCallbackCtx {
    JavaVM*   vm                  = nullptr;  // JVM pointer for cross-thread access
    jobject   callback_ref        = nullptr;  // Global ref to AsrTranscriptionCallback
    jmethodID onTranscription_mid = nullptr;  // Method ID for onTranscription(String)

    void reset() {
        vm                  = nullptr;
        callback_ref        = nullptr;
        onTranscription_mid = nullptr;
    }
};

// Global map to store callback contexts per ASR handle
static std::unordered_map<void*, AsrStreamingCallbackCtx*> g_asrStreamingContexts;
static std::mutex                                          g_asrStreamingMutex;

/**
 * Get JNIEnv* for the current thread, attaching if necessary.
 * @param vm JavaVM pointer
 * @param attached Output: true if thread was attached (must detach later)
 * @return JNIEnv* or nullptr on failure
 */
static JNIEnv* asr_get_env(JavaVM* vm, bool& attached) {
    attached    = false;
    JNIEnv* env = nullptr;
    if (!vm) return nullptr;

    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK || !env) {
        if (vm->AttachCurrentThread(&env, nullptr) != 0) return nullptr;
        attached = true;
    }
    return env;
}

/**
 * Native callback function invoked by the ASR streaming engine.
 * Routes transcription updates to the Java callback.
 */
static void asr_native_transcription_callback(const char* text, void* user_data) {
    auto* ctx = static_cast<AsrStreamingCallbackCtx*>(user_data);
    if (!ctx || !ctx->callback_ref || !ctx->onTranscription_mid) {
        LOGe("[ASR Streaming] Callback context invalid");
        return;
    }

    if (!text || strlen(text) == 0) {
        return;  // Skip empty transcriptions
    }

    bool    attached = false;
    JNIEnv* env      = asr_get_env(ctx->vm, attached);
    if (!env) {
        LOGe("[ASR Streaming] Failed to get JNIEnv for callback");
        return;
    }

    // Create Java string from transcription text
    jstring jtext = env->NewStringUTF(text);
    if (!jtext) {
        LOGe("[ASR Streaming] Failed to create Java string");
        if (attached) ctx->vm->DetachCurrentThread();
        return;
    }

    // Call the Java callback
    env->CallVoidMethod(ctx->callback_ref, ctx->onTranscription_mid, jtext);

    // Check for exceptions
    if (env->ExceptionCheck()) {
        LOGe("[ASR Streaming] Exception in onTranscription callback");
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    env->DeleteLocalRef(jtext);
    if (attached) ctx->vm->DetachCurrentThread();
}

/**
 * Initialize ASR streaming callback context.
 * @return true on success, false on failure
 */
static bool asr_init_callback_ctx(JNIEnv* env, jobject callback, AsrStreamingCallbackCtx* ctx) {
    if (!env || !callback || !ctx) return false;

    // Get JavaVM
    JavaVM* vm = nullptr;
    if (env->GetJavaVM(&vm) != 0 || !vm) {
        LOGe("[ASR Streaming] Failed to get JavaVM");
        return false;
    }

    // Get callback class and method
    jclass cb_cls = env->GetObjectClass(callback);
    if (!cb_cls) {
        LOGe("[ASR Streaming] Failed to get callback class");
        return false;
    }

    jmethodID onTranscription = env->GetMethodID(cb_cls, "onTranscription", "(Ljava/lang/String;)V");
    if (!onTranscription) {
        LOGe("[ASR Streaming] Failed to get onTranscription method");
        env->DeleteLocalRef(cb_cls);
        return false;
    }

    // Create global reference to callback
    jobject callback_global = env->NewGlobalRef(callback);
    if (!callback_global) {
        LOGe("[ASR Streaming] Failed to create global ref");
        env->DeleteLocalRef(cb_cls);
        return false;
    }

    ctx->vm                  = vm;
    ctx->callback_ref        = callback_global;
    ctx->onTranscription_mid = onTranscription;

    env->DeleteLocalRef(cb_cls);
    return true;
}

/**
 * Dispose ASR streaming callback context.
 */
static void asr_dispose_callback_ctx(JNIEnv* env, AsrStreamingCallbackCtx* ctx) {
    if (!ctx) return;

    if (ctx->callback_ref && env) {
        env->DeleteGlobalRef(ctx->callback_ref);
    }
    ctx->reset();
}

extern "C" {

geniex_AsrCreateInput extract_asr_create_input(JNIEnv* env, jobject inputObj) {
    geniex_AsrCreateInput out = {};

    if (!inputObj) {
        LOGe("extract_asr_create_input: inputObj is null");
        return out;
    }

    jclass cls = env->GetObjectClass(inputObj);
    if (!cls) {
        LOGe("[JNI] extract_asr_create_input: GetObjectClass returned null");
        return out;
    }

    out.model_name     = getStringField(env, cls, inputObj, "model_name");
    out.model_path     = getStringField(env, cls, inputObj, "model_path");
    out.tokenizer_path = getStringField(env, cls, inputObj, "tokenizer_path");

    jobject configObj = getObjectField(env, cls, inputObj, "config", "Lcom/geniex/sdk/bean/ModelConfig;");
    if (!configObj) {
        out.config = {};
    } else {
        out.config = jniutils::extract_model_config(env, configObj);
    }

    out.language  = getStringField(env, cls, inputObj, "language");
    out.plugin_id = getStringField(env, cls, inputObj, "plugin_id");
    // Resolve user-friendly device_id (cpu/gpu/npu/hybrid) via the shared
    // helper so qairt falls back to NPU with a warning instead of erroring.
    const char*              raw_device_id = getStringField(env, cls, inputObj, "device_id");
    jniutils::ResolvedDevice rdev =
        jniutils::resolve_device(out.plugin_id, out.model_name, raw_device_id ? raw_device_id : "");
    out.device_id = rdev.device_id.empty() ? nullptr : jniutils::hold_c_str(rdev.device_id);
    LOGd("device_id resolved: raw=%s, device_id=%s",
        raw_device_id ? raw_device_id : "(null)",
        rdev.device_id.empty() ? "(null)" : rdev.device_id.c_str());
    out.license_id  = getStringField(env, cls, inputObj, "license_id");
    out.license_key = getStringField(env, cls, inputObj, "license_key");

    env->DeleteLocalRef(cls);
    return out;
}

geniex_ASRConfig extract_asr_config(JNIEnv* env, jobject inputObj) {
    geniex_ASRConfig out = {};

    if (!inputObj) {
        LOGe("extract_asr_config: inputObj is null");
        return out;
    }

    jclass cls = env->GetObjectClass(inputObj);
    if (!cls) {
        LOGe("extract_asr_config: GetObjectClass returned null");
        return out;
    }

    // 1. timestamps (String -> const char*)
    out.timestamps = getStringField(env, cls, inputObj, "timestamps");
    // 2. beamSize (Integer -> int32_t)
    out.beam_size = getIntField(env, cls, inputObj, "beamSize");
    // 3. stream (boolean -> bool)
    out.stream = getBoolField(env, cls, inputObj, "stream");
    env->DeleteLocalRef(cls);
    return out;
}

/**
 * Extract ASR stream configuration from Java AsrStreamConfig object.
 * Note: AsrStreamConfig uses primitive types (Float, Int) not nullable (Float?, Int?)
 *       so we must use getPrimitiveFloatField/getPrimitiveIntField.
 */
geniex_ASRStreamConfig extract_asr_stream_config(JNIEnv* env, jobject configObj) {
    geniex_ASRStreamConfig out = {};

    // Set defaults
    out.chunk_duration   = 4.0f;
    out.overlap_duration = 3.0f;
    out.sample_rate      = 16000;
    out.max_queue_size   = 10;
    out.buffer_size      = 512;
    out.timestamps       = nullptr;
    out.beam_size        = 0;

    if (!configObj) {
        return out;  // Return defaults
    }

    jclass cls = env->GetObjectClass(configObj);
    if (!cls) {
        LOGe("extract_asr_stream_config: GetObjectClass returned null");
        return out;
    }

    // Extract primitive fields from AsrStreamConfig (non-nullable Kotlin types)
    out.chunk_duration   = getPrimitiveFloatField(env, cls, configObj, "chunkDuration");
    out.overlap_duration = getPrimitiveFloatField(env, cls, configObj, "overlapDuration");
    out.sample_rate      = getPrimitiveIntField(env, cls, configObj, "sampleRate");
    out.max_queue_size   = getPrimitiveIntField(env, cls, configObj, "maxQueueSize");
    out.buffer_size      = getPrimitiveIntField(env, cls, configObj, "bufferSize");

    // timestamps is String? (nullable) - use existing getStringField
    out.timestamps = getStringField(env, cls, configObj, "timestamps");

    // Handle nullable beamSize (Int? in Kotlin -> boxed Integer)
    jfieldID beamSizeField = env->GetFieldID(cls, "beamSize", "Ljava/lang/Integer;");
    if (beamSizeField) {
        jobject beamSizeObj = env->GetObjectField(configObj, beamSizeField);
        if (beamSizeObj) {
            jclass    integerCls     = env->FindClass("java/lang/Integer");
            jmethodID intValueMethod = env->GetMethodID(integerCls, "intValue", "()I");
            out.beam_size            = env->CallIntMethod(beamSizeObj, intValueMethod);
            env->DeleteLocalRef(integerCls);
            env->DeleteLocalRef(beamSizeObj);
        }
    } else {
        // Clear any pending exception from GetFieldID if beamSize doesn't exist
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
    }

    env->DeleteLocalRef(cls);
    return out;
}

// Streaming test callback data
struct StreamingTestData {
    std::string              accumulated_text;
    std::vector<std::string> streaming_updates;
    bool                     callback_called = false;

    void reset() {
        accumulated_text.clear();
        streaming_updates.clear();
        callback_called = false;
    }
};

// Callback function for streaming transcription
void streaming_transcription_callback(const char* text, void* user_data) {
    auto* test_data = static_cast<StreamingTestData*>(user_data);
    if (text && strlen(text) > 0) {
        test_data->callback_called = true;
        test_data->streaming_updates.push_back(std::string(text));
        test_data->accumulated_text = text;  // Keep the latest complete transcription
    }
}

// Simple WAV file reader for testing purposes
struct WavHeader {
    char     riff[4];
    uint32_t chunk_size;
    char     wave[4];
    char     fmt[4];
    uint32_t fmt_chunk_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char     data[4];
    uint32_t data_size;
};

std::vector<float> load_wav_as_float32(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        LOGe("Failed to open WAV file: %s", file_path.c_str());
        return {};
    }

    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (std::string(header.riff, 4) != "RIFF" || std::string(header.wave, 4) != "WAVE") {
        LOGe("Invalid WAV file format: %s", file_path.c_str());
        return {};
    }

    LOGd("WAV file info - Sample rate: %udHz, Channels: %ud, Bits: %ud, Data size: %ud bytes",
        header.sample_rate,
        header.num_channels,
        header.bits_per_sample,
        header.data_size);

    std::vector<float> audio_data;

    if (header.bits_per_sample == 16) {
        // Read 16-bit PCM and convert to float32
        std::vector<int16_t> pcm_data(header.data_size / sizeof(int16_t));
        file.read(reinterpret_cast<char*>(pcm_data.data()), header.data_size);

        audio_data.reserve(pcm_data.size());
        for (int16_t sample : pcm_data) {
            audio_data.push_back(static_cast<float>(sample) / 32768.0f);
        }
    } else if (header.bits_per_sample == 32) {
        // Read 32-bit float directly
        audio_data.resize(header.data_size / sizeof(float));
        file.read(reinterpret_cast<char*>(audio_data.data()), header.data_size);
    } else {
        LOGe("Unsupported bits per sample: %ud", header.bits_per_sample);
        return {};
    }

    LOGd("Loaded %zu audio samples from WAV file", audio_data.size());
    return audio_data;
}

geniex_AsrTranscribeInput extract_asr_transcribe_input(
    JNIEnv* env, jclass cls, jobject inputObj, geniex_ASRConfig* config) {
    geniex_AsrTranscribeInput out = {};

    out.audio_path = getStringField(env, cls, inputObj, "audioPath");
    out.language   = getStringField(env, cls, inputObj, "language");

    jobject configObj = getObjectField(env, cls, inputObj, "config", "Lcom/geniex/sdk/bean/AsrConfig;");
    if (!configObj) {
        out.config = nullptr;
    } else {
        *config    = extract_asr_config(env, configObj);
        out.config = config;
    }

    return out;
}

// ASR create - Initialize ASR with configuration
JNIEXPORT jlong JNICALL Java_com_geniex_sdk_jni_Asr_create(JNIEnv* env, jobject /*thiz*/, jobject asrCreateInputObj) {
    try {
        geniex_AsrCreateInput input = extract_asr_create_input(env, asrCreateInputObj);

        geniex_ASR* handle;
        LOGd("[JNI] create() geniex_asr_create called");
        int32_t err = geniex_asr_create(&input, &handle);
        if (err != GENIEX_SUCCESS || !handle) {
            LOGe("[JNI] geniex_asr_create failed, error code: %d", err);
            throw_runtime_exception(env, "Asr create failed, error code: %d", err);
            return 0;
        }
        LOGd("[JNI] create() geniex_asr_create returned handle=%p", handle);
        return reinterpret_cast<jlong>(handle);
    } catch (const std::exception& e) {
        LOGe("[JNI] create() exception: %s", e.what());
        return 0;
    }
}

// ASR destroy - Clean up ASR resources
JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_Asr_destroy(JNIEnv*, jobject, jlong handle) {
    LOGd("[JNI] geniex_asr_destroy called, handle=%p", (void*)handle);
    if (handle) {
        int32_t result = geniex_asr_destroy(reinterpret_cast<geniex_ASR*>(handle));
        if (result != GENIEX_SUCCESS) {
            LOGe("[JNI] geniex_asr_destroy failed, error code: %d", result);
        }
        return result;
    }
    return 0;
}

// ASR transcribe
JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Asr_transcribe(
    JNIEnv* env, jobject, jlong handle, jobject transcribeInputObj) {
    if (!transcribeInputObj) {
        LOGe("transcribeInputObj is null");
        return nullptr;
    }
    if (!handle) {
        LOGe("asr has been destroyed");
        return nullptr;
    }
    jclass cls = env->GetObjectClass(transcribeInputObj);
    if (!cls) {
        LOGe("[JNI] transcribeInputObj: GetObjectClass returned null");
        return nullptr;
    }

    geniex_ASRConfig          config;
    geniex_AsrTranscribeInput input = extract_asr_transcribe_input(env, cls, transcribeInputObj, &config);
    env->DeleteLocalRef(cls);

    geniex_AsrTranscribeOutput output = {};
    int32_t                    err    = geniex_asr_transcribe((geniex_ASR*)handle, &input, &output);

    if (err < 0 || !output.result.transcript) {
        throw_runtime_exception(env, "Asr transcribe failed, error code: %d", err);
        return nullptr;
    }

    // result
    jclass    resultClsOut = env->FindClass("com/geniex/sdk/bean/AsrResult");
    jmethodID resultCtor =
        env->GetMethodID(resultClsOut, "<init>", "(Ljava/lang/String;Ljava/util/List;Ljava/util/List;)V");
    LOGi("JNI ASR Transcribe result: transcript='%s'", output.result.transcript ? output.result.transcript : "");
    jobject resultObj = env->NewObject(resultClsOut,
        resultCtor,
        env->NewStringUTF(output.result.transcript),
        create_float_list(env, output.result.confidence_scores, output.result.confidence_count),
        create_float_list(env, output.result.timestamps, output.result.timestamp_count));
    geniex_free(output.result.transcript);
    geniex_free(output.result.confidence_scores);
    geniex_free(output.result.timestamps);
    // profile_data
    jobject profileDataObj = jniutils::extract_profiling_data(env, output.profile_data);
    // output
    jclass    clsOut = env->FindClass("com/geniex/sdk/bean/AsrTranscribeOutput");
    jmethodID ctor =
        env->GetMethodID(clsOut, "<init>", "(Lcom/geniex/sdk/bean/AsrResult;Lcom/geniex/sdk/bean/ProfilingData;)V");
    jobject result = env->NewObject(clsOut, ctor, resultObj, profileDataObj);
    return result;
}

// ASR listSupportedLanguages
JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Asr_listSupportedLanguages(JNIEnv* env, jobject, jlong handle) {
    geniex_AsrListSupportedLanguagesInput  input = {};
    geniex_AsrListSupportedLanguagesOutput out   = {};
    int32_t result = geniex_asr_list_supported_languages(reinterpret_cast<const geniex_ASR*>(handle), &input, &out);
    if (result == GENIEX_SUCCESS) {
        return create_string_list(env, out.language_codes, out.language_count);
    } else {
        LOGe("get support language failed and error code: %d", result);
    }
    return nullptr;
}

// =============================================================================
// ASR Streaming JNI Functions
// =============================================================================

/**
 * Begin ASR streaming session.
 *
 * JNI signature: streamBegin(handle: Long, input: AsrStreamBeginInput, callback: AsrTranscriptionCallback):
 * AsrStreamBeginOutput
 */
JNIEXPORT jobject JNICALL Java_com_geniex_sdk_jni_Asr_streamBegin(
    JNIEnv* env, jobject, jlong handle, jobject streamBeginInputObj, jobject callbackObj) {
    LOGd("[JNI] streamBegin() called, handle=%p", (void*)handle);

    // Validate inputs
    if (!handle) {
        LOGe("[JNI] streamBegin: handle is null");
        throw_runtime_exception(env, "ASR handle is null");
        return nullptr;
    }

    if (!callbackObj) {
        LOGe("[JNI] streamBegin: callback is null");
        throw_runtime_exception(env, "AsrTranscriptionCallback is required");
        return nullptr;
    }

    void* h = (void*)handle;

    // Create and initialize callback context
    auto* ctx = new AsrStreamingCallbackCtx();
    if (!asr_init_callback_ctx(env, callbackObj, ctx)) {
        LOGe("[JNI] streamBegin: Failed to initialize callback context");
        delete ctx;
        throw_runtime_exception(env, "Failed to initialize streaming callback");
        return nullptr;
    }

    // Extract input parameters
    geniex_AsrStreamBeginInput begin_input{};
    geniex_ASRStreamConfig     stream_config{};

    if (streamBeginInputObj) {
        jclass inputCls = env->GetObjectClass(streamBeginInputObj);
        if (inputCls) {
            // Extract language
            begin_input.language = getStringField(env, inputCls, streamBeginInputObj, "language");

            // Extract stream config
            jobject streamConfigObj = getObjectField(
                env, inputCls, streamBeginInputObj, "streamConfig", "Lcom/geniex/sdk/bean/AsrStreamConfig;");
            if (streamConfigObj) {
                stream_config             = extract_asr_stream_config(env, streamConfigObj);
                begin_input.stream_config = &stream_config;
                env->DeleteLocalRef(streamConfigObj);
            }

            env->DeleteLocalRef(inputCls);
        }
    }

    // Set up native callback
    begin_input.on_transcription = asr_native_transcription_callback;
    begin_input.user_data        = ctx;

    // Call native API
    geniex_AsrStreamBeginOutput begin_output{};
    int32_t res = geniex_asr_stream_begin(reinterpret_cast<geniex_ASR*>(handle), &begin_input, &begin_output);

    if (res != GENIEX_SUCCESS) {
        LOGe("[JNI] geniex_asr_stream_begin failed, error code: %d", res);
        asr_dispose_callback_ctx(env, ctx);
        delete ctx;
        throw_runtime_exception(env, "ASR stream begin failed, error code: %d", res);
        return nullptr;
    }

    // Store callback context for later cleanup
    {
        std::lock_guard<std::mutex> lock(g_asrStreamingMutex);
        // Clean up existing context if any
        if (g_asrStreamingContexts.count(h)) {
            auto* old_ctx = g_asrStreamingContexts[h];
            asr_dispose_callback_ctx(env, old_ctx);
            delete old_ctx;
        }
        g_asrStreamingContexts[h] = ctx;
    }

    LOGd("[JNI] streamBegin() success");

    // Create and return AsrStreamBeginOutput
    jclass outputCls = env->FindClass("com/geniex/sdk/bean/AsrStreamBeginOutput");
    if (!outputCls) {
        LOGe("[JNI] Failed to find AsrStreamBeginOutput class");
        return nullptr;
    }

    jmethodID ctor = env->GetMethodID(outputCls, "<init>", "(I)V");
    if (!ctor) {
        LOGe("[JNI] Failed to find AsrStreamBeginOutput constructor");
        return nullptr;
    }

    jobject result = env->NewObject(outputCls, ctor, (jint)res);
    return result;
}

/**
 * Push audio data to the streaming ASR for processing.
 *
 * JNI signature: streamPushAudio(handle: Long, audioData: FloatArray): Int
 */
JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_Asr_streamPushAudio(
    JNIEnv* env, jobject, jlong handle, jfloatArray audioDataArray) {
    if (!handle) {
        LOGe("[JNI] streamPushAudio: handle is null");
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    if (!audioDataArray) {
        LOGe("[JNI] streamPushAudio: audioData is null");
        return GENIEX_ERROR_ASR_STREAM_INVALID_AUDIO;
    }

    // Get array length and data
    jsize length = env->GetArrayLength(audioDataArray);
    if (length <= 0) {
        LOGe("[JNI] streamPushAudio: audioData is empty");
        return GENIEX_ERROR_ASR_STREAM_INVALID_AUDIO;
    }

    // Get float array elements (pinned or copied)
    jfloat* audioData = env->GetFloatArrayElements(audioDataArray, nullptr);
    if (!audioData) {
        LOGe("[JNI] streamPushAudio: Failed to get float array elements");
        return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
    }

    // Prepare input structure
    geniex_AsrStreamPushAudioInput push_input{};
    push_input.audio_data = audioData;
    push_input.length     = static_cast<int32_t>(length);

    // Call native API
    int32_t res = geniex_asr_stream_push_audio(reinterpret_cast<geniex_ASR*>(handle), &push_input);

    // Release array elements (no copy back needed as we don't modify)
    env->ReleaseFloatArrayElements(audioDataArray, audioData, JNI_ABORT);

    if (res != GENIEX_SUCCESS && res < 0) {
        LOGe("[JNI] geniex_asr_stream_push_audio failed, error code: %d", res);
    }

    return res;
}

void test_asr_streaming_transcription(geniex_ASR* asr) {
    const char* test_audio_path = "/data/local/tmp/geniex/modelfiles/assets/OSR_us_000_0010_16k.wav";
    // Check if test audio file exists
    if (!std::filesystem::exists(test_audio_path)) {
        LOGe("audio file not found: %s", test_audio_path);
        return;
    }
    // Load audio data as float32 samples
    std::vector<float> audio_data = load_wav_as_float32(test_audio_path);
    if (audio_data.empty()) {
        LOGe("audio file can not be transfer to float32.");
        return;
    }

    // Set up streaming callback data
    StreamingTestData test_data;
    test_data.reset();
    LOGd("before geniex_asr_stream_begin");
    // Begin streaming with consolidated configuration
    geniex_AsrStreamBeginInput begin_input{};
    begin_input.language         = "en";
    begin_input.on_transcription = streaming_transcription_callback;
    begin_input.user_data        = &test_data;

    geniex_AsrStreamBeginOutput begin_output{};
    int32_t                     res = geniex_asr_stream_begin(asr, &begin_input, &begin_output);
    if (res != GENIEX_SUCCESS) {
        LOGe("geniex_asr_stream_begin failed and error code:: %d", res);
        return;
    }

    // Simulate streaming by pushing audio data in chunks
    // Audio push buffer: 512 float32 samples at 16kHz (~32ms per chunk)
    const int32_t chunk_size    = 512;
    int32_t       total_samples = static_cast<int32_t>(audio_data.size());
    LOGd("Starting to stream %d total samples in chunks of %d", total_samples, chunk_size);

    for (int32_t offset = 0; offset < total_samples; offset += chunk_size) {
        int32_t samples_to_send = std::min(chunk_size, total_samples - offset);

        geniex_AsrStreamPushAudioInput push_input{};
        push_input.audio_data = audio_data.data() + offset;
        push_input.length     = samples_to_send;

        res = geniex_asr_stream_push_audio(asr, &push_input);
        // Silent check - only break if there's an error
        if (res < 0) {
            LOGe("Failed to push audio chunk: %s", geniex_get_error_message(static_cast<geniex_ErrorCode>(res)));
            break;
        }

        // Small delay to simulate real-time streaming (reduced for smaller chunks)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // Stop streaming gracefully
    geniex_AsrStreamStopInput stop_input{};
    stop_input.graceful = true;
    res                 = geniex_asr_stream_stop(asr, &stop_input);
    if (res != GENIEX_SUCCESS) {
        LOGe("geniex_asr_stream_stop failed and error code:: %d", res);
        return;
    }

    // Wait longer for processing to complete (allow ANE plugin to process all audio)
    LOGd("Waiting for transcription processing to complete...");
    std::this_thread::sleep_for(std::chrono::seconds(3));

    LOGd("Final streaming transcription: %s", test_data.accumulated_text.c_str());
    LOGd("Received %zu streaming updates", test_data.streaming_updates.size());

    // Print all streaming updates
    for (size_t i = 0; i < test_data.streaming_updates.size(); i++) {
        LOGd("Streaming update %zu: %s", i + 1, test_data.streaming_updates[i].c_str());
    }
}

/**
 * Stop the streaming ASR session.
 *
 * JNI signature: streamStop(handle: Long, graceful: Boolean): Int
 */
JNIEXPORT jint JNICALL Java_com_geniex_sdk_jni_Asr_streamStop(JNIEnv* env, jobject, jlong handle, jboolean graceful) {
    LOGd("[JNI] streamStop() called, handle=%p, graceful=%d", (void*)handle, graceful);

    if (!handle) {
        LOGe("[JNI] streamStop: handle is null");
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    void* h = (void*)handle;

    // Prepare stop input
    geniex_AsrStreamStopInput stop_input{};
    stop_input.graceful = (graceful == JNI_TRUE);

    // Call native API
    int32_t res = geniex_asr_stream_stop(reinterpret_cast<geniex_ASR*>(handle), &stop_input);

    if (res != GENIEX_SUCCESS) {
        LOGe("[JNI] geniex_asr_stream_stop failed, error code: %d", res);
    }

    // Clean up callback context
    {
        std::lock_guard<std::mutex> lock(g_asrStreamingMutex);
        if (g_asrStreamingContexts.count(h)) {
            auto* ctx = g_asrStreamingContexts[h];
            asr_dispose_callback_ctx(env, ctx);
            delete ctx;
            g_asrStreamingContexts.erase(h);
            LOGd("[JNI] streamStop: Cleaned up callback context");
        }
    }

    LOGd("[JNI] streamStop() completed with result: %d", res);
    return res;
}

}  // extern "C"
