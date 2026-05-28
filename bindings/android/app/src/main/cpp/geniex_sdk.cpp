#include <android/log.h>
#include <dlfcn.h>
#include <jni.h>
#include <pthread.h>
#include <sys/stat.h>  // For chmod()
#include <unistd.h>    // For access()
#include <unistd.h>

#include <string>

#include "android_utils.h"
#include "geniex.h"
#include "jniutils.h"

using namespace jniutils;
using namespace geniex_android_sdk;

// Route SDK log messages to Android logcat under the "GenieXSdk" tag.
static void android_sdk_log_to_logcat(geniex_LogLevel level, const char* msg) {
    int prio;
    switch (level) {
        case GENIEX_LOG_LEVEL_TRACE:
            prio = ANDROID_LOG_VERBOSE;
            break;
        case GENIEX_LOG_LEVEL_DEBUG:
            prio = ANDROID_LOG_DEBUG;
            break;
        case GENIEX_LOG_LEVEL_INFO:
            prio = ANDROID_LOG_INFO;
            break;
        case GENIEX_LOG_LEVEL_WARN:
            prio = ANDROID_LOG_WARN;
            break;
        case GENIEX_LOG_LEVEL_ERROR:
            prio = ANDROID_LOG_ERROR;
            break;
        default:
            prio = ANDROID_LOG_DEFAULT;
            break;
    }
    __android_log_print(prio, TAG, "%s", msg ? msg : "");
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    setup_redirect_stdout_stderr();

    // Verify stdout/stderr redirection is working
    fprintf(stdout, "=== GENIEX SDK: stdout redirection test - this should appear in logcat ===\n");
    fprintf(stderr, "=== GENIEX SDK: stderr redirection test - this should appear in logcat ===\n");
    fflush(stdout);
    fflush(stderr);

    // Route SDK logs to logcat before geniex_init() so initialization logs surface too.
    geniex_set_log(android_sdk_log_to_logcat);

    geniex_init();
    return JNI_VERSION_1_6;
}

using namespace jniutils;

extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_GenieXSdk_registerPlugin(
    JNIEnv* env, jobject thiz, jstring plugin_lib_path) {
    // Get the native library path from the application context
    std::string plugin_lib_path_str = jstring2str(env, plugin_lib_path);

    void* pluginSo = dlopen(plugin_lib_path_str.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!pluginSo) {
        LOGe("%s", dlerror());
    }

    void* plugin_id_func     = dlsym(pluginSo, "plugin_id");
    void* create_plugin_func = dlsym(pluginSo, "create_plugin");
    return geniex_register_plugin((geniex_plugin_id_func)plugin_id_func, (geniex_create_plugin_func)create_plugin_func);
}

extern "C" JNIEXPORT jstring JNICALL Java_com_geniex_sdk_GenieXSdk_getPluginVersion(
    JNIEnv* env, jobject thiz, jstring plugin_id) {
    std::string id     = jstring2str(env, plugin_id);
    const char* result = geniex_get_plugin_version(id.c_str());
    return env->NewStringUTF(result ? result : "");
}
