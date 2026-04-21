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

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    setup_redirect_stdout_stderr();

    // Verify stdout/stderr redirection is working
    fprintf(stdout, "=== GENIEX SDK: stdout redirection test - this should appear in logcat ===\n");
    fprintf(stderr, "=== GENIEX SDK: stderr redirection test - this should appear in logcat ===\n");
    fflush(stdout);
    fflush(stderr);

    setenv("GENIEX_TOKEN",
        "key/"
        "eyJhY2NvdW50Ijp7ImlkIjoiNDI1Y2JiNWQtNjk1NC00NDYxLWJiOWMtYzhlZjBiY2JlYzA2In0sInByb2R1Y3QiOnsiaWQiOiJkYjI4ZTNmYy"
        "1mMjU4LTQ4ZTctYmNkYi0wZmE4YjRkYTJhNWYifSwicG9saWN5Ijp7ImlkIjoiMmYyOWQyMjctNDVkZS00MzQ3LTg0YTItMjUwNTYwMmEzYzMy"
        "IiwiZHVyYXRpb24iOjMxMTA0MDAwMH0sInVzZXIiOnsiaWQiOiI3MGE2YzA4NS1jYjc3LTQ3YmEtOWUxNC1lNjFjYTA2ZThmZjUiLCJlbWFpbC"
        "I6ImFsYW40QG5leGE0YWkuY29tIn0sImxpY2Vuc2UiOnsiaWQiOiI4OTlhZGQ2NS1lOTI2LTQ2M2ItODllNi0xMjc0NzM3ZjA1MzYiLCJjcmVh"
        "dGVkIjoiMjAyNS0wOS0wNlQwMDo1MzozNi4yMDNaIiwiZXhwaXJ5IjoiMjAzNS0xMi0zMVQyMzo1OTo1OS4wMDBaIn19."
        "BXoUHIEzFMuuZbBT7RvsKO9nTi5950C6kHO64blF7XBnfKvZ6ClA8a55tmszI1ZWdngzpNFTzMM5PV5euuzMCA==",
        1);
    geniex_init();
    return JNI_VERSION_1_6;
}

using namespace jniutils;

extern "C" JNIEXPORT jint JNICALL Java_com_geniex_sdk_NexaSdk_registerPlugin(
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
