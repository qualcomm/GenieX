cmake_minimum_required(VERSION 3.10)
project(llm_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)


get_filename_component(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/../../../../../.." ABSOLUTE)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
get_filename_component(PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
message(STATUS "PARENT_DIR：${PARENT_DIR}")

# === Include & Source Paths ===
set(ML_LLM_LLAMA_SRC "${PROJECT_ROOT}/src")
set(ML_LLM_LLAMA_INCLUDE "${PROJECT_ROOT}/plugins/llama-cpp/include")
set(ML_COMMON_INCLUDE     "${PROJECT_ROOT}/include")
set(LLAMA_CPP_INCLUDE     "${PROJECT_ROOT}/third-party/llama.cpp/include")
set(LLAMA_CPP_COMMON     "${PROJECT_ROOT}/third-party/llama.cpp/common")
set(LLAMA_CPP_GGML     "${PROJECT_ROOT}/third-party/llama.cpp/ggml/include")
set(LLAMA_CPP_MTMD     "${PROJECT_ROOT}/third-party/llama.cpp/tools/mtmd")
set(LLAMA_C_LIB_INCLUDE   "${PROJECT_ROOT}/backends/llama-cpp/c-lib/include")

# === llama.cpp so path（for Android build）===
set(LLAMA_CPP_DIR        "${PROJECT_ROOT}/third-party/llama.cpp")
set(LLAMA_SHARED_LIB     "${PROJECT_ROOT}/build-android/out/geniex_llama_cpp/libllama.so")
set(LLAMA_GGML_LIB       "${PROJECT_ROOT}/build-android/out/geniex_llama_cpp/libggml.so")
set(LLAMA_MTMD_LIB  "${PROJECT_ROOT}/build-android/out/geniex_llama_cpp/libmtmd.so")
set(LLAMA_GGML_CPU_LIB  "${PROJECT_ROOT}/build-android/out/geniex_llama_cpp/libggml-cpu.so")
#set(LLAMA_GGML_CPU_LIB  "${PROJECT_ROOT}/third-party/llama.cpp/build-android/bin/libggml-cpu.so")
set(LLAMA_GGML_BASE_LIB  "${PROJECT_ROOT}/build-android/out/geniex_llama_cpp/libggml-base.so")
set(LLAMA_COMMON_LIB "${LLAMA_CPP_DIR}/build-android/common/${_cfg}/libcommon.a")
set(LLAMA_PLUGIN   "${PROJECT_ROOT}/build-android/out/geniex_llama_cpp/libgeniex_plugin.so")
set(GENIEX_BRIDGE   "${PROJECT_ROOT}/build-android/out/libgeniex_bridge.so")

add_definitions(-DBRIDGE_VERSION=\"1.0.0\")
include_directories(${PARENT_DIR})

set(ML_LLM_LLAMA_SOURCES
        ${ML_LLM_LLAMA_SRC}/llm.cpp
        ${ML_LLM_LLAMA_SRC}/registry.cpp
        ${ML_LLM_LLAMA_SRC}/ml.cpp
        ${ML_LLM_LLAMA_SRC}/vlm.cpp
        #        ${ML_LLM_LLAMA_INCLUDE}/ml-embedding.cpp
        #        ${ML_LLM_LLAMA_INCLUDE}/ml-common.cpp
        #        ${ML_LLM_LLAMA_INCLUDE}/ml-rerank.cpp
        #        ${ML_LLM_LLAMA_INCLUDE}/ml-profile.cpp

)

add_library(llm_jni SHARED
        ${PARENT_DIR}/llm_bridge_jni.cpp
        ${PARENT_DIR}/multi_modal_bridge_jni.cpp
        ${PARENT_DIR}/jni_cb.cpp
        #        ${PARENT_DIR}/embedder_bridge_jni.cpp
        #        ${PARENT_DIR}/reranker_bridge_jni.cpp
        ${PARENT_DIR}/jniutils.cpp
        ${ML_LLM_LLAMA_SOURCES}
)
target_compile_definitions(llm_jni PRIVATE GENIEX_DL GENIEX_ANDROID)
target_include_directories(llm_jni
        PRIVATE
        ${ML_LLM_LLAMA_INCLUDE}
        ${ML_LLM_LLAMA_SRC}
        ${ML_COMMON_INCLUDE}
        ${LLAMA_CPP_INCLUDE}
        ${LLAMA_C_LIB_INCLUDE}
        ${LLAMA_CPP_COMMON}
        ${LLAMA_CPP_GGML}
        ${LLAMA_CPP_MTMD}
)

target_link_libraries(llm_jni
        PRIVATE
        log
        PUBLIC ${LLAMA_SHARED_LIB}
        PUBLIC ${LLAMA_GGML_LIB}
        PUBLIC ${LLAMA_MTMD_LIB}
        PUBLIC ${LLAMA_GGML_BASE_LIB}
        PUBLIC ${LLAMA_COMMON_LIB}
        PUBLIC ${LLAMA_PLUGIN}
        PUBLIC ${GENIEX_BRIDGE}
        PUBLIC ${LLAMA_GGML_CPU_LIB}
)

set_target_properties(llm_jni PROPERTIES
        OUTPUT_NAME "llm_jni"
)