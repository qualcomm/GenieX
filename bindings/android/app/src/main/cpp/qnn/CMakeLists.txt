cmake_minimum_required(VERSION 3.10)
project(npu_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Get project root directory
get_filename_component(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/../../../../../.." ABSOLUTE)
get_filename_component(PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)

message(STATUS "NPU JNI (QNN-based) - PROJECT_ROOT: ${PROJECT_ROOT}")
message(STATUS "NPU JNI (QNN-based) - PARENT_DIR: ${PARENT_DIR}")

# Use SDK install directory for headers
set(PLUGIN_INCLUDE_DIR "${SDK_INCLUDE_DIR}")
set(QNN_LIBS_DIR "${PROJECT_ROOT}/plugins/qnn/libs/android-aarch64")  # QNN library path

# JNI wrapper sources
set(JNI_SOURCES
        ${PARENT_DIR}/android_utils.cpp
        ${PARENT_DIR}/geniex_sdk.cpp
        ${PARENT_DIR}/jniutils.cpp
        ${PARENT_DIR}/jni_cb.cpp
        ${PARENT_DIR}/llm_bridge_jni.cpp
        ${PARENT_DIR}/vlm_bridge_jni.cpp
        ${PARENT_DIR}/asr_bridge_jni.cpp
        ${PARENT_DIR}/cv_bridge_jni.cpp
        ${PARENT_DIR}/embedder_bridge_jni.cpp
        ${PARENT_DIR}/reranker_bridge_jni.cpp
        ${PARENT_DIR}/img_gen_jni.cpp
        ${PARENT_DIR}/tts_bridge_jni.cpp
)

# Create JNI wrapper library
add_library(npu_jni SHARED ${JNI_SOURCES})

# Include directories - only public plugin API
target_include_directories(npu_jni
        PRIVATE
        ${PARENT_DIR}
        ${PLUGIN_INCLUDE_DIR}
)

# Link against geniex_bridge library and QNN libs (internal)
target_link_directories(npu_jni PRIVATE
        ${QNN_LIBS_DIR}
        ${PROJECT_ROOT}/build-android/src
)

# 查找目录中的所有库文件
file(GLOB HTP_LIBRARY_FILES
        "${htp_so_dir}/*.so"
)
# 过滤掉不需要的库
set(FILTERED_LIBS)
foreach(lib ${HTP_LIBRARY_FILES})
    # 排除包含特定关键词的库
    if(NOT lib MATCHES "libCalculator_skel.so" AND
#            NOT lib MATCHES ".*[Dd]ebug.*" AND
            NOT lib MATCHES "libQnnHexagonSkel_dspApp.so" AND
            NOT lib MATCHES "libQnnHtpV79.so" AND
            NOT lib MATCHES "libQnnHtpV79Skel.so" AND
            NOT lib MATCHES "libQnnNetRunDirectV79Skel.so"
    )
        list(APPEND FILTERED_LIBS ${lib})
    endif()
endforeach()

file(GLOB NPU_LIBRARY_FILES
        "${npu_so_dir}/*.so"
)

target_link_libraries(npu_jni
        PRIVATE
        log
        # Link geniex_bridge (provides ml_* API functions from ml.h)
        PUBLIC
#        ${FILTERED_LIBS}
#        ${NPU_LIBRARY_FILES}
        ${geniex_bridge}
)

set_target_properties(npu_jni PROPERTIES
        OUTPUT_NAME "npu_jni"
)
