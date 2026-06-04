cmake_minimum_required(VERSION 3.10)
project(npu_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

get_filename_component(PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)

message(STATUS "NPU JNI - SDK_INSTALL_DIR: ${SDK_INSTALL_DIR}")

set(JNI_SOURCES
        ${PARENT_DIR}/android_utils.cpp
        ${PARENT_DIR}/geniex_sdk.cpp
        ${PARENT_DIR}/jniutils.cpp
        ${PARENT_DIR}/jni_cb.cpp
        ${PARENT_DIR}/llm_bridge_jni.cpp
        ${PARENT_DIR}/vlm_bridge_jni.cpp
        ${PARENT_DIR}/model_manager_jni.cpp
)

add_library(npu_jni SHARED ${JNI_SOURCES})

target_include_directories(npu_jni
        PRIVATE
        ${PARENT_DIR}
        ${SDK_INCLUDE_DIR}
)

target_link_libraries(npu_jni
        PRIVATE
        log
        ${geniex_bridge}
)

set_target_properties(npu_jni PROPERTIES
        OUTPUT_NAME "npu_jni"
)
