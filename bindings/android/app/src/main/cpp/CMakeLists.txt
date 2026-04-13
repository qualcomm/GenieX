project(android_project)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/../../../../../.." ABSOLUTE)
set(geniex_bridge   "${PROJECT_ROOT}/build-android/out/libgeniex_bridge.so")
set(npu_so_dir   "${PROJECT_ROOT}/build-android/out/npu")
set(htp_so_dir   "${PROJECT_ROOT}/build-android/out/npu/htp-files")

#option(BUILD_NPU_BRIDGE "Build NPU bridge (QNN-based)" ON)
#message(STATUS "BUILD_NPU_BRIDGE: ${BUILD_NPU_BRIDGE}")

add_subdirectory(qnn)

## skip llama-cpp in Android JNI bridge for now
#add_subdirectory(llama-cpp)