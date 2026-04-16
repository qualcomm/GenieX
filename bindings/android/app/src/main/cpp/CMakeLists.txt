project(android_project)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use SDK install directory structure
get_filename_component(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/../../../../../.." ABSOLUTE)
set(SDK_INSTALL_DIR "${PROJECT_ROOT}/build-android/install")
set(geniex_bridge   "${SDK_INSTALL_DIR}/lib/libgeniex.so")
set(SDK_INCLUDE_DIR "${SDK_INSTALL_DIR}/include")

add_subdirectory(qnn)

## skip llama-cpp in Android JNI bridge for now
#add_subdirectory(llama-cpp)