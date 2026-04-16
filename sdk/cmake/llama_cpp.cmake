if(TARGET llama)
    return()
endif()

set(GGML_BLAS OFF)
set(GGML_NATIVE OFF)

if(GENIEX_DL)
    set(GGML_BACKEND_DL ON CACHE BOOL "Enable ggml dynamic loading")
else()
    set(GGML_BACKEND_DL OFF CACHE BOOL "Disable ggml dynamic loading")
endif()

set(LLAMA_BUILD_COMMON ON)
set(LLAMA_CURL OFF)
set(LLAMA_BUILD_TESTS OFF)
set(LLAMA_BUILD_TOOLS ON)
set(LLAMA_BUILD_EXAMPLES OFF)
set(LLAMA_BUILD_SERVER OFF)
set(BUILD_SHARED_LIBS ON)

set(GENIEX_LLAMA_CPP ON)

# Find OpenSSL first without CMAKE_SYSTEM_PATH to prefer local installations
if (CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(OpenCL_INCLUDE_DIR "$ENV{OPENCL_SDK_ROOT}/include")
    set(OpenCL_LIBRARY "$ENV{OPENCL_SDK_ROOT}/lib/OpenCL.lib")
endif()
# add opencl lib for linux arm64
# if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
#     set(OpenCL_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/linux-arm64/include")
#     set(OpenCL_LIBRARY "${CMAKE_CURRENT_LIST_DIR}/linux-arm64/lib/OpenCL.so")
# endif()
# add opencl lib for android aarch64
# if (CMAKE_SYSTEM_NAME STREQUAL "Android" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
#     set(OpenCL_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/android-arm64/include")
#     set(OpenCL_LIBRARY "${CMAKE_CURRENT_LIST_DIR}/android-arm64/lib/libOpenCL.so")
# endif()
set(OPENCL_SDK_ROOT "$ENV{OPENCL_SDK_ROOT}")
set(HEXAGON_SDK_ROOT "$ENV{HEXAGON_SDK_ROOT}")

set(GENIEX_LLAMA_CPP_DIR "${CMAKE_SOURCE_DIR}/../third-party/llama.cpp"
    CACHE PATH "Path to llama.cpp source directory")

# EXCLUDE_FROM_ALL suppresses third-party install() rules.
# Targets still build because SDK plugins link against them.
add_subdirectory(${GENIEX_LLAMA_CPP_DIR} ${CMAKE_BINARY_DIR}/third-party/llama.cpp EXCLUDE_FROM_ALL)

# Export list of llama.cpp libraries for plugin installation
set(LLAMA_LIBS common mtmd llama ggml ggml-base ggml-cpu ggml-opencl ggml-hexagon)
