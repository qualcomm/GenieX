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

set(GENIEX_LLAMA_CPP_DIR "${CMAKE_SOURCE_DIR}/../third-party/llama.cpp"
    CACHE PATH "Path to llama.cpp source directory")

# EXCLUDE_FROM_ALL suppresses third-party install() rules.
# Targets still build because SDK plugins link against them.
add_subdirectory(${GENIEX_LLAMA_CPP_DIR} ${CMAKE_BINARY_DIR}/third-party/llama.cpp EXCLUDE_FROM_ALL)

# Export list of llama.cpp libraries for plugin installation
set(LLAMA_LIBS common mtmd llama ggml ggml-base ggml-cpu ggml-opencl ggml-hexagon)
