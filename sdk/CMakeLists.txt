cmake_minimum_required(VERSION 3.16)
# C compiler is required by native dependencies.
project(geniex C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set RPATH configuration for all targets to use relative paths
set(CMAKE_BUILD_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

# Set UTF-8 encoding for MSVC
add_compile_options("$<$<AND:$<C_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/utf-8>")
add_compile_options("$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/utf-8>")

if(WIN32)
    add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

# Convert CMAKE_TOOLCHAIN_FILE to absolute path if set
if(DEFINED CMAKE_TOOLCHAIN_FILE AND NOT IS_ABSOLUTE "${CMAKE_TOOLCHAIN_FILE}")
    get_filename_component(ABSOLUTE_TOOLCHAIN_PATH "${CMAKE_TOOLCHAIN_FILE}" ABSOLUTE)
    set(CMAKE_TOOLCHAIN_FILE ${ABSOLUTE_TOOLCHAIN_PATH})
endif()

set(GENIEX_VERSION "v0.0.0" CACHE STRING "Geniex library version")

# ===============================================================================
# Build Options
# ===============================================================================

option(GENIEX_PLUGIN_LLAMA_CPP  "Enable llama.cpp backend"       ON)
option(GENIEX_PLUGIN_QAIRT      "Enable QAIRT backend"           ON)
option(GENIEX_DEBUG             "Enable debug mode"              ON)
option(GENIEX_TEST              "Enable unit tests"              ON)
option(GENIEX_MODEL_MANAGER     "Enable model manager"           OFF)

message(STATUS "=============== Options ===============")
message(STATUS "GENIEX_VERSION:          ${GENIEX_VERSION}")
message(STATUS "GENIEX_PLUGIN_LLAMA_CPP: ${GENIEX_PLUGIN_LLAMA_CPP}")
message(STATUS "GENIEX_PLUGIN_QAIRT:     ${GENIEX_PLUGIN_QAIRT}")
message(STATUS "GENIEX_DEBUG:            ${GENIEX_DEBUG}")
message(STATUS "GENIEX_TEST:             ${GENIEX_TEST}")
message(STATUS "GENIEX_MODEL_MANAGER:    ${GENIEX_MODEL_MANAGER}")
message(STATUS "=======================================")

# ===============================================================================
# Rust Model Manager (optional)
# ===============================================================================
if(GENIEX_MODEL_MANAGER)
    set(RUST_MM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/model-manager")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(RUST_PROFILE "debug")
        set(RUST_BUILD_FLAG "")
    else()
        set(RUST_PROFILE "release")
        set(RUST_BUILD_FLAG "--release")
    endif()
    set(RUST_TARGET_DIR "${RUST_MM_DIR}/target/${RUST_PROFILE}")

    # cargo's static-library output file name is platform-dependent:
    # Windows produces `<crate>.lib`, everything else produces
    # `lib<crate>.a`. We export RUST_MODEL_LIB as the absolute path the
    # linker will consume, and use the same path as the custom-command
    # output so ninja can track it as a build artifact.
    if(WIN32)
        set(RUST_MODEL_LIB "${RUST_TARGET_DIR}/geniex_model.lib")
    else()
        set(RUST_MODEL_LIB "${RUST_TARGET_DIR}/libgeniex_model.a")
    endif()

    add_custom_command(
        OUTPUT "${RUST_MODEL_LIB}"
        COMMAND cargo build ${RUST_BUILD_FLAG}
                --manifest-path "${RUST_MM_DIR}/Cargo.toml"
        WORKING_DIRECTORY "${RUST_MM_DIR}"
        COMMENT "Building model manager (${RUST_PROFILE})"
        VERBATIM
    )
    add_custom_target(model_manager_build
        DEPENDS "${RUST_MODEL_LIB}"
    )
endif()

enable_testing()

# headers

set(COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# plugin ids

set(GENIEX_PLUGIN_ID_LLAMA_CPP "llama_cpp")
add_compile_definitions(GENIEX_PLUGIN_ID_LLAMA_CPP="${GENIEX_PLUGIN_ID_LLAMA_CPP}")

set(GENIEX_PLUGIN_ID_QAIRT "qairt")
add_compile_definitions(GENIEX_PLUGIN_ID_QAIRT="${GENIEX_PLUGIN_ID_QAIRT}")

# compile definitions

if(GENIEX_DEBUG)
    add_compile_definitions(GENIEX_DEBUG)
    add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
else()
    set(CMAKE_BUILD_TYPE Release)
endif()

# subdirectories

add_subdirectory(./src)
add_subdirectory(./plugins)
add_subdirectory(./tests)
