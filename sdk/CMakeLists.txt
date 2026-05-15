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

set(GENIEX_VERSION "v0.0.0" CACHE STRING "Geniex library version")

# ===============================================================================
# Build Options
# ===============================================================================

option(GENIEX_PLUGIN_LLAMA_CPP  "Enable llama.cpp backend"       ON)
option(GENIEX_PLUGIN_QAIRT      "Enable QAIRT backend"           ON)
option(GENIEX_TEST              "Enable unit tests"              OFF)
option(GENIEX_MODEL_MANAGER     "Enable model manager"           ON)

message(STATUS "=============== Options ===============")
message(STATUS "GENIEX_VERSION:          ${GENIEX_VERSION}")
message(STATUS "GENIEX_PLUGIN_LLAMA_CPP: ${GENIEX_PLUGIN_LLAMA_CPP}")
message(STATUS "GENIEX_PLUGIN_QAIRT:     ${GENIEX_PLUGIN_QAIRT}")
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

    # Cross-compile cases we cover here:
    #   - Android: x86_64 Linux container targeting aarch64-linux-android.
    #   - Linux aarch64 from a non-aarch64 host (CI Docker container).
    # Windows native ARM64 runs host == target, so plain `cargo build` is
    # correct there.
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        # NDK host tag is hard-coded (linux-x86_64 CI image); API level
        # matches sdk/CMakePresets.json (ANDROID_PLATFORM=android-31).
        set(_ndk_bin
            "$ENV{ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin")
        # cc-rs (pulled in by transitive C deps like onig_sys) looks for an
        # un-suffixed `aarch64-linux-android-clang`; the NDK only ships the
        # API-versioned wrapper. Point CC/AR at the real names so Rust
        # build scripts can invoke clang without relying on the short form.
        set(RUST_CARGO_ENV
            CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=${_ndk_bin}/aarch64-linux-android31-clang
            CC_aarch64_linux_android=${_ndk_bin}/aarch64-linux-android31-clang
            AR_aarch64_linux_android=${_ndk_bin}/llvm-ar)
        set(RUST_CARGO_TARGET_FLAGS --target aarch64-linux-android)
        set(RUST_TARGET_DIR "${RUST_MM_DIR}/target/aarch64-linux-android/${RUST_PROFILE}")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux"
           AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64"
           AND NOT CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(RUST_CARGO_ENV "")
        set(RUST_CARGO_TARGET_FLAGS --target aarch64-unknown-linux-gnu)
        set(RUST_TARGET_DIR "${RUST_MM_DIR}/target/aarch64-unknown-linux-gnu/${RUST_PROFILE}")
    else()
        set(RUST_CARGO_ENV "")
        set(RUST_CARGO_TARGET_FLAGS "")
        set(RUST_TARGET_DIR "${RUST_MM_DIR}/target/${RUST_PROFILE}")
    endif()

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
        COMMAND ${CMAKE_COMMAND} -E env ${RUST_CARGO_ENV}
                cargo build ${RUST_BUILD_FLAG} ${RUST_CARGO_TARGET_FLAGS}
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

# compile definitions

add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# subdirectories

add_subdirectory(./src)
add_subdirectory(./plugins)
add_subdirectory(./tests)
