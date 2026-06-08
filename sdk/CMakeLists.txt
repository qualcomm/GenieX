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

set(GENIEX_VERSION "v0.0.0" CACHE STRING "GenieX library version")

# ===============================================================================
# Build Options
# ===============================================================================

option(GENIEX_PLUGIN_LLAMA_CPP  "Enable llama.cpp backend"       ON)
option(GENIEX_PLUGIN_QAIRT      "Enable QAIRT backend"           ON)
option(GENIEX_BENCHMARK         "Build benchmark example"        OFF)

message(STATUS "=============== Options ===============")
message(STATUS "GENIEX_VERSION:          ${GENIEX_VERSION}")
message(STATUS "GENIEX_PLUGIN_LLAMA_CPP: ${GENIEX_PLUGIN_LLAMA_CPP}")
message(STATUS "GENIEX_PLUGIN_QAIRT:     ${GENIEX_PLUGIN_QAIRT}")
message(STATUS "GENIEX_BENCHMARK:        ${GENIEX_BENCHMARK}")
message(STATUS "=======================================")

# ===============================================================================
# Rust Model Manager
# ===============================================================================
set(RUST_MM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/model-manager")
# Place cargo's target dir inside the CMake build directory so a clean
# build (`rm -rf build/` / `cmake --build --target clean`) wipes the
# Rust artifacts too. Otherwise stale staticlibs in
# model-manager/target/ silently survive clean builds.
set(RUST_TARGET_ROOT "${CMAKE_CURRENT_BINARY_DIR}/cargo-target")
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
        CARGO_TARGET_DIR=${RUST_TARGET_ROOT}
        CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=${_ndk_bin}/aarch64-linux-android31-clang
        CC_aarch64_linux_android=${_ndk_bin}/aarch64-linux-android31-clang
        AR_aarch64_linux_android=${_ndk_bin}/llvm-ar)
    set(RUST_CARGO_TARGET_FLAGS --target aarch64-linux-android)
    set(RUST_TARGET_DIR "${RUST_TARGET_ROOT}/aarch64-linux-android/${RUST_PROFILE}")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux"
       AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64"
       AND NOT CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(RUST_CARGO_ENV CARGO_TARGET_DIR=${RUST_TARGET_ROOT})
    set(RUST_CARGO_TARGET_FLAGS --target aarch64-unknown-linux-gnu)
    set(RUST_TARGET_DIR "${RUST_TARGET_ROOT}/aarch64-unknown-linux-gnu/${RUST_PROFILE}")
else()
    set(RUST_CARGO_ENV CARGO_TARGET_DIR=${RUST_TARGET_ROOT})
    set(RUST_CARGO_TARGET_FLAGS "")
    set(RUST_TARGET_DIR "${RUST_TARGET_ROOT}/${RUST_PROFILE}")
endif()

# cargo's static-library output file name is platform-dependent:
# Windows produces `<crate>.lib`, everything else produces
# `lib<crate>.a`. We export RUST_MODEL_LIB as the absolute path the
# linker will consume.
if(WIN32)
    set(RUST_MODEL_LIB "${RUST_TARGET_DIR}/geniex_model.lib")
else()
    set(RUST_MODEL_LIB "${RUST_TARGET_DIR}/libgeniex_model.a")
endif()

# Always invoke cargo and let it decide whether to rebuild — CMake has
# no visibility into the Rust source graph, so an OUTPUT-driven custom
# command would skip cargo whenever libgeniex_model.a happens to exist,
# even after the Rust sources change. cargo's own incremental check is
# cheap and authoritative; if nothing changed it won't touch the .a.
add_custom_target(model_manager_build ALL
    BYPRODUCTS "${RUST_MODEL_LIB}"
    COMMAND ${CMAKE_COMMAND} -E env ${RUST_CARGO_ENV}
            cargo build ${RUST_BUILD_FLAG} ${RUST_CARGO_TARGET_FLAGS}
            --manifest-path "${RUST_MM_DIR}/Cargo.toml"
    WORKING_DIRECTORY "${RUST_MM_DIR}"
    COMMENT "Building model manager (${RUST_PROFILE})"
    VERBATIM
)

enable_testing()

# ===============================================================================
# Third-party patches
# ===============================================================================
# Apply local patches to vendored submodules at configure time so the submodules
# themselves stay pointed at upstream commits. Runs every configure and detects
# an already-patched tree via `git apply --reverse --check`, so `git submodule
# update --force` or a manual submodule revert can't silently leave a
# half-patched build. Rebasing a submodule onto a new upstream tip requires
# refreshing the corresponding .patch.
#
# * llama-hexagon-release-sessions.patch exposes a
#   "ggml_backend_hexagon_release_sessions" entry via get_proc_address, used by
#   the llama.cpp plugin to tear down HTP FastRPC channels on model unload so
#   QAIRT's HTP re-init is not poisoned by stale sessions. See
#   sdk/plugins/llama_cpp/src/llm.cpp for the caller.
if(GENIEX_PLUGIN_LLAMA_CPP)
    set(_LLAMA_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/../third-party/llama.cpp")
    set(_LLAMA_PATCH "${CMAKE_CURRENT_SOURCE_DIR}/patches/llama-hexagon-release-sessions.patch")
    if(EXISTS "${_LLAMA_PATCH}")
        find_package(Git QUIET REQUIRED)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} -C "${_LLAMA_DIR}" apply --reverse --check "${_LLAMA_PATCH}"
            RESULT_VARIABLE _reverse_check_rc
            OUTPUT_QUIET ERROR_QUIET)
        if(NOT _reverse_check_rc EQUAL 0)
            message(STATUS "Applying llama.cpp patch: ${_LLAMA_PATCH}")
            execute_process(
                COMMAND ${GIT_EXECUTABLE} -C "${_LLAMA_DIR}" apply "${_LLAMA_PATCH}"
                RESULT_VARIABLE _apply_rc)
            if(NOT _apply_rc EQUAL 0)
                message(FATAL_ERROR
                    "Failed to apply ${_LLAMA_PATCH} to ${_LLAMA_DIR}. "
                    "If the submodule was bumped, refresh the patch against the new tip.")
            endif()
        endif()
    endif()
endif()

# headers

set(COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# compile definitions

add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# subdirectories

add_subdirectory(./src)
add_subdirectory(./plugins)
if(GENIEX_BENCHMARK)
    add_subdirectory(./benchmark)
endif()
