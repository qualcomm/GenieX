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
option(GENIEX_CPU_ONLY          "CPU inference only, baseline ISA" OFF)

message(STATUS "=============== Options ===============")
message(STATUS "GENIEX_VERSION:          ${GENIEX_VERSION}")
message(STATUS "GENIEX_PLUGIN_LLAMA_CPP: ${GENIEX_PLUGIN_LLAMA_CPP}")
message(STATUS "GENIEX_PLUGIN_QAIRT:     ${GENIEX_PLUGIN_QAIRT}")
message(STATUS "GENIEX_BENCHMARK:        ${GENIEX_BENCHMARK}")
message(STATUS "GENIEX_CPU_ONLY:         ${GENIEX_CPU_ONLY}")
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
    # Pin the linker to the aarch64 cross-gcc; rustc otherwise links via
    # rust-lld pointed at the host (x86_64) gcc-ld, which breaks once a
    # dependency is emitted as a shared object.
    # gcc-13 matches the on-device libstdc++ ABI (#458); the *-unknown-* names
    # are symlinks from the toolchain image. A toolchain file may name its own
    # compiler and archiver instead (cmake/arm64-oe-linux.cmake does).
    if(NOT DEFINED GENIEX_RUST_CC)
        set(GENIEX_RUST_CC aarch64-linux-gnu-gcc-13)
    endif()
    if(NOT DEFINED GENIEX_RUST_AR)
        set(GENIEX_RUST_AR aarch64-linux-gnu-ar)
    endif()
    set(RUST_CARGO_ENV
        CARGO_TARGET_DIR=${RUST_TARGET_ROOT}
        CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=${GENIEX_RUST_CC}
        CC_aarch64_unknown_linux_gnu=${GENIEX_RUST_CC}
        AR_aarch64_unknown_linux_gnu=${GENIEX_RUST_AR})
    # A compiler that is not a sysroot-baked wrapper needs the sysroot spelled
    # out, for cc-rs builds and for the final link alike.
    if(CMAKE_SYSROOT)
        list(APPEND RUST_CARGO_ENV
            CFLAGS_aarch64_unknown_linux_gnu=--sysroot=${CMAKE_SYSROOT}
            "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS=-C link-arg=--sysroot=${CMAKE_SYSROOT}")
    endif()
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
# * llama-hexagon-power-mode.patch adds the HTP DCVS/HMX power-mode table from
#   our internal llama.cpp PR (ggml/llama.cpp#340, unmerged upstream). Delete
#   this file once the submodule is bumped past that merge.
# * llama-hexagon-power-mode-setter.patch is a permanent GenieX addition on top
#   of the above: exposes "ggml_backend_hexagon_set_power_mode" via
#   get_proc_address so the llama.cpp plugin can change the power mode
#   per-model (llama-hexagon-power-mode.patch alone only reads it once, from
#   GGML_HEXAGON_POWER_MODE, at backend-registry construction). See
#   sdk/plugins/llama_cpp/src/htp_session.cpp for the caller.
# * llama-hexagon-sdk-select.patch is a permanent GenieX addition that lets the
#   HTP skels target DSP firmware older than upstream assumes. It adds cache
#   variables, all defaulting to upstream behavior:
#   GGML_HEXAGON_HTP_SDK_ROOT / GGML_HEXAGON_HTP_TOOLS_ROOT (build only the
#   skels with another SDK; the CPU-side library needs SDK 6.x headers),
#   GGML_HEXAGON_TOOLV (prebuilt lib dir prefix; Hexagon SDK 5.5.x is toolv87,
#   not toolv19), GGML_HEXAGON_HTP_ARCHS (SDK 5.5.x has no QuRT for v79/v81),
#   and GGML_HEXAGON_SWIV_UTILITY (embed the SWIV CRC that functional-safety
#   firmware requires before it loads a library). It also finds qaic in the
#   SDK 5.5.x layout, maps the two HMX voltage corners SDK 5.5.x lacks to the
#   next lower ones, and lets g++ before 13 compile the CPU-side library
#   (no _Float16 in C++). Skels built against SDK 6.x headers import
#   compute_resource_attr_init_v2, which such firmware does not export.
# * llama-hexagon-yield-between-ops.patch is a permanent GenieX addition that
#   makes the HTP skel honor a VTCM/HMX release request between ops instead of
#   only between op batches. A batch can be a whole graph, so a long prefill
#   kept both resources for its full duration; on functional-safety firmware
#   the periodic DSP self-test then times out and the host shuts the DSP down.
#   It also splits non-contiguous copies into 128 KiB chunks and yields between
#   them: one transposed copy in a vision encoder takes about 70 ms.
# * llama-hexagon-hmx-pad-k.patch lets the flat 2D HMX matmul take F16/F32
#   weights whose K is not a multiple of 32 by padding K with zeros. Without it
#   they fall back to HVX: the SigLIP so400m FFN down projection (K = 4304)
#   takes about 100 ms per op instead of 3 ms, long enough to trip the
#   functional-safety self-test above.
# * llama-opencl-adreno-cpy-constreg.patch works around an Adreno OpenCL
#   compiler crash: kernel_cpy_f32_f32_pack's min(ne00, get_local_size(0))
#   emits an IMINrr with two constant-class operands, which the driver's
#   instruction validator rejects (NumConstRegsError). See GenieX #1250. The
#   same fix is applied to kernel_add: the strength-reduced
#   get_local_size(0) * nb0X step emits an equivalent UMULLrr, hit on a
#   QCS9075 board (Adreno driver 1.855.3) during VLM inference. See
#   qcom-ai-hub/geniex#1566.
# * llama-zero-free-mem-even-split.patch keeps load_tensors from throwing when
#   every device reports 0 free memory: the splits normalize to NaN and the
#   device lookup runs off the end. Adreno 643 on QCS6490 hits it because the
#   OpenCL backend subtracts a fixed 1 GiB margin from global_mem_size. See #1497.
# * llama-opencl-adreno-xmem-gemm-off.patch defaults the Adreno xmem F16xF32
#   GEMM path (upstream #27640) back to off. On this Adreno generation it
#   corrupts output after repeated model loads through the same OpenCL
#   context (quality/logits-parity tests pass individually, fail once several
#   gpu/hybrid models have loaded in one process); upstream's own
#   GGML_OPENCL_ADRENO_XMEM_GEMM=0 override confirms the fix.
if(GENIEX_PLUGIN_LLAMA_CPP)
    set(_LLAMA_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/../third-party/llama.cpp")
    set(_LLAMA_PATCHES)
    if(GGML_HEXAGON)
        list(APPEND _LLAMA_PATCHES
            "${CMAKE_CURRENT_SOURCE_DIR}/patches/llama-hexagon-release-sessions.patch"
            "${CMAKE_CURRENT_SOURCE_DIR}/patches/llama-hexagon-power-mode.patch"
            "${CMAKE_CURRENT_SOURCE_DIR}/patches/llama-hexagon-power-mode-setter.patch"
            "${CMAKE_CURRENT_SOURCE_DIR}/patches/llama-hexagon-sdk-select.patch"
            "${CMAKE_CURRENT_SOURCE_DIR}/patches/llama-hexagon-yield-between-ops.patch"
            "${CMAKE_CURRENT_SOURCE_DIR}/patches/llama-hexagon-hmx-pad-k.patch")
    endif()
    if(GGML_OPENCL)
        list(APPEND _LLAMA_PATCHES
            "${CMAKE_CURRENT_SOURCE_DIR}/patches/llama-opencl-adreno-cpy-constreg.patch"
            "${CMAKE_CURRENT_SOURCE_DIR}/patches/llama-zero-free-mem-even-split.patch"
            "${CMAKE_CURRENT_SOURCE_DIR}/patches/llama-opencl-adreno-xmem-gemm-off.patch")
    endif()
    find_package(Git QUIET REQUIRED)
    foreach(_LLAMA_PATCH ${_LLAMA_PATCHES})
        if(EXISTS "${_LLAMA_PATCH}")
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
    endforeach()
endif()

# headers

set(COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# compile definitions

add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(GENIEX_CPU_ONLY)
    add_compile_definitions(GENIEX_CPU_ONLY)
endif()

# subdirectories

add_subdirectory(./src)
add_subdirectory(./plugins)
if(GENIEX_BENCHMARK)
    add_subdirectory(./benchmark)
endif()
