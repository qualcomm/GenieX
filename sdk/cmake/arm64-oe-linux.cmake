set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# OpenEmbedded / Yocto SDK, for boards whose rootfs is older than the Debian
# cross toolchain targets (e.g. glibc 2.35 / libstdc++ 6.0.29): building with
# the image's own gcc and sysroot is the only way to match its symbol versions.
#
# OE_TOOLCHAIN_ROOT uses the layout QAIRT's own makefiles expect:
#   sysroots/x86_64-qtisdk-linux/usr/bin/aarch64-oe-linux/aarch64-oe-linux-{gcc,g++,ar}
#   sysroots/armv8a-oe-linux/
if(NOT DEFINED ENV{OE_TOOLCHAIN_ROOT})
    message(FATAL_ERROR "Set OE_TOOLCHAIN_ROOT to the OpenEmbedded SDK root.")
endif()
file(TO_CMAKE_PATH "$ENV{OE_TOOLCHAIN_ROOT}" _oe_root)
set(_oe_bin "${_oe_root}/sysroots/x86_64-qtisdk-linux/usr/bin/aarch64-oe-linux")

set(CMAKE_C_COMPILER   "${_oe_bin}/aarch64-oe-linux-gcc")
set(CMAKE_CXX_COMPILER "${_oe_bin}/aarch64-oe-linux-g++")
set(CMAKE_SYSROOT      "${_oe_root}/sysroots/armv8a-oe-linux")

# cargo, git, python and the Hexagon tools run on the host.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# The sysroot goes in the flags as well: CMake identifies the compiler before
# it applies CMAKE_SYSROOT, and a bare OE gcc cannot link without one.
set(_oe_flags "--sysroot=${CMAKE_SYSROOT} -march=armv8.2-a+fp16+dotprod -ftree-vectorize -fno-finite-math-only -flto -D_GNU_SOURCE")
set(CMAKE_C_FLAGS   "${_oe_flags}")
set(CMAKE_CXX_FLAGS "${_oe_flags}")

# Picked up by sdk/CMakeLists.txt for the Rust model manager cross build.
set(GENIEX_RUST_CC "${CMAKE_C_COMPILER}")
set(GENIEX_RUST_AR "${_oe_bin}/aarch64-oe-linux-ar")

message(STATUS "Using OpenEmbedded cross compile toolchain for ARM64 Linux (${_oe_root})")
