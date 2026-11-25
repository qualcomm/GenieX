set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc-11)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++-11)

# CPU-only variant for NPU-less Dragonwing IoT SoCs (unoq). No ISA extensions:
# without LSE, gcc stops inlining the LDADDAL that trapped in std::mutex. #1217
set(CMAKE_C_FLAGS "-march=armv8-a -ftree-vectorize -fno-finite-math-only -flto -D_GNU_SOURCE")
set(CMAKE_CXX_FLAGS "-march=armv8-a -ftree-vectorize -fno-finite-math-only -flto -D_GNU_SOURCE")

message(STATUS "Using baseline armv8.0-a cross compile toolchain for ARM64 Linux (gcc-11, GLIBC 2.41)")
