set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# gcc-13: default gcc-14 emits CXXABI_1.3.15, missing on Qualcomm Linux. See #458.
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc-13)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++-13)

# -moutline-atomics: armv8.2-a implies FEAT_LSE, so GCC would inline LSE atomics
# (LDADDAL/CAS) with no fallback, SIGILL-ing on ARMv8.0 CPUs. See #1217.
set(CMAKE_C_FLAGS "-march=armv8.2-a+fp16+dotprod -moutline-atomics -ftree-vectorize -fno-finite-math-only -flto -D_GNU_SOURCE")
set(CMAKE_CXX_FLAGS "-march=armv8.2-a+fp16+dotprod -moutline-atomics -ftree-vectorize -fno-finite-math-only -flto -D_GNU_SOURCE")

message(STATUS "Using cross compile toolchain for ARM64 Linux (gcc-13)")
