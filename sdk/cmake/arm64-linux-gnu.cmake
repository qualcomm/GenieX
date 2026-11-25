set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc-11)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++-11)

set(CMAKE_C_FLAGS "-march=armv8.2-a+fp16+dotprod -ftree-vectorize -fno-finite-math-only -flto -D_GNU_SOURCE")
set(CMAKE_CXX_FLAGS "-march=armv8.2-a+fp16+dotprod -ftree-vectorize -fno-finite-math-only -flto -D_GNU_SOURCE")

message(STATUS "Using cross compile toolchain for ARM64 Linux (gcc-11, GLIBC 2.41)")
