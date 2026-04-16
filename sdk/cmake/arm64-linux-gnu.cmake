set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_CUDA_HOST_COMPILER aarch64-linux-gnu-g++)

# set to real x86 nvcc path
#set(CMAKE_CUDA_COMPILER nvcc)
#set(CMAKE_CUDA_FLAGS "-I/root/micromamba/include -L/root/micromamba/lib")
message(STATUS "Using cross compile toolchain for ARM64 Linux")
message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
