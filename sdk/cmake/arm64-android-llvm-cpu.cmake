set(ANDROID_NDK_ROOT "$ENV{ANDROID_NDK_ROOT}")
include("${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake")

# CPU-only variant for armv8.0 devices, which the default armv8.7 build traps on
# at startup. No ISA extensions. #1217
set(CMAKE_C_FLAGS "-march=armv8-a -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE")
set(CMAKE_CXX_FLAGS "-march=armv8-a -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE")
