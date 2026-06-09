add_executable(geniex_benchmark benchmark.c)

target_include_directories(geniex_benchmark
    PRIVATE
        ${COMMON_INCLUDE}
)

target_link_libraries(geniex_benchmark
    PRIVATE geniex
)

# sqrt() in summarize_full() needs libm on glibc cross-compile targets
# (Linux aarch64). bionic / msvcrt fold sqrt into libc, but glibc keeps it
# in libm and the linker enforces --as-needed.
if(NOT WIN32 AND NOT ANDROID)
    target_link_libraries(geniex_benchmark PRIVATE m)
endif()

target_compile_definitions(geniex_benchmark PRIVATE GENIEX_SHARED)

set_target_properties(geniex_benchmark PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

install(TARGETS geniex_benchmark RUNTIME DESTINATION bin)
