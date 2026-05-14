if(GENIEX_PLUGIN_LLAMA_CPP)
    add_subdirectory(./llama_cpp)
endif()

if(GENIEX_PLUGIN_QAIRT)
    add_subdirectory(./qairt)
endif()

if(WIN32 AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    find_file(_openmp_dll
        NAMES libomp140.aarch64.dll libomp.dll
        HINTS
            "$ENV{VCToolsRedistDir}/arm64/Microsoft.VC143.OpenMP.LLVM"
            "$ENV{VCINSTALLDIR}/Tools/Llvm/ARM64/bin"
            "$ENV{VCINSTALLDIR}/Tools/Llvm/bin"
    )
    if(NOT _openmp_dll)
        message(FATAL_ERROR "OpenMP runtime DLL not found")
    endif()

    install(FILES "${_openmp_dll}"
            DESTINATION lib/llama_cpp
            RENAME "libomp140.aarch64.dll")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64"
   AND CMAKE_CROSSCOMPILING)
    # Qualcomm Linux does not ship libgomp; bundle it so the wheel is self-contained.
    execute_process(
        COMMAND ${CMAKE_C_COMPILER} -print-file-name=libgomp.so.1
        OUTPUT_VARIABLE _libgomp_path
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0 OR _libgomp_path STREQUAL "libgomp.so.1" OR NOT EXISTS "${_libgomp_path}")
        message(FATAL_ERROR "Cross toolchain cannot locate libgomp.so.1 (got '${_libgomp_path}').")
    endif()
    # Resolve symlink so install(FILES) copies the blob, not a dangling link.
    file(REAL_PATH "${_libgomp_path}" _libgomp_real)
    install(FILES "${_libgomp_real}"
            DESTINATION lib/llama_cpp
            RENAME "libgomp.so.1")
endif()
