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