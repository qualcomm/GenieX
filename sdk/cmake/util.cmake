# Function to find common library files for the current platform
# Returns the list of files in the output variable specified
function(find_common_libs output_var)
    # Determine platform-specific library folder and runtime library patterns
    if(WIN32)
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
            set(LIB_FOLDER "windows-arm64")
        endif()
        set(LIB_PATTERNS "*.dll")
    elseif(UNIX)
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
            set(LIB_FOLDER "linux-aarch64")
        endif()
        set(LIB_PATTERNS "*.so" "*.so.*")
    else()
        message(WARNING "Unknown platform for common libs")
        return()
    endif()

    # Export folder name to parent scope for other targets to use
    set(COMMON_LIB_FOLDER_NAME ${LIB_FOLDER} PARENT_SCOPE)

    # Find all common library files recursively, excluding ortlib-* directories
    set(GENIEX_COMMON_LIBS_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/${LIB_FOLDER}")
    set(all_lib_files)
    foreach(pattern ${LIB_PATTERNS})
        file(GLOB_RECURSE lib_files "${GENIEX_COMMON_LIBS_SRC_DIR}/${pattern}")
        # Filter out files from ortlib-* directories (plugin-specific ONNX Runtime libs)
        list(FILTER lib_files EXCLUDE REGEX "/ortlib-[^/]+/")
        list(APPEND all_lib_files ${lib_files})
    endforeach()

    # Return the list to parent scope
    set(${output_var} ${all_lib_files} PARENT_SCOPE)
    
    # Also export the source directory for logging purposes
    set(GENIEX_COMMON_LIBS_SRC_DIR ${GENIEX_COMMON_LIBS_SRC_DIR} PARENT_SCOPE)
endfunction()