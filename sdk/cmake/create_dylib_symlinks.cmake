# Script to create version symlinks for shared library files at build time
#
# Creates SOVERSION symlinks for Linux:
#   Linux:  libfoo.so.0    -> libfoo.so.0.9.4
#           libfoo.so      -> libfoo.so.0

if(NOT DEFINED LIB_DIR)
    message(FATAL_ERROR "LIB_DIR must be defined")
endif()

# Helper function to create a symlink
function(create_lib_symlink LIB_DIR SYMLINK_NAME TARGET_NAME)
    set(SYMLINK_PATH "${LIB_DIR}/${SYMLINK_NAME}")

    # Remove existing symlink if it exists
    if(EXISTS "${SYMLINK_PATH}")
        file(REMOVE "${SYMLINK_PATH}")
    endif()

    # Create symlink
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink "${TARGET_NAME}" "${SYMLINK_NAME}"
        WORKING_DIRECTORY "${LIB_DIR}"
        RESULT_VARIABLE SYMLINK_RESULT
    )

    if(SYMLINK_RESULT EQUAL 0)
        message(STATUS "Created symlink: ${SYMLINK_NAME} -> ${TARGET_NAME}")
    else()
        message(WARNING "Failed to create symlink: ${SYMLINK_NAME} -> ${TARGET_NAME}")
    endif()
endfunction()

# Process Linux .so files (libname.so.X.Y.Z pattern)
file(GLOB VERSIONED_SO_LIBS "${LIB_DIR}/lib*.so.*.*.*")

foreach(FULL_LIB ${VERSIONED_SO_LIBS})
    get_filename_component(LIB_NAME ${FULL_LIB} NAME)

    # Extract the major version symlink name (libname.so.X)
    # Pattern: libname.so.0.9.4 -> libname.so.0
    string(REGEX REPLACE "^(lib[^.]+\\.so\\.[0-9]+)\\.[0-9]+\\.[0-9]+$" "\\1" MAJOR_LIB ${LIB_NAME})

    # Extract the base symlink name (libname.so)
    # Pattern: libname.so.0.9.4 -> libname.so
    string(REGEX REPLACE "^(lib[^.]+\\.so)\\.[0-9]+\\.[0-9]+\\.[0-9]+$" "\\1" BASE_LIB ${LIB_NAME})

    # Create major version symlink (libname.so.0 -> libname.so.0.9.4)
    if(NOT "${MAJOR_LIB}" STREQUAL "${LIB_NAME}")
        create_lib_symlink("${LIB_DIR}" "${MAJOR_LIB}" "${LIB_NAME}")
    endif()

    # Create base symlink (libname.so -> libname.so.0)
    if(NOT "${BASE_LIB}" STREQUAL "${LIB_NAME}" AND NOT "${BASE_LIB}" STREQUAL "${MAJOR_LIB}")
        create_lib_symlink("${LIB_DIR}" "${BASE_LIB}" "${MAJOR_LIB}")
    endif()

endforeach()

