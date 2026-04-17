# setup qairt

set(BUILD_EXAMPLES OFF)
set(GENIEX_BUILD_VLM OFF)

# Keep Rust artifact paths short on Windows to avoid MAX_PATH linker failures
# in third-party/tokenizers-cpp when building QAIRT dependencies.
if(WIN32)
    set(TOKENIZERS_CPP_CARGO_TARGET_DIR "${CMAKE_BINARY_DIR}/tok")
endif()

add_subdirectory(
    ../../../third-party/geniex-qairt
    ${CMAKE_BINARY_DIR}/third-party/geniex-qairt
    EXCLUDE_FROM_ALL)

# setup plugin

add_subdirectory(./src)

# setup install

set(QAIRT_LIBS geniex_core nexaproc nexa-sampling geniex_vlm)
foreach(dep IN LISTS QAIRT_LIBS)
    if(TARGET ${dep})
        install(TARGETS ${dep}
            LIBRARY DESTINATION lib/qairt
            RUNTIME DESTINATION lib/qairt)
    endif()
endforeach()

# Install htp-files-v{arch}/ folders from the build tree into lib/qairt/.
# geniex-qairt's POST_BUILD rule copies them next to geniex_core after each build;
# its own install() rules are suppressed by EXCLUDE_FROM_ALL in qairt.cmake.
# resolveHtpPaths() in runtime.h locates these relative to geniex_core at runtime.
install(CODE [[
    file(GLOB _htp_dirs LIST_DIRECTORIES true "$<TARGET_FILE_DIR:geniex_core>/htp-files-v*")
    foreach(_dir IN LISTS _htp_dirs)
        if(IS_DIRECTORY "${_dir}")
            get_filename_component(_name "${_dir}" NAME)
            message(STATUS "Installing ${_name} -> ${CMAKE_INSTALL_PREFIX}/lib/qairt/${_name}")
            file(INSTALL "${_dir}/" DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/qairt/${_name}")
        endif()
    endforeach()
]])
