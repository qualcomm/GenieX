add_subdirectory(./src)

foreach(dep IN LISTS QAIRT_LIBS)
    #TODO: remove this after done
    if(dep STREQUAL "geniex_vlm" AND NOT GENIEX_QAIRT_ENABLE_VLM)
        continue()
    endif()

    if(TARGET ${dep})
        install(TARGETS ${dep}
            LIBRARY DESTINATION lib/qairt
            RUNTIME DESTINATION lib/qairt
            ARCHIVE DESTINATION lib/qairt)
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
