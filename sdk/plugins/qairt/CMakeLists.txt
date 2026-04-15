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

set(_qairt_htp_root "${GENIEX_QAIRT_DIR}/third-party")
file(GLOB _qairt_htp_dirs LIST_DIRECTORIES true "${_qairt_htp_root}/htp-files-*")

if(_qairt_htp_dirs)
    message(STATUS "Bundling QAIRT HTP runtime folders to lib/qairt:")
    foreach(_htp_dir IN LISTS _qairt_htp_dirs)
        message(STATUS "  - ${_htp_dir}")
    endforeach()
    install(DIRECTORY ${_qairt_htp_dirs} DESTINATION lib/qairt)
else()
    message(WARNING "No htp-files-* folders found under ${_qairt_htp_root}")
endif()
