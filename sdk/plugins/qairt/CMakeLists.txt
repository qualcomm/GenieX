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
