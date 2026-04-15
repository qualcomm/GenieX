add_subdirectory(./src)

foreach(dep IN LISTS QAIRT_LIBS)
    if(TARGET ${dep})
        install(TARGETS ${dep}
            LIBRARY DESTINATION lib/qairt
            RUNTIME DESTINATION lib/qairt
            ARCHIVE DESTINATION lib/qairt)
    endif()
endforeach()
