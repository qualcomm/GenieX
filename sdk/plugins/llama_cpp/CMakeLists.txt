add_subdirectory(./src)

if(GENIEX_COMMON_LIB_FILES)
    install(FILES ${GENIEX_COMMON_LIB_FILES} DESTINATION lib/llama_cpp)
endif()

foreach(dep IN LISTS LLAMA_LIBS)
    if(TARGET ${dep})
        install(TARGETS ${dep}
            LIBRARY DESTINATION lib/llama_cpp
            RUNTIME DESTINATION lib/llama_cpp
            ARCHIVE DESTINATION lib/llama_cpp)
    endif()
endforeach()
