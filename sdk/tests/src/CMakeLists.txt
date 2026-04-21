set(SUITS "llm" "vlm" "embedding" "asr" "tts" "cv" "rerank" "image-gen" "diarize")
set(BRIDGE_SUITS "device" "version")

if(GENIEX_PLUGIN_LLAMA_CPP)
    list(APPEND PLUGINS "llama_cpp")
endif()
if(GENIEX_PLUGIN_QAIRT)
    list(APPEND PLUGINS "qairt")
endif()

foreach(name ${SUITS})
    set(target_name "geniex_test_${name}")
    add_executable(${target_name} ./${name}.cpp)
    target_include_directories(${target_name}
        PUBLIC ${COMMON_INCLUDES}
        PRIVATE ../include)
    target_link_libraries(${target_name} PUBLIC geniex)
    target_compile_definitions(${target_name} PRIVATE DOCTEST_CONFIG_IMPLEMENT)
    target_compile_definitions(${target_name} PRIVATE GENIEX_SHARED)

    install(TARGETS ${target_name} RUNTIME DESTINATION bin)

    # Register plugins
    foreach(plugin ${PLUGINS})
        add_test(NAME ${target_name}_${plugin}
            COMMAND ${target_name} --success --test-suite=${plugin})
        set_tests_properties(${target_name}_${plugin}
            PROPERTIES
            LABELS "${name}_${plugin}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                ENVIRONMENT "GENIEX_PLUGIN_PATH=${CMAKE_BINARY_DIR}/plugins")
    endforeach()
endforeach()

foreach(name ${BRIDGE_SUITS})
    set(target_name "geniex_test_${name}")
    add_executable(${target_name} ./${name}.cpp)
    target_include_directories(${target_name}
        PUBLIC ${COMMON_INCLUDES}
        PRIVATE ../include)
    target_link_libraries(${target_name} PUBLIC geniex)
    target_compile_definitions(${target_name} PRIVATE DOCTEST_CONFIG_IMPLEMENT)
    target_compile_definitions(${target_name} PRIVATE GENIEX_SHARED)

    install(TARGETS ${target_name} RUNTIME DESTINATION bin)

    add_test(NAME ${target_name}
        COMMAND ${target_name} --success)
    set_tests_properties(${target_name}
        PROPERTIES
        LABELS "${name}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        ENVIRONMENT "GENIEX_PLUGIN_PATH=${CMAKE_BINARY_DIR}/plugins")
endforeach()
