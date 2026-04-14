set(SUITS "llm" "vlm" "embedding" "asr" "tts" "cv" "rerank" "image-gen" "diarize")
set(BRIDGE_SUITS "device" "version")

if (GENIEX_PLUGIN_LLAMA_CPP)
    list(APPEND PLUGINS "llama_cpp")
endif()
if (GENIEX_PLUGIN_QNN)
    list(APPEND PLUGINS "qairt")
endif()


if(${CMAKE_GENERATOR} MATCHES "Visual Studio")
    copy_output(tests/src/$<CONFIG> geniex_bridge OpenSSL::SSL OpenSSL::Crypto)
    copy_output(tests/src/$<CONFIG>/${COMMON_LIB_REL_PATH} ${GENIEX_COMMON_LIB_FILES} ${LLAMA_LIBS})
else()
    copy_output(tests/src geniex_bridge OpenSSL::SSL OpenSSL::Crypto)
    copy_output(tests/src/${COMMON_LIB_REL_PATH} ${GENIEX_COMMON_LIB_FILES} ${LLAMA_LIBS})
endif()

foreach(name ${SUITS})
    set(target_name "geniex_test_${name}")
    add_executable(${target_name} ./${name}.cpp)
    target_include_directories(${target_name}
        PUBLIC ${COMMON_INCLUDES}
        PRIVATE ../include)
    target_link_libraries(${target_name} PUBLIC geniex_bridge)
    target_compile_definitions(${target_name} PRIVATE DOCTEST_CONFIG_IMPLEMENT)
    target_compile_definitions(${target_name} PRIVATE ML_SHARED)

    # Register plugins
    foreach(plugin ${PLUGINS})
        add_test(NAME ${target_name}_${plugin}
            COMMAND ${target_name} --success --test-suite=${plugin})
        set_tests_properties(${target_name}_${plugin}
            PROPERTIES
            LABELS "${name}_${plugin}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                ENVIRONMENT "GENIEX_PLUGIN_PATH=${CMAKE_BINARY_DIR}/out")
    endforeach()

    # Copy test executables to out/tests for Android: TODO: remove in future
    if(ANDROID)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/out/tests
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${target_name}> ${CMAKE_BINARY_DIR}/out/tests/
            COMMENT "Copying ${target_name} to out/tests for Android")
    endif()
endforeach()

foreach(name ${BRIDGE_SUITS})
    set(target_name "geniex_test_${name}")
    add_executable(${target_name} ./${name}.cpp)
    target_include_directories(${target_name}
        PUBLIC ${COMMON_INCLUDES}
        PRIVATE ../include)
    target_link_libraries(${target_name} PUBLIC geniex_bridge)
    target_compile_definitions(${target_name} PRIVATE DOCTEST_CONFIG_IMPLEMENT)
    target_compile_definitions(${target_name} PRIVATE ML_SHARED)

    add_test(NAME ${target_name}
        COMMAND ${target_name} --success)
    set_tests_properties(${target_name}
        PROPERTIES
        LABELS "${name}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        ENVIRONMENT "GENIEX_PLUGIN_PATH=${CMAKE_BINARY_DIR}/out")
endforeach()
