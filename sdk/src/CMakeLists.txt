file(GLOB BRIDGE_SRC *.cpp)

add_library(geniex SHARED ${BRIDGE_SRC})

target_compile_definitions(geniex PRIVATE GENIEX_VERSION="${GENIEX_VERSION}")
target_compile_definitions(geniex PRIVATE ML_SHARED ML_BUILD)

target_include_directories(geniex PUBLIC ${COMMON_INCLUDE})

install(TARGETS geniex
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib)

install(FILES ${CMAKE_SOURCE_DIR}/include/ml.h DESTINATION include)
