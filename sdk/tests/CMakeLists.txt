# SPDX-License-Identifier: BSD-3-Clause

add_executable(geniex_xray_contract_test xray_contract_test.cpp)
target_compile_features(geniex_xray_contract_test PRIVATE cxx_std_17)
target_include_directories(geniex_xray_contract_test PRIVATE
    "${CMAKE_SOURCE_DIR}/include")

add_test(NAME geniex.xray.contract COMMAND geniex_xray_contract_test)

add_executable(geniex_xray_c_contract_test
    xray_c_contract_test.cpp
    ../src/xray.cpp)
target_compile_features(geniex_xray_c_contract_test PRIVATE cxx_std_17)
target_include_directories(geniex_xray_c_contract_test PRIVATE
    "${CMAKE_SOURCE_DIR}/include")

add_test(NAME geniex.xray.c_contract COMMAND geniex_xray_c_contract_test)

add_executable(geniex_device_route_test
    device_route_test.cpp
    ../src/device.cpp
    ../src/utils.cpp
    ../src/error.cpp)
target_compile_features(geniex_device_route_test PRIVATE cxx_std_17)
target_compile_definitions(geniex_device_route_test PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_include_directories(geniex_device_route_test PRIVATE
    "${CMAKE_SOURCE_DIR}/include")
add_test(NAME geniex.device.route COMMAND geniex_device_route_test)
