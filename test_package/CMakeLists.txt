project(TestVcl)
cmake_minimum_required(3.5)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)

find_package(vcl REQUIRED)

add_executable(test_vcl test_vcl.cpp)
target_link_libraries(test_vcl vcl)

