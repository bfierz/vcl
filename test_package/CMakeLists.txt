project(TestVcl)
cmake_minimum_required(VERSION 3.5)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(CMAKE_CXX_STANDARD 14)


add_executable(test_vcl test_vcl.cpp)
conan_target_link_libraries(test_vcl)

