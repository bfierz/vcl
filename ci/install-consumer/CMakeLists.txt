# Minimal consumer of an installed VCL tree. It is built by ci/test-install-consumer.sh
# and checks that find_package(vcl) provides working vcl:: targets.
cmake_minimum_required(VERSION 3.23)
project(VclInstallConsumer CXX)

find_package(vcl CONFIG REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE vcl::core vcl::math vcl::geometry vcl::components)
