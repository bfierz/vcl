project(TestVcl)
cmake_minimum_required(VERSION 3.5)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

option(FOONATHAN_MEMORY_BUILD_EXAMPLES OFF)
option(FOONATHAN_MEMORY_BUILD_TESTS OFF)
subdirs(../src/externals/memory)

option(BuildTests OFF)
subdirs(../src/externals/json)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_vcl
	../src/tests/vcl.core/allocator.cpp
	../src/tests/vcl.core/bitvector.cpp
	../src/tests/vcl.core/flags.cpp
	../src/tests/vcl.core/gather.cpp
	../src/tests/vcl.core/interleavedarray.cpp
	../src/tests/vcl.core/load.cpp
	../src/tests/vcl.core/minmax.cpp
	../src/tests/vcl.core/rtti.cpp
	../src/tests/vcl.core/scatter.cpp
	../src/tests/vcl.core/scopeguard.cpp
	../src/tests/vcl.core/simd.cpp
	../src/tests/vcl.core/smart_ptr.cpp)

target_link_libraries(test_vcl
	CONAN_PKG::Abseil
	foonathan_memory
	nlohmann_json
)

conan_target_link_libraries(test_vcl)

