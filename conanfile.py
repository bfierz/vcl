import os
from conans import ConanFile, CMake, tools
from conans.tools import download, unzip

class VclConan(ConanFile):
    name = "vcl"
    version = "2ea4dec"
    generators = "cmake"
    settings = "os","compiler","build_type","arch"
    options = { "vectorization": ["AVX", "AVX2", "SSE4_2", "SSE2" ] }
    default_options = "vectorization=SSE2"
    requires = (("Eigen3/3.3.3@bschindler/testing"))

    exports = ["FindVcl.cmake"]
    url="https://github.com/bfierz/vcl.git"
    license="MIT"
    description="Visual Computing Library (VCL)"
    exports_sources = "src/*"

    ZIP_FOLDER_NAME = "%s" % version

    def source(self):
        tools.replace_in_file("src/CMakeLists.txt", "PROJECT(VisualComputingLibrary)", '''PROJECT(VisualComputingLibrary)
                include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
                conan_basic_setup()
                set(VCL_EIGEN_DIR ${CONAN_EIGEN3_ROOT})''')

    def build(self):
        cmake = CMake(self.settings)

        vectorization_key = "VCL_VECTORIZE_" + str(self.options.vectorization) + ":BOOL"
        cmake.configure(self, source_dir=self.conanfile_directory + "/src/", build_dir="./", 
                defs={"VCL_BUILD_BENCHMARKS:BOOL":"off",
                    "VCL_BUILD_TESTS:BOOL":"on",
                    "VCL_BUILD_TOOLS:BOOL":"off",
                    "VCL_BUILD_EXAMPLES:BOOL" : "off",
                    vectorization_key : "on"}
                )
        cmake.build(self)

    def package(self):
        self.copy("FindEigen3.cmake", ".", ".")
        self.copy("*", dst="Eigen", src="eigen-eigen-67e894c6cd8f/Eigen")
        self.copy("*", dst="unsupported", src="eigen-eigen-67e894c6cd8f/unsupported")
        self.copy("*", dst="cmake", src="eigen-eigen-67e894c6cd8f/cmake")

    def package_info(self):
        self.cpp_info.includedirs = ['.']
