import os
from conans import ConanFile, CMake, tools
from conans.tools import download, unzip

class VclConan(ConanFile):
    name = "vcl"
    version = "master"
    generators = "cmake"
    settings = "os","compiler","build_type","arch"
    options = { 
            "vectorization": ["AVX", "AVX2", "SSE4_2" ], 
            "fPIC": [True, False]
            }
    default_options = \
            "vectorization=AVX", \
            "fPIC=False"

    requires = (("Eigen3/3.3.3@bschindler/testing"))

    url="https://github.com/bfierz/vcl.git"
    license="MIT"
    description="Visual Computing Library (VCL)"
    exports_sources = "src/*"

    def source(self):
        tools.replace_in_file("src/CMakeLists.txt", "PROJECT(VisualComputingLibrary)", '''PROJECT(VisualComputingLibrary)
                include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
                conan_basic_setup()
                set(VCL_EIGEN_DIR ${CONAN_EIGEN3_ROOT})''')

    def config_options(self):
        if self.settings.compiler == "Visual Studio":
            self.options.remove("fPIC")
            self.settings.remove("build_type")
            self.settings.compiler["Visual Studio"].remove("runtime")

    def build(self):
        vectorization_key = "VCL_VECTORIZE_" + str(self.options.vectorization) + ":BOOL"

        defs={"VCL_BUILD_BENCHMARKS:BOOL":"off",
            "VCL_BUILD_TESTS:BOOL":"on",
            "VCL_BUILD_TOOLS:BOOL":"off",
            "VCL_BUILD_EXAMPLES:BOOL" : "off",
            vectorization_key : "on"
        }
        
        if self.settings.os != "Windows" and self.options.fPIC:
            defs["CMAKE_POSITION_INDEPENDENT_CODE:BOOL"] = "on"

        cmake = CMake(self.settings)
        cmake.configure(self, source_dir=self.conanfile_directory + "/src/", build_dir="./", defs=defs)

        if cmake.is_multi_configuration:
            self.run("cmake --build . --target vcl_geometry --config Debug")
            self.run("cmake --build . --target vcl_math --config Debug")
            self.run("cmake --build . --target vcl_geometry --config Release")
            self.run("cmake --build . --target vcl_math --config Release")
        else:
            cmake.build(self, target="vcl_geometry")
            cmake.build(self, target="vcl_math")

    def package(self):
        self.copy("*.a", dst="lib", src="lib")
        self.copy("*.lib", dst="lib", src="lib")
        self.copy("*.h", dst="include", src="src/libs")
        self.copy("config.h", dst="include/vcl.core/vcl/config", src="libs/vcl.core/vcl/config")

    def package_info(self):
        self.cpp_info.includedirs = ['include/vcl.core', 'include/vcl.math', 'include/vcl.geometry']
        if self.settings.os == "Windows":
            self.cpp_info.debug.libs = ['vcl_core_d.lib', 'vcl_math_d.lib', 'vcl_geometry_d.lib']
            self.cpp_info.release.libs = ['vcl_core.lib', 'vcl_math.lib', 'vcl_geometry.lib']
        else:
            self.cpp_info.libs = ['libvcl_core.a', 'libvcl_math.a', 'libvcl_geometry.a']
        self.cpp_info.libdirs = [ "lib" ]
