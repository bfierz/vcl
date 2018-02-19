import os
from conans import ConanFile, CMake, tools
from conans.tools import download, unzip

class VclConan(ConanFile):
    name = "vcl"
    version = "2018.01"
    generators = "cmake"
    settings = "os","compiler","build_type","arch"
    options = { 
            "vectorization": ["AVX", "AVX2", "SSE4_2" ], 
            "fPIC": [True, False]
            }
    default_options = \
            "vectorization=AVX", \
            "fPIC=False"

    requires = "abseil/20180208@bincrafters/stable", \
               "eigen/3.3.4@conan/stable", \
               "fmt/4.1.0@bincrafters/stable", \
               "gsl_microsoft/20180102@bincrafters/stable"

    url="https://github.com/bfierz/vcl.git"
    license="MIT"
    description="Visual Computing Library (VCL)"
    exports_sources = "src/*"

    def source(self):
        pass

    def config_options(self):
        if self.settings.compiler == "Visual Studio":
            self.options.remove("fPIC")
            self.settings.remove("build_type")
            self.settings.compiler["Visual Studio"].remove("runtime")

    def build(self):
        vectorization_key = "VCL_VECTORIZE"

        defs={"VCL_USE_CONAN:BOOL":"on",
            "VCL_BUILD_BENCHMARKS:BOOL":"off",
            "VCL_BUILD_TESTS:BOOL":"on",
            "VCL_BUILD_TOOLS:BOOL":"off",
            "VCL_BUILD_EXAMPLES:BOOL" : "off",
            vectorization_key : str(self.options.vectorization)
        }
        
        if self.settings.os != "Windows" and self.options.fPIC:
            defs["CMAKE_POSITION_INDEPENDENT_CODE:BOOL"] = "on"

        cmake = CMake(self)
        cmake.configure(source_dir=self.source_folder + "/src/", build_dir="./", defs=defs)

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
        self.copy("*.inl", dst="include", src="src/libs")
        self.copy("config.h", dst="include/vcl.core/vcl/config", src="libs/vcl.core/vcl/config")

    def package_info(self):
        self.cpp_info.includedirs = ['include/vcl.core', 'include/vcl.math', 'include/vcl.geometry']
        if self.settings.os == "Windows":
            self.cpp_info.debug.libs = ['vcl_core_d.lib', 'vcl_math_d.lib', 'vcl_geometry_d.lib']
            self.cpp_info.release.libs = ['vcl_core.lib', 'vcl_math.lib', 'vcl_geometry.lib']
        else:
            self.cpp_info.libs = ['libvcl_core.a', 'libvcl_math.a', 'libvcl_geometry.a']
        self.cpp_info.libdirs = [ "lib" ]
