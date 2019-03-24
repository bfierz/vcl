import os
from conans import ConanFile, CMake, tools
from conans.tools import download, unzip

class VclConan(ConanFile):
    name = "vcl"
    version = "master"
    generators = "cmake"
    settings = "os","compiler","build_type","arch"
    options = { 
            "vectorization": ["AVX", "AVX 2", "SSE 4.2" ], 
            "fPIC": [True, False]
            }
    default_options = \
            "vectorization=AVX", \
            "fPIC=False"

    requires = "abseil/20180600@bincrafters/stable", \
               "eigen/3.3.7@conan/stable", \
               "fmt/4.1.0@bincrafters/stable", \
               "glew/2.1.0@bincrafters/stable",

    url="https://github.com/bfierz/vcl.git"
    license="MIT"
    description="Visual Computing Library (VCL)"
    exports_sources = "src/*"

    def source(self):
        tools.replace_in_file("src/CMakeLists.txt", "project(VisualComputingLibrary)", '''project(VisualComputingLibrary)
                include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
                conan_basic_setup(TARGETS)''')


    def config_options(self):
        if self.settings.compiler == "Visual Studio":
            del self.options.fPIC
            # Support multi-package configuration
            #del self.settings.build_type
            #del self.settings.compiler["Visual Studio"].runtime
        else:
            self.options["abseil"].fPIC = self.options.fPIC
            self.options["glew"].fPIC = self.options.fPIC

    def build(self):
        cmake = CMake(self)
        # Configure which parts to build
        cmake.definitions["VCL_BUILD_BENCHMARKS"] = False
        cmake.definitions["VCL_BUILD_TESTS"] = False
        cmake.definitions["VCL_BUILD_TOOLS"] = False
        cmake.definitions["VCL_BUILD_EXAMPLES"] = False
        # Support multi-package configuration
        #cmake.definitions["CMAKE_DEBUG_POSTFIX"] = "_d"
        # Configure features
        cmake.definitions["VCL_VECTORIZE"] = str(self.options.vectorization)
        cmake.definitions["VCL_OPENGL_SUPPORT"] = True
        if self.settings.os != "Windows":
            cmake.definitions["CMAKE_POSITION_INDEPENDENT_CODE"] = self.options.fPIC
        # Configure external targets
        cmake.definitions["vcl_ext_absl"] = "CONAN_PKG::abseil"
        cmake.definitions["vcl_ext_eigen"] = "CONAN_PKG::eigen"
        cmake.definitions["vcl_ext_fmt"] = "CONAN_PKG::fmt"
        cmake.definitions["vcl_ext_glew"] = "CONAN_PKG::glew"
        cmake.configure(source_dir=self.source_folder + "/src/")

        if cmake.is_multi_configuration:
            cmake.build(target="vcl_core",     args=["--config","Debug"])
            cmake.build(target="vcl_geometry", args=["--config","Debug"])
            cmake.build(target="vcl_graphics", args=["--config","Debug"])
            cmake.build(target="vcl_math",     args=["--config","Debug"])
            cmake.build(target="vcl_core",     args=["--config","Release"])
            cmake.build(target="vcl_geometry", args=["--config","Release"])
            cmake.build(target="vcl_graphics", args=["--config","Release"])
            cmake.build(target="vcl_math",     args=["--config","Release"])
        else:
            cmake.build(target="vcl_core")
            cmake.build(target="vcl_geometry")
            cmake.build(target="vcl_graphics")
            cmake.build(target="vcl_math")

    def package(self):
        self.copy("*.dll", dst="bin", src="bin")
        self.copy("*.a", dst="lib", src="lib")
        self.copy("*.lib", dst="lib", src="lib")
        self.copy("*.h", dst="include", src="src/libs")
        self.copy("*.inl", dst="include", src="src/libs")
        self.copy("config.h", dst="include/vcl.core/vcl/config", src="libs/vcl.core/vcl/config")

    def package_info(self):
        self.cpp_info.includedirs = ['include/vcl.core', 'include/vcl.math', 'include/vcl.graphics', 'include/vcl.geometry']
        if self.settings.os == "Windows":
            if hasattr(self.settings, "build_type"):
                self.cpp_info.debug.libs = ['vcl_core_d.lib', 'vcl_math_d.lib', 'vcl_geometry_d.lib', 'vcl_graphics_d.lib']
                self.cpp_info.release.libs = ['vcl_core.lib', 'vcl_math.lib', 'vcl_geometry.lib', 'vcl_graphics.lib']
            else:
                self.cpp_info.libs = ['vcl_core.lib', 'vcl_math.lib', 'vcl_geometry.lib', 'vcl_graphics.lib']
        else:
            self.cpp_info.libs = ['libvcl_core.a', 'libvcl_math.a', 'libvcl_geometry.a', 'libvcl_graphics.a']
        self.cpp_info.libdirs = [ "lib" ]
