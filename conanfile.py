"""Consumer conanfile of the Visual Computing Library.

It only drives the build of this repository: it generates the CMake toolchain
and the dependency config files, and translates the Conan options into the
VCL_* cache variables. It does not package VCL.

    conan install . -pr:h=conan/profiles/linux-gcc -s build_type=Release --build=missing
    cmake --preset conan-release
    cmake --build --preset conan-release
"""
from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain, cmake_layout

# Single source of truth for the options: option name -> (default, CMake cache variable).
# The `options` and `default_options` attributes and the cache variables written in
# generate() all derive from this table, so they cannot drift apart.
VCL_OPTIONS = {
    "with_opengl": (False, "VCL_OPENGL_SUPPORT"),
    "with_egl": (False, "VCL_EGL_SUPPORT"),
    "with_vulkan": (False, "VCL_VULKAN_SUPPORT"),
    "with_d3d12": (False, "VCL_D3D12_SUPPORT"),
    "with_webgpu": (False, "VCL_WEBGPU_SUPPORT"),
    "build_tests": (True, "VCL_BUILD_TESTS"),
    "build_benchmarks": (True, "VCL_BUILD_BENCHMARKS"),
    "build_examples": (False, "VCL_BUILD_EXAMPLES"),
    "build_tools": (True, "VCL_BUILD_TOOLS"),
}


class VclConan(ConanFile):
    settings = "os", "arch", "compiler", "build_type"
    options = {name: [True, False] for name in VCL_OPTIONS}
    default_options = {name: default for name, (default, _) in VCL_OPTIONS.items()}

    def layout(self):
        cmake_layout(self)

    def generate(self):
        CMakeDeps(self).generate()

        # The profile's compiler.cppstd is the only source of the C++ standard:
        # CMakeToolchain emits CMAKE_CXX_STANDARD from it.
        tc = CMakeToolchain(self)
        for name, (_, variable) in VCL_OPTIONS.items():
            tc.cache_variables[variable] = bool(self.options.get_safe(name))
        tc.generate()
