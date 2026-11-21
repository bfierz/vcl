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

    # Resolved from ConanCenter; all three are usable with C++17. Versions are tracked here and
    # in conan.lock.
    def requirements(self):
        self.requires("eigen/3.4.1")
        self.requires("abseil/20250814.2")
        self.requires("fmt/11.2.0")

    # Shader compilers run on the build machine, so they come from the build context: glslc from
    # shaderc compiles GLSL to SPIR-V, dxc compiles HLSL. Their bin dirs are put on the PATH of
    # the generated CMake presets, where the shader CMake modules find them with find_program.
    # ConanCenter ships shaderc binaries only for a few compilers (msvc 194, gcc 13, apple-clang 17);
    # other build profiles build shaderc from source. dxc comes from bfierz/conan-index.
    def build_requirements(self):
        if any(self.options.get_safe(o) for o in ("with_opengl", "with_vulkan", "with_webgpu")):
            self.tool_requires("shaderc/2026.4")
        if self.options.get_safe("with_d3d12"):
            self.tool_requires("dxc/1.9.2609")

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
