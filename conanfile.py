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
    # Packages the Vulkan validation layers; consumed at runtime only, so no CMake variable
    "with_vulkan_validation": (False, None),
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

        # Graphics backends. Driver-bound libraries (OpenGL, and EGL on Linux through
        # egl/system) stay on the system; see docs/adr/0001-conan-with-find-package-seam.md.
        if self.options.with_opengl:
            self.requires("glew/2.2.0")
            if self.options.with_egl:
                if self.settings.os == "Windows":
                    # Recipe lives in bfierz/conan-index; it exports egl::egl like egl/system
                    self.requires("mcnopper-egl/[*]")
                elif self.settings.os != "Emscripten":
                    self.requires("egl/system")
        if self.options.with_vulkan:
            # Only the vendor-neutral headers and the loader: no Vulkan SDK is needed
            self.requires("vulkan-headers/1.4.313.0")
            self.requires("vulkan-loader/1.4.313.0")
            self.requires("spirv-cross/cci.20211113")
            if self.options.with_vulkan_validation:
                self.requires("vulkan-validationlayers/1.4.350.0")
        if self.options.with_d3d12:
            self.requires("directx-headers/1.614.0")
        if self.options.with_webgpu and self.settings.os != "Emscripten":
            # Resolved only: the WebGPU backend targets the 2022 Dawn API and is not built until it
            # is ported. Emscripten ships its own WebGPU bindings. Dawn 159 needs C++20, so with_webgpu
            # requires a profile with compiler.cppstd=20 (the C++17 profiles report the package invalid).
            self.requires("dawn/159.0.8154")

        if self.options.build_examples:
            # The examples share one imgui; the pin overrides the versions that implot (1.92.9b),
            # imguizmo (1.90.5) and imfiledialog (>=1.90) ask for. The package ships the platform and
            # renderer backends as sources, which src/examples/imgui compiles once.
            self.requires("imgui/1.92.9b", override=True)
            if self.settings.os != "Emscripten" and any(
                self.options.get_safe(o) for o in ("with_opengl", "with_vulkan")
            ):
                self.requires("glfw/3.4")
            if self.options.with_opengl:
                self.requires("implot/1.1")
            if self.options.with_opengl:
                # Brings stb with it
                self.requires("imfiledialog/cci.20240916")
            if self.options.with_d3d12 and self.settings.os == "Windows":
                # ConanCenter's imguizmo/cci.20231114 uses ImGui::BeginChildFrame, which imgui 1.92
                # removed. This snapshot of upstream master comes from bfierz/conan-index.
                self.requires("imguizmo/cci.20261010")

    # Shader compilers run on the build machine, so they come from the build context: glslc from
    # shaderc compiles GLSL to SPIR-V, dxc compiles HLSL. Their bin dirs are put on the PATH of
    # the generated CMake presets, where the shader CMake modules find them with find_program.
    # ConanCenter ships shaderc binaries only for a few compilers (msvc 194, gcc 13, apple-clang 17);
    # other build profiles build shaderc from source. dxc comes from bfierz/conan-index.
    def build_requirements(self):
        if any(self.options.get_safe(o) for o in ("with_opengl", "with_vulkan")):
            self.tool_requires("shaderc/2026.4")
        if self.options.get_safe("with_d3d12"):
            self.tool_requires("dxc/1.9.2609")
        if self.options.build_tests:
            self.test_requires("gtest/1.17.0")
            self.test_requires("nlohmann_json/3.12.0")
        if self.options.build_benchmarks:
            self.test_requires("benchmark/1.9.5")
            self.test_requires("foonathan-memory/0.7.3")
            self.test_requires("geometrictoolsengine/8.0")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        CMakeDeps(self).generate()

        # The profile's compiler.cppstd is the only source of the C++ standard:
        # CMakeToolchain emits CMAKE_CXX_STANDARD from it.
        tc = CMakeToolchain(self)
        for name, (_, variable) in VCL_OPTIONS.items():
            if variable is None:
                continue
            tc.cache_variables[variable] = bool(self.options.get_safe(name))
        tc.generate()
