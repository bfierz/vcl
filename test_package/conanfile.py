from conans import ConanFile, CMake
import os

class VclReuseConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    requires = "gtest/1.8.0@bincrafters/stable"
    generators = "cmake"
    
    def configure(self):
        if self.settings.os != "Windows":
            self.options["vcl"].fPIC = True
            self.options["gtest"].fPIC = True
        self.options["gtest"].shared = False

    def build(self):
        cmake = CMake(self)
        cmake.definitions["VCL_VECTORIZE"] = str(self.options["vcl"].vectorization)
        cmake.configure()
        cmake.build()

    def test(self):
        self.run(os.sep.join([".", "bin", "test_vcl"]))
