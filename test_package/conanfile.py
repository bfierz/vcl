from conans import ConanFile, CMake
import os

channel = os.getenv("CONAN_CHANNEL", "testing")
username = os.getenv("CONAN_USERNAME", "bfierz")

class VclReuseConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    requires = "gtest/1.8.0@bincrafters/stable"
    generators = "cmake"
    
    def configure(self):
        if self.settings.os != "Windows":
            self.options["vcl"].fPIC=True

    def build(self):
        cmake = CMake(self)
        self.run('cmake %s %s' % (self.source_folder, cmake.command_line))
        self.run("cmake --build . %s" % cmake.build_config)

    def test(self):
        self.run(os.sep.join([".","bin", "test_vcl"]))
