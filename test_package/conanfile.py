from conans import ConanFile, CMake
import os

channel = os.getenv("CONAN_CHANNEL", "testing")
username = os.getenv("CONAN_USERNAME", "bschindler")

class VclReuseConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    requires = "vcl/2ea4dec@%s/%s" % (username, channel)
    generators = "cmake"

    def build(self):
        cmake = CMake(self.settings)
        self.run('cmake %s %s' % (self.conanfile_directory, cmake.command_line))

    def test(self):
        return
        # self.run(os.sep.join([".","bin", "test"]))
