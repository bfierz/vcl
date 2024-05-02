/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2015 Basil Fierz
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <gtest/gtest.h>

// C++ standard library
#include <algorithm>
#include <regex>

// VCL
#include <vcl/graphics/opengl/context.h>

// Force the use of the NVIDIA GPU in an Optimius system
#ifdef VCL_COMPILER_MSVC
extern "C"
{
	_declspec(dllexport) unsigned int NvOptimusEnablement = 0x00000001;
}
#endif

bool isLlvmPipe = false; // true, if environment variable GALLIUM_DRIVER=llvmpipe is set
bool isZink = false;     // true, if environment variable GALLIUM_DRIVER=zink is set (use software vulkan: ZINK_USE_LAVAPIPE=true)
bool isD3D = false;      // true, if environment variable GALLIUM_DRIVER=d3d12 is set (use D3D12 WARP: LIBGL_ALWAYS_SOFTWARE=1)

int main(int argc, char** argv)
{
	if (std::any_of(argv, argv + argc, [](const char* arg) { return strcmp(arg, "--gtest_list_tests") == 0; }))
	{
		::testing::InitGoogleTest(&argc, argv);
		return RUN_ALL_TESTS();
	}

	// OpenGL context used during the unit-tests
	Vcl::Graphics::OpenGL::ContextDesc ctx_desc;
	ctx_desc.MajorVersion = 4;
	ctx_desc.MinorVersion = 3;
	ctx_desc.Type = Vcl::Graphics::OpenGL::ContextType::Core;
	ctx_desc.Debug = true;
	Vcl::Graphics::OpenGL::Context ctx(ctx_desc);

	// Check if we're running on Mesa
	std::regex word_regex(R"(Mesa (\d+)\.(\d+)\.(\d+))");
	std::string opengl_version((const char*)glGetString(GL_VERSION));

	std::smatch match;
	if (std::regex_search(opengl_version, match, word_regex) && match.size() == 4)
	{
		// Check if we're running on MESAs LLVMPipe
		isLlvmPipe = strncmp((const char*)glGetString(GL_RENDERER), "llvmpipe", 8) == 0;
		isZink = strncmp((const char*)glGetString(GL_RENDERER), "zink", 4) == 0;
		isD3D = strncmp((const char*)glGetString(GL_RENDERER), "D3D12", 5) == 0;

		// Check minimum required version
		const auto major = std::atoi(match[1].str().c_str());
		const auto minor = std::atoi(match[2].str().c_str());
		const auto patch = std::atoi(match[3].str().c_str());

#ifdef VCL_ABI_WINAPI
		// Require at least Mesa 22.1 for proper Windows support
		if (major < 22 || (major == 22 && minor < 1))
		{
			std::cerr << "Detected running OpenGL through Mesa " << major << "." << minor << "." << patch << ". ";
			std::cerr << "Required minimum version is 22.1.0" << std::endl;
			return 1;
		}
#else
		// Require at least Mesa 21.3 for proper Linux support
		if (major < 21 || (major == 21 && minor < 3))
		{
			std::cerr << "Detected running OpenGL through Mesa " << major << "." << minor << "." << patch << ". ";
			std::cerr << "Required minimum version is 21.3.0" << std::endl;
			return 1;
		}
#endif
	}

	// Run the tests
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
