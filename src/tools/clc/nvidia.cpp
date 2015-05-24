/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014 - 2015 Basil Fierz
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

// VCL configuration
#include "nvidia.h"

// C++ standard library
#include <iostream>

// Dynamic linking
VCL_BEGIN_EXTERNAL_HEADERS
#ifdef VCL_ABI_WINAPI
#	include <windows.h>
#elif defined(VCL_ABI_POSIX)
#	include <dlfcn.h>
#endif
VCL_END_EXTERNAL_HEADERS

#ifdef VCL_ABI_WINAPI
HMODULE nvCompilerModule;
#elif defined(VCL_ABI_POSIX)
void* nvCompilerModule;
#endif

tNvCliCompileProgram nvCompileProgram;
tNvCliCompileLogFree nvCompileLogFree;
tNvCliCompiledProgramFree nvCompiledProgramFree;

namespace Vcl { namespace Tools { namespace Clc { namespace Nvidia
{
#ifdef VCL_ABI_WINAPI

	void print_error()
	{
		char* message = nullptr;
		FormatMessage
		(
			FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
			nullptr,
			GetLastError(),
			0,
			(LPSTR) &message,
			0,
			nullptr
		);
#	ifdef VCL_ABI_WIN64
		std::cerr << "nvcompiler.dll: " << message << std::endl;
#	else
		std::cerr << "nvcompiler32.dll: " << message << std::endl;
#	endif
		LocalFree(message);
	}

	bool loadCompiler()
	{
#	ifdef VCL_ABI_WIN64
		nvCompilerModule = LoadLibrary("nvcompiler.dll");
#	else 
		nvCompilerModule = LoadLibrary("nvcompiler32.dll");
#	endif
		if (!nvCompilerModule)
		{
			print_error();
			return false;
		}

		nvCompileProgram = (tNvCliCompileProgram) GetProcAddress(nvCompilerModule, "NvCliCompileProgram");
		if (!nvCompileProgram)
		{
			print_error();
			return false;
		}

		nvCompileLogFree = (tNvCliCompileLogFree) GetProcAddress(nvCompilerModule, "NvCliCompileLogFree");
		if (!nvCompileLogFree)
		{
			print_error();
			return false;
		}

		nvCompiledProgramFree = (tNvCliCompiledProgramFree) GetProcAddress(nvCompilerModule, "NvCliCompiledProgramFree");
		if (!nvCompiledProgramFree)
		{
			print_error();
			return false;
		}

		return true;
	}
	
	void releaseCompiler()
	{
		nvCompileProgram = nullptr;
		nvCompileLogFree = nullptr;
		nvCompiledProgramFree = nullptr;

		if (nvCompilerModule)
		{
			FreeLibrary(nvCompilerModule);
			nvCompilerModule = nullptr;
		}
	}
	
#elif defined(VCL_ABI_POSIX)
#endif
}}}}
