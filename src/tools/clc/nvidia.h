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
#include <vcl/config/global.h>

// Prototypes of the NV compiler
// Description of prototypes is taken from: https://github.com/ljbade/clcc
//void NvCliCompileLogFree(const char* compileLog);
//void NvCliCompiledProgramFree(const char* compiledProgram);

typedef int (*tNvCliCompileProgram)
(
	const char** sourceStrings, unsigned int sourceStringsCount, const size_t* sourceStringsLengths,
	const char*  compilerOptions, char** compileLogRet, char** compiledProgramRet
);
typedef void (*tNvCliCompileLogFree) (const char* compileLog);
typedef void (*tNvCliCompiledProgramFree) (const char* compiledProgram);

namespace Vcl { namespace Tools { namespace Clc { namespace Nvidia
{
	bool loadCompiler();
	void releaseCompiler();

	int compileProgram
	(
		const char** sourceStrings, unsigned int sourceStringsCount, const size_t* sourceStringsLengths,
		const char*  compilerOptions, char** compileLogRet, char** compiledProgramRet
	);
}}}}
