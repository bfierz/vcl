/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014-2015 Basil Fierz
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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

VCL_BEGIN_EXTERNAL_HEADERS
#ifdef VCL_CUDA_SUPPORT
#	include <cuda.h>
#endif // VCL_CUDA_SUPPORT
	
#ifdef VCL_DEBUG
#	include <iostream>
#endif // VCL_DEBUG
VCL_END_EXTERNAL_HEADERS

/*
 *	Support macros
 */
#ifndef VCL_CU_SAFE_CALL
#	ifdef VCL_DEBUG
#		define VCL_CU_SAFE_CALL(call) do { CUresult error0507 = call; if (error0507 != CUDA_SUCCESS) { std::cout << "CUDA Error\tFile: " << __FILE__ << ", " << __LINE__ << ": " << error0507 << std::endl; __debugbreak(); }} while (VCL_EVAL_FALSE)
#	else
#		define VCL_CU_SAFE_CALL(call) call
#	endif // VCL_DEBUG
#endif // VCL_CL_SAFE_CALL
