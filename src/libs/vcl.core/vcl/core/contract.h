/* 
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014 Basil Fierz
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

// C++ standard library
#include <cassert>

#if defined (DEBUG) || defined (_DEBUG)
#	define debug_printf(x) printf(x);
#	ifndef VCL_NO_CONTRACTS
#		define VCL_CONTRACT
#	endif
#else
#	define debug_printf(x)
#	ifdef VCL_CHECK_CONTRACTS
#		define VCL_CONTRACT        
#	endif
#endif


#ifdef VCL_CONTRACT

	#ifdef __GNUC__
		#define VCL_CONTRACT_SPRINTF snprintf
	#else
		#define VCL_CONTRACT_SPRINTF sprintf_s
	#endif

	// Definitions of contracts
	#define Check(expr, description,...) assert((expr))
	#define Require(expr, description,...) assert((expr))
	#define Ensure(expr, description,...) assert((expr))
	#define DebugError(description,...) assert((expr))
	#define AssertBlock if(VCL_EVAL_TRUE)
#else
	#define Check(expr, description,...)
	#define Require(expr, description,...)
	#define Ensure(expr, description,...)
	#define DebugError(description,...) 
	#define AssertBlock if(VCL_EVAL_FALSE)   
#endif
