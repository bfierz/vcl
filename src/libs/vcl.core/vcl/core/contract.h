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

// C runtime library
#include <stdio.h>

#if defined (DEBUG) || defined (_DEBUG)
#	ifndef VCL_NO_CONTRACTS
#		define VCL_CONTRACT
#	endif
#else
#	ifdef VCL_CHECK_CONTRACTS
#		define VCL_CONTRACT        
#	endif
#endif

#ifdef VCL_CONTRACT
#	ifdef __GNUC__
#		define VCL_CONTRACT_SPRINTF snprintf
#	else
#		define VCL_CONTRACT_SPRINTF sprintf_s
#	endif

namespace Vcl { namespace Assert
{
	const char* format();
	const char* format(char* format, ...);

	bool handler(const char* title, const char* message, bool* b);
}}
	/*!
	 *	\brief Defintion of a assert
	 *
	 *	This macro defines when the assert handler called and if the debugger should be invoked
	 */
#	define vcl_assert(type, expr, description, ...)                        \
	{                                                                      \
		static bool ignoreAlways = false;                                  \
		if (!ignoreAlways && !(expr))                                      \
		{                                                                  \
			char msgbuf[1024];                                             \
			VCL_CONTRACT_SPRINTF(msgbuf,"%s in %s:%d:\n '%s' \n %s \n %s \n", type, __FILE__, __LINE__, VCL_PP_STRINGIZE(expr), description, Vcl::Assert::format(__VA_ARGS__)); \
			if(Vcl::Assert::handler(description, msgbuf, &ignoreAlways))   \
			{                                                              \
				VCL_DEBUG_BREAK;                                           \
			}                                                              \
		}                                                                  \
	}

	// Define wrappers around the contract method
	#define Check(expr, description,...)   vcl_assert("Check",   (expr), description, __VA_ARGS__)
	#define Require(expr, description,...) vcl_assert("Require", (expr), description, __VA_ARGS__)
	#define Ensure(expr, description,...)  vcl_assert("Ensure",  (expr), description, __VA_ARGS__)
	#define DebugError(description,...)    vcl_assert("Error",   (VCL_EVAL_FALSE), description, __VA_ARGS__)
	#define AssertBlock if(VCL_EVAL_TRUE)
#else
	#define Check(expr, description,...)
	#define Require(expr, description,...)
	#define Ensure(expr, description,...)
	#define DebugError(description,...) 
	#define AssertBlock if(VCL_EVAL_FALSE)   
#endif
