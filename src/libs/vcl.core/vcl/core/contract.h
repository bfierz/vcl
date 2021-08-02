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

#if defined(VCL_USE_CONTRACTS) && (defined(DEBUG) || defined(_DEBUG))
#	ifndef VCL_NO_CONTRACTS
#		define VCL_CONTRACT
#	endif
#else
#	ifdef VCL_CHECK_CONTRACTS
#		define VCL_CONTRACT
#	endif
#endif

#ifdef VCL_CONTRACT

// C++ standard library
#	include <string>

namespace Vcl { namespace Assert {
	bool handleAssert(const char* type, const char* file, size_t line, const char* expr, const char* description, const char* note, bool& ignore);
	bool handleAssert(const char* type, const char* file, size_t line, const char* expr, const char* description, const std::string& note, bool& ignore);
}}
	/*!
	 *	\brief Defintion of a assert
	 *
	 *	This macro defines when the assert handler called and if the debugger should be invoked
	 */
#	define vcl_assert(type, expr, description)                                    \
	if (!(expr))                                                                  \
	{                                                                             \
		static bool ignoreAlways = false;                                         \
		if (!ignoreAlways && Vcl::Assert::handleAssert(type, __FILE__, __LINE__,  \
			VCL_PP_STRINGIZE(expr), description, nullptr, ignoreAlways))          \
		{                                                                         \
			VCL_DEBUG_BREAK;                                                      \
		}                                                                         \
	}

#	define vcl_assert_ex(type, expr, description, note)                           \
	if (!(expr))                                                                  \
	{                                                                             \
		static bool ignoreAlways = false;                                         \
		if (!ignoreAlways && Vcl::Assert::handleAssert(type, __FILE__, __LINE__,  \
			VCL_PP_STRINGIZE(expr), description, note, ignoreAlways))             \
		{                                                                         \
			VCL_DEBUG_BREAK;                                                      \
		}                                                                         \
	}

	// Define wrappers around the contract method
#	ifndef VclCheck
#		define VclCheck(expr, description)              vcl_assert("Check",   (expr), description)
#	endif									     
#	ifndef VclCheckEx							     
#		define VclCheckEx(expr, description, note)   vcl_assert_ex("Check",   (expr), description, note)
#	endif									     
#	ifndef VclRequire							     
#		define VclRequire(expr, description)            vcl_assert("Require", (expr), description)
#	endif
#	ifndef VclRequireEx
#		define VclRequireEx(expr, description, note) vcl_assert_ex("Require", (expr), description, note)
#	endif
#	ifndef VclEnsure
#		define VclEnsure(expr, description)             vcl_assert("Ensure",  (expr), description)
#	endif
#	ifndef VclEnsureEx
#		define VclEnsureEx(expr, description, note)  vcl_assert_ex("Ensure",  (expr), description, note)
#	endif
#	ifndef VclDebugError
#		define VclDebugError(description)               vcl_assert("Error",   (VCL_EVAL_FALSE), description)
#	endif
#	ifndef VclDebugErrorEx
#		define VclDebugErrorEx(description, note)    vcl_assert_ex("Error",   (VCL_EVAL_FALSE), description, note)
#	endif
#	ifndef VclAssertBlock
#		define VclAssertBlock if(VCL_EVAL_TRUE)
#	endif
#else
#	ifndef VclCheck
#		define VclCheck(expr, description)
#	endif
#	ifndef VclCheckEx
#		define VclCheckEx(expr, description, note)
#	endif
#	ifndef VclRequire
#		define VclRequire(expr, description)
#	endif
#	ifndef VclRequireEx
#		define VclRequireEx(expr, description, note)
#	endif
#	ifndef VclEnsure
#		define VclEnsure(expr, description)
#	endif
#	ifndef VclEnsureEx
#		define VclEnsureEx(expr, description, note)
#	endif
#	ifndef VclDebugError
#		define VclDebugError(description)
#	endif
#	ifndef VclDebugErrorEx
#		define VclDebugErrorEx(description, note)
#	endif
#	ifndef VclAssertBlock
#		define VclAssertBlock if (VCL_EVAL_FALSE)
#	endif
#endif
