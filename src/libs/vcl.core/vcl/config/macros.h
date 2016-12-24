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

#include <vcl/config/compiler.h>

// Note: Some of the macros in this file are based on the description in
// http://molecularmusings.wordpress.com/2011/07/12/a-plethora-of-macros/

#ifdef _DEBUG
#	define VCL_DEBUG
#endif // _DEBUG

#ifdef NDEBUG
#	ifdef VCL_DEBUG
#		undef VCL_DEBUG
#	endif // VCL_DEBUG
#endif // NDEBUG

#ifndef NOMINMAX
#	define NOMINMAX
#endif // NOMINMAX

#define VCL_UNREFERENCED_PARAMETER(param) ((void) param)

#define VCL_SAFE_DELETE(ptr) if (ptr != NULL) { delete(ptr); ptr = NULL; }
#define VCL_SAFE_DELETE_ARRAY(ptr) if (ptr != NULL) { delete[](ptr); ptr = NULL; }

#define VCL_EVAL_TRUE  (std::sqrt(2.0f) > 0.0f)
#define VCL_EVAL_FALSE (std::sqrt(2.0f) < 0.0f)

// Stringizes a string, even macros
#define VCL_PP_STRINGIZE_HELPER(token)    #token
#define VCL_PP_STRINGIZE(str)             VCL_PP_STRINGIZE_HELPER(str)

// Concatenates two strings, even when the strings are macros themselves
#define VCL_PP_JOIN(x, y)                    VCL_PP_JOIN_HELPER(x, y)
#define VCL_PP_JOIN_HELPER(x, y)             VCL_PP_JOIN_HELPER_HELPER(x, y)
#define VCL_PP_JOIN_HELPER_HELPER(x, y)      x##y

// VCL_VA_NUM_ARGS() is a very nifty macro to retrieve the number of arguments handed to a variable-argument macro
// unfortunately, VS 2010 (up to at least 2015) still has this compiler bug which treats a __VA_ARGS__ argument as being one single parameter:
// https://connect.microsoft.com/VisualStudio/feedback/details/521844/variadic-macro-treating-va-args-as-a-single-parameter-for-other-macros#details
#ifdef VCL_COMPILER_MSVC
#	define VCL_PP_VA_NUM_ARGS_HELPER(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#	define VCL_PP_VA_NUM_ARGS_REVERSE_SEQUENCE   10, 9, 8, 7, 6, 5, 4, 3, 2, 1
#	define VCL_PP_LEFT_PARENTHESIS (
#	define VCL_PP_RIGHT_PARENTHESIS )
#	define VCL_PP_VA_NUM_ARGS(...) VCL_PP_VA_NUM_ARGS_HELPER VCL_PP_LEFT_PARENTHESIS __VA_ARGS__, VCL_PP_VA_NUM_ARGS_REVERSE_SEQUENCE VCL_PP_RIGHT_PARENTHESIS
#else
#	define VCL_PP_VA_NUM_ARGS_HELPER(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#	define VCL_PP_VA_NUM_ARGS(...) VCL_PP_VA_NUM_ARGS_HELPER(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#endif

// VCL_PASS_VA passes __VA_ARGS__ as multiple parameters to another macro, working around the above-mentioned bug
#ifdef VCL_COMPILER_MSVC
#	define VCL_PP_PASS_VA(...)	VCL_PP_LEFT_PARENTHESIS __VA_ARGS__ VCL_PP_RIGHT_PARENTHESIS
#else
#	define VCL_PP_PASS_VA(...)	( __VA_ARGS__ )
#endif

/*
 *	Compiler specific macros
 */
#ifdef VCL_COMPILER_MSVC
#	define __VCL_CONFIG_MACROS_STR2__(x) #x
#	define __VCL_CONFIG_MACROS_STR1__(x) __VCL_CONFIG_MACROS_STR2__(x)
#	define __VCL_CONFIG_MACROS_LOC_WARNING__ __FILE__ "("__VCL_CONFIG_MACROS_STR1__(__LINE__)") : warning Vcl: "
#	define __VCL_CONFIG_MACROS_LOC_ERROR__ __FILE__ "("__VCL_CONFIG_MACROS_STR1__(__LINE__)") : error Vcl: "
#	define __VCL_CONFIG_MACROS_LOC_MESSAGE__ __FILE__ "("__VCL_CONFIG_MACROS_STR1__(__LINE__)") : log Vcl: "

#	define VCL_WARNING(msg) __pragma(message(__VCL_CONFIG_MACROS_LOC_WARNING__#msg))
#	define VCL_ERROR(msg) __pragma(message(__VCL_CONFIG_MACROS_LOC_ERROR__#msg))
#	define VCL_MESSAGE(msg) __pragma(message(__VCL_CONFIG_MACROS_LOC_MESSAGE__#msg))

// Used in switch-statements whose default-case can never be reached, resulting in more optimal code
#	define VCL_NO_SWITCH_DEFAULT __assume(0)
#else
#	define VCL_WARNING(msg)
#	define VCL_ERROR(msg)
#	define VCL_MESSAGE(msg)

#	define VCL_NO_SWITCH_DEFAULT
#endif // VCL_COMPILER_MSVC


#ifdef VCL_COMPILER_MSVC
#	define VCL_BEGIN_EXTERNAL_HEADERS __pragma(warning(push, 1))
#	define VCL_END_EXTERNAL_HEADERS __pragma(warning(pop))
#else
#	define VCL_BEGIN_EXTERNAL_HEADERS
#	define VCL_END_EXTERNAL_HEADERS
#endif // VCL_COMPILER_MSVC

// Logic functions
#define implies(a,b) (!(a) || (b))
