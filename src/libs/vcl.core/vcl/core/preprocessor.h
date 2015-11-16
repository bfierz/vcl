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

// Preprocessor programming
// Original source: http://altdevblogaday.com/wp-content/uploads/2011/10/Assert.txt

/// concatenates tokens, even when the tokens are macros themselves
#define VCL_PP_JOIN_2(_0, _1)																	VCL_PP_JOIN(_0, _1)
#define VCL_PP_JOIN_3(_0, _1, _2)																VCL_PP_JOIN_2(VCL_PP_JOIN_2(_0, _1), _2)
#define VCL_PP_JOIN_4(_0, _1, _2, _3)															VCL_PP_JOIN_2(VCL_PP_JOIN_3(_0, _1, _2), _3)
#define VCL_PP_JOIN_5(_0, _1, _2, _3, _4)														VCL_PP_JOIN_2(VCL_PP_JOIN_4(_0, _1, _2, _3), _4)
#define VCL_PP_JOIN_6(_0, _1, _2, _3, _4, _5)													VCL_PP_JOIN_2(VCL_PP_JOIN_5(_0, _1, _2, _3, _4), _5)
#define VCL_PP_JOIN_7(_0, _1, _2, _3, _4, _5, _6)												VCL_PP_JOIN_2(VCL_PP_JOIN_6(_0, _1, _2, _3, _4, _5), _6)
#define VCL_PP_JOIN_8(_0, _1, _2, _3, _4, _5, _6, _7)											VCL_PP_JOIN_2(VCL_PP_JOIN_7(_0, _1, _2, _3, _4, _5, _6), _7)
#define VCL_PP_JOIN_9(_0, _1, _2, _3, _4, _5, _6, _7, _8)										VCL_PP_JOIN_2(VCL_PP_JOIN_8(_0, _1, _2, _3, _4, _5, _6, _7), _8)
#define VCL_PP_JOIN_10(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9)									VCL_PP_JOIN_2(VCL_PP_JOIN_9(_0, _1, _2, _3, _4, _5, _6, _7, _8), _9)
#define VCL_PP_JOIN_11(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10)								VCL_PP_JOIN_2(VCL_PP_JOIN_10(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9), _10)
#define VCL_PP_JOIN_12(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11)						VCL_PP_JOIN_2(VCL_PP_JOIN_11(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10), _11)
#define VCL_PP_JOIN_13(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12)					VCL_PP_JOIN_2(VCL_PP_JOIN_12(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11), _12)
#define VCL_PP_JOIN_14(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13)				VCL_PP_JOIN_2(VCL_PP_JOIN_13(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12), _13)
#define VCL_PP_JOIN_15(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14)			VCL_PP_JOIN_2(VCL_PP_JOIN_14(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13), _14)
#define VCL_PP_JOIN_16(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15)	VCL_PP_JOIN_2(VCL_PP_JOIN_15(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14), _15)


/// chooses a value based on a condition
#define VCL_PP_IF_0(t, f)			f
#define VCL_PP_IF_1(t, f)			t
#define VCL_PP_IF(cond, t, f)		VCL_PP_JOIN_2(VCL_PP_IF_, VCL_PP_TO_BOOL(cond))(t, f)


/// converts a condition into a boolean 0 (=false) or 1 (=true)
#define VCL_PP_TO_BOOL_0 0
#define VCL_PP_TO_BOOL_1 1
#define VCL_PP_TO_BOOL_2 1
#define VCL_PP_TO_BOOL_3 1
#define VCL_PP_TO_BOOL_4 1
#define VCL_PP_TO_BOOL_5 1
#define VCL_PP_TO_BOOL_6 1
#define VCL_PP_TO_BOOL_7 1
#define VCL_PP_TO_BOOL_8 1
#define VCL_PP_TO_BOOL_9 1
#define VCL_PP_TO_BOOL_10 1
#define VCL_PP_TO_BOOL_11 1
#define VCL_PP_TO_BOOL_12 1
#define VCL_PP_TO_BOOL_13 1
#define VCL_PP_TO_BOOL_14 1
#define VCL_PP_TO_BOOL_15 1
#define VCL_PP_TO_BOOL_16 1

#define VCL_PP_TO_BOOL(x)		VCL_PP_JOIN_2(VCL_PP_TO_BOOL_, x)


/// Returns 1 if the arguments to the variadic macro are separated by a comma, 0 otherwise.
#define VCL_PP_HAS_COMMA(...)							VCL_PP_HAS_COMMA_EVAL(VCL_PP_HAS_COMMA_ARGS(__VA_ARGS__, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0))
#define VCL_PP_HAS_COMMA_EVAL(...)						__VA_ARGS__
#define VCL_PP_HAS_COMMA_ARGS(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, ...) _16


/// Returns 1 if the argument list to the variadic macro is empty, 0 otherwise.
#define VCL_PP_IS_EMPTY(...)													\
	VCL_PP_HAS_COMMA															\
	(																			\
		VCL_PP_JOIN_5															\
		(																		\
			VCL_PP_IS_EMPTY_CASE_,												\
			VCL_PP_HAS_COMMA(__VA_ARGS__),										\
			VCL_PP_HAS_COMMA(VCL_PP_IS_EMPTY_BRACKET_TEST __VA_ARGS__),			\
			VCL_PP_HAS_COMMA(__VA_ARGS__ (~)),									\
			VCL_PP_HAS_COMMA(VCL_PP_IS_EMPTY_BRACKET_TEST __VA_ARGS__ (~))		\
		)																		\
	)

#define VCL_PP_IS_EMPTY_CASE_0001			,
#define VCL_PP_IS_EMPTY_BRACKET_TEST(...)	,


/// Expand any number of arguments into a list of operations called with those arguments
#define VCL_PP_VA_EXPAND_ARGS_0(op, empty)
#define VCL_PP_VA_EXPAND_ARGS_1(op, a1)																			op(a1, 0)
#define VCL_PP_VA_EXPAND_ARGS_2(op, a1, a2)																		op(a1, 0) op(a2, 1)
#define VCL_PP_VA_EXPAND_ARGS_3(op, a1, a2, a3)																	op(a1, 0) op(a2, 1) op(a3, 2)
#define VCL_PP_VA_EXPAND_ARGS_4(op, a1, a2, a3, a4)																op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3)
#define VCL_PP_VA_EXPAND_ARGS_5(op, a1, a2, a3, a4, a5)															op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4)
#define VCL_PP_VA_EXPAND_ARGS_6(op, a1, a2, a3, a4, a5, a6)														op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5)
#define VCL_PP_VA_EXPAND_ARGS_7(op, a1, a2, a3, a4, a5, a6, a7)													op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6)
#define VCL_PP_VA_EXPAND_ARGS_8(op, a1, a2, a3, a4, a5, a6, a7, a8)												op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6) op(a8, 7)
#define VCL_PP_VA_EXPAND_ARGS_9(op, a1, a2, a3, a4, a5, a6, a7, a8, a9)											op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6) op(a8, 7) op(a9, 8)
#define VCL_PP_VA_EXPAND_ARGS_10(op, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)									op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6) op(a8, 7) op(a9, 8) op(a10, 9)
#define VCL_PP_VA_EXPAND_ARGS_11(op, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)								op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6) op(a8, 7) op(a9, 8) op(a10, 9) op(a11, 10)
#define VCL_PP_VA_EXPAND_ARGS_12(op, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12)							op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6) op(a8, 7) op(a9, 8) op(a10, 9) op(a11, 10) op(a12, 11)
#define VCL_PP_VA_EXPAND_ARGS_13(op, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13)					op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6) op(a8, 7) op(a9, 8) op(a10, 9) op(a11, 10) op(a12, 11) op(a13, 12)
#define VCL_PP_VA_EXPAND_ARGS_14(op, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14)				op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6) op(a8, 7) op(a9, 8) op(a10, 9) op(a11, 10) op(a12, 11) op(a13, 12) op(a14, 13)
#define VCL_PP_VA_EXPAND_ARGS_15(op, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15)			op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6) op(a8, 7) op(a9, 8) op(a10, 9) op(a11, 10) op(a12, 11) op(a13, 12) op(a14, 13) op(a15, 14)
#define VCL_PP_VA_EXPAND_ARGS_16(op, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16)		op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6) op(a8, 7) op(a9, 8) op(a10, 9) op(a11, 10) op(a12, 11) op(a13, 12) op(a14, 13) op(a15, 14) op(a16, 15)

#define VCL_PP_VA_EXPAND_ARGS(op, ...)		VCL_PP_JOIN_2(VCL_PP_VA_EXPAND_ARGS_, VCL_PP_VA_NUM_ARGS(__VA_ARGS__)) VCL_PP_PASS_VA(op, __VA_ARGS__)


/// Turns any legal C++ expression into nothing
#define VCL_UNUSED_IMPL(symExpr, n)					, (void)sizeof(symExpr)
#define VCL_UNUSED(...)								(void)sizeof(true) VCL_PP_EXPAND_ARGS VCL_PP_PASS_VA(VCL_UNUSED_IMPL, __VA_ARGS__)


/// Breaks into the debugger (if it is attached)
#ifdef VCL_DEBUG
#	define VCL_BREAKPOINT								((IsDebuggerPresent() != 0) ? __debugbreak() : VCL_UNUSED(true))
#else
#	define VCL_BREAKPOINT								VCL_UNUSED(true)
#endif

// Define an anonymous variable
#ifdef __COUNTER__
#	define VCL_ANONYMOUS_VARIABLE(str) VCL_PP_JOIN(str, __COUNTER__)
#else
#	define VCL_ANONYMOUS_VARIABLE(str) VCL_PP_JOIN(str, __LINE__)
#endif
