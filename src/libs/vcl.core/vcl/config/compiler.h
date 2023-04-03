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

#include <vcl/config/config.h>

// Check library configuration
#if defined __ICC || defined __ICL
#	define VCL_COMPILER_ICC
#	if (__INTEL_COMPILER < 1700)
#		warning "Minimum supported version is ICC 17. Good luck."
#	endif
#elif defined _MSC_VER && !defined __clang__
#	define VCL_COMPILER_MSVC
// Microsoft compiler versions
// MSVC++ 10.0  _MSC_VER == 1600 (Visual Studio 2010)
// MSVC++ 11.0  _MSC_VER == 1700 (Visual Studio 2012)
// MSVC++ 12.0  _MSC_VER == 1800 (Visual Studio 2013)
// MSVC++ 14.0  _MSC_VER == 1900 (Visual Studio 2015)
// MSVC++ 14.10 _MSC_VER == 1910 (Visual Studio 2017)
// MSVC++ 14.11 _MSC_VER == 1911 (Visual Studio 2017 Update 3)
// MSVC++ 14.12 _MSC_VER == 1912 (Visual Studio 2017 Update 5)
// MSVC++ 14.13 _MSC_VER == 1913 (Visual Studio 2017 Update 6)
// MSVC++ 14.14 _MSC_VER == 1914 (Visual Studio 2017 Update 7)
// MSVC++ 14.15 _MSC_VER == 1915 (Visual Studio 2017 Update 8)
// MSVC++ 14.16 _MSC_VER == 1916 (Visual Studio 2017 Update 9)
// MSVC++ 14.20 _MSC_VER == 1920 (Visual Studio 2019)
// MSVC++ 14.30 _MSC_VER == 1930 (Visual Studio 2022)
#	if (_MSC_VER < 1900)
#		warning "Minimum supported version is MSVC 2015. Good luck."
#	endif
#elif defined __clang__
#	define VCL_COMPILER_CLANG
#	if (__clang_major__ < 3) || (__clang_major__ == 3 && __clang_minor__ < 5)
#		warning "Minimum supported version is Clang 3.5. Good luck."
#	endif
#elif defined __GNUC__
#	define VCL_COMPILER_GNU
#	if (__GNUC__ < 5)
#		warning "Minimum supported version is GCC 5. Good luck."
#	endif
#endif

// Identify C++ standard
#if defined _MSVC_LANG
#	if _MSVC_LANG >= 202002L
#		define VCL_HAS_STDCXX20 1
#	endif
#	if _MSVC_LANG >= 201703L
#		define VCL_HAS_STDCXX17 1
#	endif
#	if _MSVC_LANG >= 201402L
#		define VCL_HAS_STDCXX14 1
#	endif
#	if _MSVC_LANG >= 201103L
#		define VCL_HAS_STDCXX11 1
#	endif
#elif defined __cplusplus
#	if __cplusplus >= 202002L
#		define VCL_HAS_STDCXX20 1
#	endif
#	if __cplusplus >= 201703L
#		define VCL_HAS_STDCXX17 1
#	endif
#	if __cplusplus >= 201402L
#		define VCL_HAS_STDCXX14 1
#	endif
#	if __cplusplus >= 201103L
#		define VCL_HAS_STDCXX11 1
#	endif
#endif
#ifndef VCL_HAS_STDCXX20
#	define VCL_HAS_STDCXX20 0
#endif
#ifndef VCL_HAS_STDCXX17
#	define VCL_HAS_STDCXX17 0
#endif
#ifndef VCL_HAS_STDCXX14
#	define VCL_HAS_STDCXX14 0
#endif
#ifndef VCL_HAS_STDCXX11
#	define VCL_HAS_STDCXX11 0
#endif

// Identify system ABI
#if defined VCL_COMPILER_ICC || defined VCL_COMPILER_MSVC || defined VCL_COMPILER_CLANG || defined VCL_COMPILER_GNU
#	if defined(_WIN32)
#		define VCL_ABI_WINAPI
#		if defined(_WIN64)
#			define VCL_ABI_WIN64
#		else
#			define VCL_ABI_WIN32
#		endif
#	endif

#	if defined(__unix) && __unix == 1
#		define VCL_ABI_POSIX
#	endif
#endif

// Identify CPU instruction set
#if defined VCL_COMPILER_ICC

// The Intel compiler only supports Intel64 and x86 instruction sets
#	if defined _M_X64 || defined __x86_64 || defined __x86_64__
#		define VCL_ARCH_X64
#	else
#		define VCL_ARCH_X86
#	endif

#elif defined VCL_COMPILER_MSVC
#	if defined _M_IX86
#		define VCL_ARCH_X86
#	endif

#	if defined _M_X64
#		define VCL_ARCH_X64
#	endif

#	if defined _M_ARM
#		define VCL_ARCH_ARM
#	endif

#	if defined _M_ARM64
#		define VCL_ARCH_ARM64
#	endif

#elif defined VCL_COMPILER_GNU || defined VCL_COMPILER_CLANG
#	if defined __i386__ || defined __i686__
#		define VCL_ARCH_X86
#	endif

#	if defined __x86_64
#		define VCL_ARCH_X64
#	endif

#	if defined __arm__
#		define VCL_ARCH_ARM
#	endif

#	if defined __aarch64__
#		define VCL_ARCH_ARM64
#	endif

#	if defined EMSCRIPTEN
#		define VCL_ARCH_WEBASM
#	endif

#endif

// Identify supported compiler features
#if defined VCL_COMPILER_MSVC

// Force inline
#	define VCL_STRONG_INLINE __forceinline

// Enter the debugger
#	define VCL_DEBUG_BREAK __debugbreak()

#	define VCL_CALLBACK __stdcall

#elif defined(VCL_COMPILER_GNU) || defined(VCL_COMPILER_CLANG)

// Inlining
#	define VCL_STRONG_INLINE inline

#	define VCL_DEBUG_BREAK __builtin_trap()

#	if defined(_MSC_VER) && defined(VCL_COMPILER_CLANG)
#		pragma clang diagnostic push
#		pragma clang diagnostic ignored "-Wreserved-id-macro"
#		define __ENABLE_MSVC_VECTOR_TYPES_IMP_DETAILS
#		pragma clang diagnostic pop
#		define VCL_CALLBACK __attribute__((__stdcall__))
#	else
#		define VCL_CALLBACK
#	endif // defined(_MSC_VER) && defined(VCL_COMPILER_CLANG)

// Add missing definition for max_align_t for compatibility with older clang version (3.4, 3.5)
#	if defined(VCL_COMPILER_CLANG) && !defined(_MSC_VER) && !defined(__APPLE_CC__)
#		if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L) || __cplusplus >= 201103L
#			if !defined(__CLANG_MAX_ALIGN_T_DEFINED) && !defined(_GCC_MAX_ALIGN_T) && !defined(__DEFINED_max_align_t)
typedef struct
{
	long long __clang_max_align_nonce1 __attribute__((__aligned__(__alignof__(long long))));
	long double __clang_max_align_nonce2 __attribute__((__aligned__(__alignof__(long double))));
} max_align_t;
#				define __DEFINED_max_align_t
#			endif
#		endif
#	endif

#elif defined(VCL_COMPILER_ICC)

#	define VCL_STRONG_INLINE __forceinline
#	define VCL_DEBUG_BREAK
#	define VCL_CALLBACK

#else // No compiler found
#	define VCL_STRONG_INLINE inline
#	define VCL_DEBUG_BREAK
#	define VCL_CALLBACK
#endif

////////////////////////////////////////////////////////////////////////////////
// Evaluate compiler feature support
////////////////////////////////////////////////////////////////////////////////

// alignas/alignof
#if defined(VCL_COMPILER_MSVC)
#	if (_MSC_VER <= 1800)
#		include <xkeycheck.h>
#		if defined alignof
#			undef alignof
#		endif
#		define alignof(x) __alignof(x)

#		define alignas(x) __declspec(align(x))
#	endif
#endif

// if constexpr
#if defined(VCL_COMPILER_MSVC)
#	if (_MSC_VER >= 1912 && VCL_HAS_STDCXX17)
#		define VCL_IF_CONSTEXPR if constexpr
#	else
#		define VCL_IF_CONSTEXPR if
#	endif
#elif defined(VCL_COMPILER_GNU)
#	if defined(__cpp_if_constexpr) && __cpp_if_constexpr >= 201606
#		define VCL_IF_CONSTEXPR if constexpr
#	else
#		define VCL_IF_CONSTEXPR if
#	endif
#elif defined(VCL_COMPILER_CLANG)
#	if __clang_major__ < 3 || (__clang_major__ == 3 && __clang_minor__ < 9) || __cplusplus < 201703l
#		define VCL_IF_CONSTEXPR if
#	else
#		define VCL_IF_CONSTEXPR if constexpr
#	endif
#endif

// noexcept
#if defined(VCL_COMPILER_MSVC)
#	if (_MSC_VER <= 1800)
#		define noexcept _NOEXCEPT
#		define VCL_NOEXCEPT_PARAM(param)
#	else
#		define VCL_NOEXCEPT_PARAM(param) noexcept(param)
#	endif // _MSC_VER
#elif defined(VCL_COMPILER_GNU) || defined(VCL_COMPILER_CLANG) || defined(VCL_COMPILER_ICC)
#	define VCL_NOEXCEPT_PARAM(param) noexcept(param)
#else
#	define VCL_NOEXCEPT_PARAM(param)
#endif

// thread_local
#if defined(VCL_COMPILER_MSVC)
#	if (_MSC_VER < 1900)
#		define thread_local __declspec(thread)
#	endif
#elif defined(VCL_COMPILER_GNU)
#	if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8)
#		define thread_local __thread
#	endif
#elif defined(VCL_COMPILER_CLANG)
#	if __clang_major__ < 3 || (__clang_major__ == 3 && __clang_minor__ < 3)
#		define thread_local __thread
#	endif
#endif

////////////////////////////////////////////////////////////////////////////////
// Evaluate standard library support
////////////////////////////////////////////////////////////////////////////////

// chrono
#if defined(VCL_COMPILER_MSVC)
#	if _MSC_VER >= 1700
#		define VCL_STL_CHRONO
#	endif
#elif defined(VCL_COMPILER_GNU) || defined(VCL_COMPILER_CLANG)
#	if __cplusplus >= 201103L
#		define VCL_STL_CHRONO
#	endif
#endif

////////////////////////////////////////////////////////////////////////////////
// Configure SIMD
////////////////////////////////////////////////////////////////////////////////
#if (defined(VCL_ARCH_X86) || defined(VCL_ARCH_X64) || defined(VCL_ARCH_WEBASM))

#	ifdef VCL_VECTORIZE_AVX512
#		ifndef VCL_VECTORIZE_AVX2
#			define VCL_VECTORIZE_AVX2
#		endif
#	endif
#	ifdef VCL_VECTORIZE_AVX2
#		ifndef VCL_VECTORIZE_AVX
#			define VCL_VECTORIZE_AVX
#		endif
#	endif
#	ifdef VCL_VECTORIZE_AVX
#		ifndef VCL_VECTORIZE_SSE4_2
#			define VCL_VECTORIZE_SSE4_2
#		endif
#	endif

#	ifdef VCL_VECTORIZE_SSE4_2
#		ifndef VCL_VECTORIZE_SSE4_1
#			define VCL_VECTORIZE_SSE4_1
#		endif
#	endif
#	ifdef VCL_VECTORIZE_SSE4_1
#		ifndef VCL_VECTORIZE_SSSE3
#			define VCL_VECTORIZE_SSSE3
#		endif
#	endif
#	ifdef VCL_VECTORIZE_SSSE3
#		ifndef VCL_VECTORIZE_SSE3
#			define VCL_VECTORIZE_SSE3
#		endif
#	endif
#	ifdef VCL_VECTORIZE_SSE3
#		ifndef VCL_VECTORIZE_SSE2
#			define VCL_VECTORIZE_SSE2
#		endif
#	endif
#	ifdef VCL_VECTORIZE_SSE2
#		ifndef VCL_VECTORIZE_SSE
#			define VCL_VECTORIZE_SSE
#		endif
#	endif

#	if defined VCL_VECTORIZE_AVX
extern "C"
{
#		include <immintrin.h>
}
#	elif defined(VCL_VECTORIZE_SSE)
extern "C"
{
#		ifdef VCL_VECTORIZE_SSE4_2
#			include <nmmintrin.h>
#		elif defined VCL_VECTORIZE_SSE4_1
#			include <smmintrin.h>
#		elif defined VCL_VECTORIZE_SSSE3
#			include <tmmintrin.h>
#		elif defined VCL_VECTORIZE_SSE3
#			include <pmmintrin.h>
#		elif defined VCL_VECTORIZE_SSE2
#			include <emmintrin.h>
#			include <xmmintrin.h>
#			include <mmintrin.h>
#		endif
}
#	endif // defined(VCL_VECTORIZE_SSE)

#elif (defined(VCL_ARCH_ARM) || defined(VCL_ARCH_ARM64)) && defined VCL_VECTORIZE_NEON
#	include <arm_neon.h>
#endif
