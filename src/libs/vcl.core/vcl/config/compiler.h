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

#include <type_traits>

#include <vcl/config/config.h>

// Check library configuration
#if defined _MSC_VER && !defined __clang__ && !defined VCL_COMPILER_MSVC
#	error "VCL was not configured for MSVC"
#elif defined __clang__ && !defined VCL_COMPILER_CLANG
#	error "VCL was not configured for CLANG"
#elif defined __GNUC__ && !defined VCL_COMPILER_GNU
#	error "VCL was not configured for the GNU C++ compiler"
#elif defined __INTEL_COMPILER && !defined VCL_COMPILER_ICL
#	error "VCL was not configured for the Intel C++ compiler"
#endif

#if defined (VCL_COMPILER_MSVC)
#	if defined (_WIN32)
#		define VCL_ABI_WINAPI
#		if defined (_WIN64)
#			define VCL_ABI_WIN64
#		else
#			define VCL_ABI_WIN32
#		endif // _WIN64
#	endif // _WIN32
#	if (defined(_M_IX86))
#		define VCL_ARCH_X86
#	endif // _M_IX86

#	if (defined(_M_X64))
#		define VCL_ARCH_X64
#	endif // _M_X64

#	if (defined(_M_ARM))
#		define VCL_ARCH_ARM
#	endif // _M_ARM

// Inlining
#	define VCL_STRONG_INLINE __forceinline

#	define VCL_DEBUG_BREAK __debugbreak()

#	define VCL_ALIGN(alignment) __declspec(align(alignment))

#	define VCL_CALLBACK __stdcall

// Support for the alignment operator
#	if (_MSC_VER <= 1800)
#		include <xkeycheck.h>
#		if defined alignof
#			undef alignof
#		endif /* alignof */
//#		define alignof(x) __alignof(decltype(*static_cast<std::remove_reference<std::remove_pointer<(x)>::type>::type*>(0)))
#		define alignof(x) __alignof(x)
#	endif /* _MSC_VER <= 1800 */

// Enable the noexcept-keyword
#	if (_MSC_VER <= 1800)
#		define noexcept _NOEXCEPT
#		define VCL_NOEXCEPT_PARAM(param)
#	else
#		define VCL_NOEXCEPT_PARAM(param) noexcept(param)
#	endif /* _MSC_VER <= 1800 */

// Enable the thread_local-keyword
#	if (_MSC_VER <= 1900)
#		define thread_local __declspec(thread)
#	endif /* _MSC_VER <= 1900 */

// Enable constexpr on certain Microsoft compilers
#	define VCL_CONSTEXPR_CPP11 constexpr
#	define VCL_CONSTEXPR_CPP14

// STL support
#	define VCL_STL_CHRONO

#elif defined (VCL_COMPILER_GNU) || defined (VCL_COMPILER_CLANG)
#	if defined (_WIN32)
#		define VCL_ABI_WINAPI
#		if defined (_WIN64)
#			define VCL_ABI_WIN64
#		else
#			define VCL_ABI_WIN32
#		endif /* _WIN64 */
#	elif __unix == 1
#		define VCL_ABI_POSIX
#	endif /* _WIN32 */
#	if (defined(__i686__))
#		define VCL_ARCH_X86
#	endif /* _M_IX86 */
#	if (defined(__x86_64))
#		define VCL_ARCH_X64
#	endif /* _M_X64 */

// Inlining
#	define VCL_STRONG_INLINE inline

#	define VCL_DEBUG_BREAK __builtin_trap()

#	define VCL_ALIGN(x) __attribute__((aligned(x)))

#	define VCL_CALLBACK __attribute__ ((__stdcall__))

#	define VCL_NOEXCEPT_PARAM(param) noexcept(param)

#	if defined(_MSC_VER) && defined(VCL_COMPILER_CLANG)
#		define __ENABLE_MSVC_VECTOR_TYPES_IMP_DETAILS
#	endif // defined(_MSC_VER) && defined(VCL_COMPILER_CLANG)

#	define VCL_CONSTEXPR_CPP11 constexpr
#	define VCL_CONSTEXPR_CPP14 constexpr

// STL support
//#	if __cplusplus >= 201103L
#		define VCL_STL_CHRONO
//#	endif

#else // No compiler found
#	define VCL_STRONG_INLINE inline
#	define VCL_DEBUG_BREAK
#	define VCL_ALIGN(x)
#	define VCL_CALLBACK
#	define VCL_NOEXCEPT_PARAM(param)
#endif

// Configure macros for SIMD
#if (defined(VCL_ARCH_X86) || defined(VCL_ARCH_X64))

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
#		ifdef VCL_VECTORIZE_AVX2
#			include <immintrin.h>
#		else
#			include <immintrin.h>
#		endif
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

#elif defined VCL_ARCH_ARM && defined VCL_VECTORIZE_NEON
#	include <arm_neon.h>
#endif

// Implement missing standard function
#if defined (VCL_COMPILER_MSVC)
		
// Support for fmin/fmax with low overhead
#	if (_MSC_VER < 1800)
namespace std
{
#		if (defined(VCL_VECTORIZE_AVX) || defined(VCL_VECTORIZE_SSE))
	inline float fmin(float x, float y)
	{
		float z;
		_mm_store_ss(&z, _mm_min_ss(_mm_set_ss(x), _mm_set_ss(y)));
		return z;
	}
	inline double fmin(double x, double y)
	{
		double z;
		_mm_store_sd(&z, _mm_min_sd(_mm_set_sd(x), _mm_set_sd(y)));
		return z;
	}
	inline float fmax(float x, float y)
	{
		float z;
		_mm_store_ss(&z, _mm_max_ss(_mm_set_ss(x), _mm_set_ss(y)));
		return z;
	}
	inline double fmax(double x, double y)
	{
		double z;
		_mm_store_sd(&z, _mm_max_sd(_mm_set_sd(x), _mm_set_sd(y)));
		return z;
	}
#		endif
}
#	endif // _MSC_VER < 1800
#endif
