/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2015 Basil Fierz
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
#include <vcl/math/jacobisvd33_mcadams.h>

#ifdef VCL_VECTORIZE_SSE

// VCL
#	include <vcl/core/contract.h>

// McAdams SVD library
#	define USE_SSE_IMPLEMENTATION

#	define USE_ACCURATE_RSQRT_IN_JACOBI_CONJUGATION
// #define PERFORM_STRICT_QUATERNION_RENORMALIZATION
// #define PRINT_DEBUGGING_OUTPUT

#	define COMPUTE_V_AS_MATRIX
// #define COMPUTE_V_AS_QUATERNION
#	define COMPUTE_U_AS_MATRIX
// #define COMPUTE_U_AS_QUATERNION

// Disable runtime asserts usage of uninitialized variables. Necessary for constructs like 'var = xor(var, var)'
#	ifdef VCL_COMPILER_MSVC
#		pragma runtime_checks("u", off)
#		pragma warning(disable : 4700)
#	elif defined VCL_COMPILER_GNU
#		pragma GCC diagnostic push
#		pragma GCC diagnostic ignored "-Wuninitialized"
#		pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#		pragma GCC diagnostic ignored "-Wunused-variable"
#	elif defined VCL_COMPILER_CLANG
#		pragma clang diagnostic push
#		pragma clang diagnostic ignored "-Wmissing-prototypes"
#		pragma clang diagnostic ignored "-Wold-style-cast"
#		pragma clang diagnostic ignored "-Wuninitialized"
#		pragma clang diagnostic ignored "-Wunused-variable"
#	endif

#	include <vcl/math/mcadams/Singular_Value_Decomposition_Preamble.hpp>

namespace Vcl { namespace Mathematics {
	int McAdamsJacobiSVD(Eigen::Matrix<float4, 3, 3>& A, Eigen::Matrix<float4, 3, 3>& U, Eigen::Matrix<float4, 3, 3>& V, unsigned int sweeps)
	{
		using ::sqrt;

#	define JACOBI_CONJUGATION_SWEEPS (int)sweeps

#	include <vcl/math/mcadams/Singular_Value_Decomposition_Kernel_Declarations.hpp>

		ENABLE_SSE_IMPLEMENTATION(Va11 = _mm_loadu_ps(reinterpret_cast<float*>(&A(0, 0)));)
		ENABLE_SSE_IMPLEMENTATION(Va21 = _mm_loadu_ps(reinterpret_cast<float*>(&A(1, 0)));)
		ENABLE_SSE_IMPLEMENTATION(Va31 = _mm_loadu_ps(reinterpret_cast<float*>(&A(2, 0)));)
		ENABLE_SSE_IMPLEMENTATION(Va12 = _mm_loadu_ps(reinterpret_cast<float*>(&A(0, 1)));)
		ENABLE_SSE_IMPLEMENTATION(Va22 = _mm_loadu_ps(reinterpret_cast<float*>(&A(1, 1)));)
		ENABLE_SSE_IMPLEMENTATION(Va32 = _mm_loadu_ps(reinterpret_cast<float*>(&A(2, 1)));)
		ENABLE_SSE_IMPLEMENTATION(Va13 = _mm_loadu_ps(reinterpret_cast<float*>(&A(0, 2)));)
		ENABLE_SSE_IMPLEMENTATION(Va23 = _mm_loadu_ps(reinterpret_cast<float*>(&A(1, 2)));)
		ENABLE_SSE_IMPLEMENTATION(Va33 = _mm_loadu_ps(reinterpret_cast<float*>(&A(2, 2)));)

#	include <vcl/math/mcadams/Singular_Value_Decomposition_Main_Kernel_Body.hpp>

		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&U(0, 0)), Vu11);)
		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&U(1, 0)), Vu21);)
		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&U(2, 0)), Vu31);)
		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&U(0, 1)), Vu12);)
		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&U(1, 1)), Vu22);)
		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&U(2, 1)), Vu32);)
		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&U(0, 2)), Vu13);)
		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&U(1, 2)), Vu23);)
		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&U(2, 2)), Vu33);)

		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&V(0, 0)), Vv11);)
		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&V(1, 0)), Vv21);)
		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&V(2, 0)), Vv31);)
		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&V(0, 1)), Vv12);)
		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&V(1, 1)), Vv22);)
		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&V(2, 1)), Vv32);)
		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&V(0, 2)), Vv13);)
		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&V(1, 2)), Vv23);)
		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&V(2, 2)), Vv33);)

		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&A(0, 0)), Va11);)
		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&A(1, 1)), Va22);)
		ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(reinterpret_cast<float*>(&A(2, 2)), Va33);)

		return JACOBI_CONJUGATION_SWEEPS * 3 + 3;
	}
}}
#	ifdef VCL_COMPILER_MSVC
#		pragma warning(default : 4700)
#		pragma runtime_checks("u", restore)
#	elif defined VCL_COMPILER_GNU
#		pragma GCC diagnostic pop
#	elif defined VCL_COMPILER_CLANG
#		pragma clang diagnostic pop
#	endif
#endif // defined(VCL_VECTORIZE_SSE)
