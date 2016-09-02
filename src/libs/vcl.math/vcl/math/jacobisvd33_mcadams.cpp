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
#include <vcl/core/contract.h>
#include <vcl/math/math.h>

// McAdams SVD library
#define USE_SCALAR_IMPLEMENTATION

#define USE_ACCURATE_RSQRT_IN_JACOBI_CONJUGATION
// #define PERFORM_STRICT_QUATERNION_RENORMALIZATION
// #define PRINT_DEBUGGING_OUTPUT
// #define HAS_RSQRT

#define COMPUTE_V_AS_MATRIX
// #define COMPUTE_V_AS_QUATERNION
#define COMPUTE_U_AS_MATRIX
// #define COMPUTE_U_AS_QUATERNION

#include <vcl/math/mcadams/Singular_Value_Decomposition_Preamble.hpp>

namespace Vcl { namespace Mathematics
{
#ifdef VCL_COMPILER_MSVC
#	pragma runtime_checks( "u", off )  // Disable runtime asserts usage of uninitialized variables. Necessary for constructs like 'var = xor(var, var)'
#endif
	int McAdamsJacobiSVD(Eigen::Matrix<float, 3, 3>& A, Eigen::Matrix<float, 3, 3>& U, Eigen::Matrix<float, 3, 3>& V, unsigned int sweeps)
	{
		using ::sqrt;
		using ::rsqrt;

#define JACOBI_CONJUGATION_SWEEPS (int) sweeps


#include <vcl/math/mcadams/Singular_Value_Decomposition_Kernel_Declarations.hpp>

		ENABLE_SCALAR_IMPLEMENTATION(Sa11.f = A(0, 0);)
		ENABLE_SCALAR_IMPLEMENTATION(Sa21.f = A(1, 0);)
		ENABLE_SCALAR_IMPLEMENTATION(Sa31.f = A(2, 0);)
		ENABLE_SCALAR_IMPLEMENTATION(Sa12.f = A(0, 1);)
		ENABLE_SCALAR_IMPLEMENTATION(Sa22.f = A(1, 1);)
		ENABLE_SCALAR_IMPLEMENTATION(Sa32.f = A(2, 1);)
		ENABLE_SCALAR_IMPLEMENTATION(Sa13.f = A(0, 2);)
		ENABLE_SCALAR_IMPLEMENTATION(Sa23.f = A(1, 2);)
		ENABLE_SCALAR_IMPLEMENTATION(Sa33.f = A(2, 2);)

#include <vcl/math/mcadams/Singular_Value_Decomposition_Main_Kernel_Body.hpp>

		ENABLE_SCALAR_IMPLEMENTATION(U(0, 0) = Su11.f;)
		ENABLE_SCALAR_IMPLEMENTATION(U(1, 0) = Su21.f;)
		ENABLE_SCALAR_IMPLEMENTATION(U(2, 0) = Su31.f;)
		ENABLE_SCALAR_IMPLEMENTATION(U(0, 1) = Su12.f;)
		ENABLE_SCALAR_IMPLEMENTATION(U(1, 1) = Su22.f;)
		ENABLE_SCALAR_IMPLEMENTATION(U(2, 1) = Su32.f;)
		ENABLE_SCALAR_IMPLEMENTATION(U(0, 2) = Su13.f;)
		ENABLE_SCALAR_IMPLEMENTATION(U(1, 2) = Su23.f;)
		ENABLE_SCALAR_IMPLEMENTATION(U(2, 2) = Su33.f;)

		ENABLE_SCALAR_IMPLEMENTATION(V(0, 0) = Sv11.f;)
		ENABLE_SCALAR_IMPLEMENTATION(V(1, 0) = Sv21.f;)
		ENABLE_SCALAR_IMPLEMENTATION(V(2, 0) = Sv31.f;)
		ENABLE_SCALAR_IMPLEMENTATION(V(0, 1) = Sv12.f;)
		ENABLE_SCALAR_IMPLEMENTATION(V(1, 1) = Sv22.f;)
		ENABLE_SCALAR_IMPLEMENTATION(V(2, 1) = Sv32.f;)
		ENABLE_SCALAR_IMPLEMENTATION(V(0, 2) = Sv13.f;)
		ENABLE_SCALAR_IMPLEMENTATION(V(1, 2) = Sv23.f;)
		ENABLE_SCALAR_IMPLEMENTATION(V(2, 2) = Sv33.f;)

		ENABLE_SCALAR_IMPLEMENTATION(A(0, 0) = Sa11.f;)
		ENABLE_SCALAR_IMPLEMENTATION(A(1, 1) = Sa22.f;)
		ENABLE_SCALAR_IMPLEMENTATION(A(2, 2) = Sa33.f;)

		return JACOBI_CONJUGATION_SWEEPS * 3 + 3;
	}
#ifdef VCL_COMPILER_MSVC
#	pragma runtime_checks( "u", restore )
#endif
}}
#endif // defined(VCL_VECTORIZE_SSE)
