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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <memory>

// FUSE
#include <vcl/util/waveletnoise_modulo.h>

namespace Vcl { namespace Util { namespace Details {
	inline std::unique_ptr<std::mt19937> make_twister(unsigned int seed)
	{
		auto twister = std::make_unique<std::mt19937>();
		twister->seed(seed);
		return twister;
	}

	// clang-format off
	VCL_CPP_CONSTEXPR_11 std::array<float, 32> ACoeffs =
	{
		 0.000334f,-0.001528f, 0.000410f, 0.003545f,-0.000938f,-0.008233f, 0.002172f, 0.019120f,
		-0.005040f,-0.044412f, 0.011655f, 0.103311f,-0.025936f,-0.243780f, 0.033979f, 0.655340f,
		 0.655340f, 0.033979f,-0.243780f,-0.025936f, 0.103311f, 0.011655f,-0.044412f,-0.005040f,
		 0.019120f, 0.002172f,-0.008233f,-0.000938f, 0.003546f, 0.000410f,-0.001528f, 0.000334f
	};
	// clang-format on
	VCL_CPP_CONSTEXPR_11 std::array<float, 4> PCoeffs = { 0.25f, 0.75f, 0.75f, 0.25f };

	//! Evaluate quadratic B-spline basis functions
	inline void evaluateQuadraticSplineBasisImpl(float t, std::array<float, 3>& w) noexcept
	{
		w[0] = t * t / 2.0f;
		w[2] = (1.0f - t) * (1.0f - t) / 2.0f;
		w[1] = 1.0f - w[0] - w[2];
	}

	//! Evaluate quadratic B-spline basis functions
	inline void evaluateQuadraticSplineBasis(float p, std::array<float, 3>& w, int& mid) noexcept
	{
		const float midf = ceil(p - 0.5f);
		const float t = midf - (p - 0.5f);
		mid = static_cast<int>(midf);

		evaluateQuadraticSplineBasisImpl(t, w);
	}

	//! Evaluate derivative of the quadratic B-spline basis functions
	inline void evaluateDQuadraticSplineBasisImpl(float t, std::array<float, 3>& w) noexcept
	{
		w[0] = -t;
		w[2] = (1.0f - t);
		w[1] = 2.0f * t - 1.0f;
	}

	//! Evaluate derivative of the quadratic B-spline basis functions
	inline void evaluateDQuadraticSplineBasis(float p, std::array<float, 3>& w, int& mid) noexcept
	{
		const float midf = ceil(p - 0.5f);
		const float t = midf - (p - 0.5f);
		mid = static_cast<int>(midf);

		evaluateDQuadraticSplineBasisImpl(t, w);
	}

	//! Downsample values according to the wavelet coefficients
	template<int N>
	void downsample(stdext::span<const float> from, stdext::span<float> to, int n, int stride) noexcept
	{
		const stdext::span<const float> a = ACoeffs;
		for (int i = 0; i < n / 2; i++)
		{
			to[i * stride] = 0;
			for (int k = 2 * i - 16; k < 2 * i + 16; k++)
			{
				to[i * stride] += a[16 + k - 2 * i] * from[Vcl::Util::FastMath<N>::modulo(k) * stride];
			}
		}
	}

	//! Upsample values according to the wavelet coefficients
	template<int N>
	void upsample(stdext::span<const float> from, stdext::span<float> to, int n, int stride) noexcept
	{
		const stdext::span<const float> p = PCoeffs;
		for (int i = 0; i < n; i++)
		{
			to[i * stride] = 0;
			for (int k = i / 2; k <= i / 2 + 1; k++)
			{
				to[i * stride] += p[2 + i - 2 * k] * from[Vcl::Util::FastMath<N / 2>::modulo(k) * stride];
			}
		}
	}
}}}
