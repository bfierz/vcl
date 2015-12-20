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
#include <vcl/config/eigen.h>

// C++ standard libary
#include <vector>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Util
{
	template<int N>
	inline int fast_modulo(int x)
	{
		int m = x % N; 
		return (m < 0) ? m + N : m;
	}
	
	template<> inline int fast_modulo<128>(int x) { return x & 127; }
	template<> inline int fast_modulo< 64>(int x) { return x &  63; }
	template<> inline int fast_modulo< 32>(int x) { return x &  32; }
	template<> inline int fast_modulo< 16>(int x) { return x &  15; }

	/*!
	 *	Wavelet noise implementation by Robert L. Cook and Tony DeRose
	 */
	template<int N>
	class WaveletNoise
	{
	public:
		WaveletNoise();
		~WaveletNoise();

	public: // Evaluation
		float evaluate(const float p[3]) const;
		float evaluate(const float p[3], float normal[3]) const;
		float evaluate(const float p[3], float s, float normal[3], int first_band, int nr_bands, float *w) const;

		float dx(const float p[3]) const;
		float dy(const float p[3]) const;
		float dz(const float p[3]) const;
		void dxDyDz(const float p[3], float final[3][3]) const;

		void velocity(const float p[3], float v[3]) const;

	public: // Properties
		float minValue() const { return _min; }
		float maxValue() const { return _max; }

	public: // Access
		const int getNoiseTileSize() const { return N; }
		const float* getNoiseTileData() const { return _noiseTileData.data(); }

	private: // Helper methods
		static void downsample(float* from, float* to, int n, int stride);
		static void upsample(float* from, float* to, int n, int stride);

	private: // Constants
		const static float ACoeffs[32];
		const static float PCoeffs[4];

	private: // Member fields
		std::vector<float> _noiseTileData;
		float _min, _max;
	};
}}

namespace Vcl { namespace Util
{
#ifndef VCL_UTIL_WAVELETNOISE_INST
	extern template class WaveletNoise<32>;
	extern template class WaveletNoise<64>;
	extern template class WaveletNoise<128>;
#endif // VCL_UTIL_WAVELETNOISE_INST
}}
