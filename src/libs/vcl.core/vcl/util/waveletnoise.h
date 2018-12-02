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
#include <array>
#include <random>
#include <vector>

// GSL
#include <gsl/gsl>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Util
{
	/*!
	 *	Wavelet noise implementation by Robert L. Cook and Tony DeRose
	 */
	template<int N>
	class WaveletNoise
	{
	public: // Type definitions
		using Vec3 = std::array<float, 3>;
		using Mat33 = std::array<std::array<float, 3>, 3>;

	public:
		WaveletNoise();
		WaveletNoise(unsigned int seed);
		WaveletNoise(std::mt19937& rnd_gen);

	public: // Evaluation
		float evaluate(const Vec3& p) const;
		float evaluate(const Vec3& p, const Vec3& normal) const;
		float evaluate(const Vec3& p, float s, const Vec3* normal, int first_band, int nr_bands, gsl::span<const float> w) const;

		float dx(const Vec3& p) const;
		float dy(const Vec3& p) const;
		float dz(const Vec3& p) const;
		void dxDyDz(const Vec3& p, Mat33& final) const;

		Vec3 velocity(const Vec3& p) const;

	public: // Properties
		float minValue() const noexcept { return _min; }
		float maxValue() const noexcept { return _max; }

	public: // Access
		int getNoiseTileSize() const noexcept { return N; }
		const float* getNoiseTileData() const noexcept { return _noiseTileData.data(); }

	protected: // Helper methods

		//! Special constructor taking an initialized set of random numbers
		WaveletNoise(gsl::span<float> noise_data_base);

		//! Initialize the noise data
		void initializeNoise(gsl::span<float> noise_data_base);
	private:
		//! Noise data
		std::vector<float> _noiseTileData;

		float _min; //!< Minimum noise data
		float _max;	//!< Maximum noise data
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
