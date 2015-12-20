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
#include <memory>

// VCL
#include <vcl/util/waveletnoise.h>

namespace Vcl { namespace Util
{
	/*!
	 *	Vector noise based on the SIGGRAPH 2007 paper by Bridson
	 */
	template<int N>
	class VectorNoise
	{
	public:
		VectorNoise();
		~VectorNoise();

	public: // Evaluation
		Eigen::Vector3f evaluate(const float p[3]) const;

	public: // Access
		const int size() const { return N; }
		const void noiseData(const float** n1, const float** n2, const float** n3) const
		{
			*n1 = _noise1->getNoiseTileData();
			*n2 = _noise2->getNoiseTileData();
			*n3 = _noise3->getNoiseTileData();
		}

	private: // Member fields
		std::unique_ptr<WaveletNoise<N>> _noise1, _noise2, _noise3;
	};
}}

namespace Vcl { namespace Util
{
#ifndef VCL_UTIL_VECTORNOISE_INST
	extern template class VectorNoise<32>;
	extern template class VectorNoise<64>;
	extern template class VectorNoise<128>;
#endif // VCL_UTIL_VECTORNOISE_INST
}}
