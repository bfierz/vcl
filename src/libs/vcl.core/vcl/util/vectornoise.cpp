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
#define VCL_UTIL_VECTORNOISE_INST
#include <vcl/util/vectornoise.h>

namespace Vcl { namespace Util
{
	template<int N> VectorNoise<N>::VectorNoise()
	{
		_noise1 = std::make_unique<WaveletNoise<N>>();
		_noise2 = std::make_unique<WaveletNoise<N>>();
		_noise3 = std::make_unique<WaveletNoise<N>>();
	}

	template<int N> VectorNoise<N>::~VectorNoise()
	{
	}

	template<int N> Eigen::Vector3f VectorNoise<N>::evaluate(const float p[3]) const
	{
		const float f1y = _noise1->dy(p);
		const float f1z = _noise1->dz(p);

		const float f2x = _noise2->dx(p);
		const float f2z = _noise2->dz(p);

		const float f3x = _noise3->dx(p);
		const float f3y = _noise3->dy(p);

		Eigen::Vector3f v;
		v.x() = f3y - f2z;
		v.y() = f1z - f3x;
		v.z() = f2x - f1y;

		return v;
	}

	template class VectorNoise<32>;
	template class VectorNoise<64>;
	template class VectorNoise<128>;
}}
