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
#include <cmath>
#include <vector>

namespace Vcl { namespace Mathematics
{
	/*!
	 *	Wavlet stack implementation from "Kim, Thürey - Wavelet Turbulence for Fluid Simulation"
	 */
	class WaveletStack3D
	{
	public:
		WaveletStack3D(int x_res, int y_res, int z_res);
		~WaveletStack3D();

		void CreateStack(float* data, int levels);
		void CreateSingleLayer(float* data);

	public:
		const std::vector<float*>& Stack() const { return mStack; }

	private:
		void DoubleSize(float*& input, int xRes, int yRes, int zRes);
		void Decompose(
			const float* input,
			float*& lowFreq,
			float*& highFreq,
			float*& halfSize,
			int xRes,
			int yRes,
			int zRes);

	private:
		int mResX, mResY, mResZ;

		std::vector<float*> mStack;
		std::vector<float*> mCoeff;

		// Temporary buffers
		float* mBufferInput;
		float* mBufferLowFreq;
		float* mBufferHighFreq;
		float* mBufferHalfSize;
	};
}}
