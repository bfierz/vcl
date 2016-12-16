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
#include <vcl/math/waveletstack3d.h>

// C++ standard library
#include <cstring>

namespace Vcl { namespace Mathematics
{
	////////////////////////////////////////////////////////////////////////
	// basic up / downsampling
	////////////////////////////////////////////////////////////////////////
	static void Downsample(const float *from, float *to, int n, int stride)
	{
		static const float aCoeffs[32] = {
			0.000334f,-0.001528f, 0.000410f, 0.003545f,-0.000938f,-0.008233f, 0.002172f, 0.019120f,
			-0.005040f,-0.044412f, 0.011655f, 0.103311f,-0.025936f,-0.243780f, 0.033979f, 0.655340f,
			0.655340f, 0.033979f,-0.243780f,-0.025936f, 0.103311f, 0.011655f,-0.044412f,-0.005040f,
			0.019120f, 0.002172f,-0.008233f,-0.000938f, 0.003546f, 0.000410f,-0.001528f, 0.000334f
		};
		static const float *const aCoCenter= &aCoeffs[16];

		for (int i = 0; i < n / 2; i++) {
			to[i * stride] = 0;
			for (int k = 2 * i - 16; k < 2 * i + 16; k++) { 
				// handle boundary
				float fromval; 
				if(k<0) {
					fromval = from[0];
				} else if(k>n-1) {
					fromval = from[(n-1) * stride];
				} else {
					fromval = from[k * stride]; 
				} 
				
				to[i * stride] += aCoCenter[k - 2*i] * fromval; 
			}
		}
	}

	static void Upsample(const float *from, float *to, int n, int stride)
	{
		static const float pCoeffs[4] = { 0.25f, 0.75f, 0.75f, 0.25f };
		static const float *const pCoCenter = &pCoeffs[2];
	
		for (int i = 0; i < n; i++) {
			to[i * stride] = 0;
			for (int k = i / 2; k <= i / 2 + 1; k++) {
				float fromval;
				// handle boundary
				if(k>n/2) {
					fromval = from[(n/2) * stride];
				} else {
					fromval = from[k * stride]; 
				} 

				to[i * stride] += pCoCenter[i - 2 * k] * fromval; 
			}
		}
	}

	////////////////////////////////////////////////////////////////////////
	// directional up & downsampling
	////////////////////////////////////////////////////////////////////////

	// some convenience functions for an nxn image
	static void DownsampleX(float* to, const float* from, int sx,int sy, int sz) {
		
		for (int iy = 0; iy < sy; iy++) 
			for (int iz = 0; iz < sz; iz++) {
				const int i = iy * sx + iz*sx*sy;
				Downsample(&from[i], &to[i], sx, 1);
			}
	}
	static void DownsampleY(float* to, const float* from, int sx,int sy, int sz) {
		
		for (int ix = 0; ix < sx; ix++) 
			for (int iz = 0; iz < sz; iz++) {
				const int i = ix + iz*sx*sy;
				Downsample(&from[i], &to[i], sy, sx);
		}
	}
	static void DownsampleZ(float* to, const float* from, int sx,int sy, int sz) {
		
		for (int ix = 0; ix < sx; ix++) 
			for (int iy = 0; iy < sy; iy++) {
				const int i = ix + iy*sx;
				Downsample(&from[i], &to[i], sz, sx*sy);
		}
	}
	static void UpsampleX(float* to, const float* from, int sx,int sy, int sz) {
		
		for (int iy = 0; iy < sy; iy++) 
			for (int iz = 0; iz < sz; iz++) {
				const int i = iy * sx + iz*sx*sy;
				Upsample(&from[i], &to[i], sx, 1);
			}
	}
	static void UpsampleY(float* to, const float* from, int sx,int sy, int sz) {
		
		for (int ix = 0; ix < sx; ix++) 
			for (int iz = 0; iz < sz; iz++) {
				const int i = ix + iz*sx*sy;
				Upsample(&from[i], &to[i], sy, sx);
			}
	}
	static void UpsampleZ(float* to, const float* from, int sx,int sy, int sz) {
		
		for (int ix = 0; ix < sx; ix++) 
			for (int iy = 0; iy < sy; iy++) {
				const int i = ix + iy*sx;
				Upsample(&from[i], &to[i], sz, sx*sy);
			}
	}

	WaveletStack3D::WaveletStack3D(int x_res, int y_res, int z_res)
	: mResX(x_res), mResY(y_res), mResZ(z_res)
	{
		mBufferInput = new float[mResX*mResY*mResZ];
		mBufferLowFreq = new float[mResX*mResY*mResZ];
		mBufferHighFreq = new float[mResX*mResY*mResZ];
		mBufferHalfSize = new float[mResX*mResY*mResZ];
	}

	WaveletStack3D::~WaveletStack3D()
	{
		VCL_SAFE_DELETE_ARRAY(mBufferInput);
		VCL_SAFE_DELETE_ARRAY(mBufferLowFreq);
		VCL_SAFE_DELETE_ARRAY(mBufferHighFreq);
		VCL_SAFE_DELETE_ARRAY(mBufferHalfSize);

		for (size_t x = 0; x < mStack.size(); x++)
		{
			if (mStack[x] != NULL) delete mStack[x];
		}

		for (size_t x = 0; x < mCoeff.size(); x++)
		{
			if (mCoeff[x] != NULL) delete mCoeff[x];
		}

		mStack.clear();
		mCoeff.clear();
	}

	void WaveletStack3D::DoubleSize(float*& input, int xRes, int yRes, int zRes)
	{
		VCL_UNREFERENCED_PARAMETER(input);
		VCL_UNREFERENCED_PARAMETER(xRes);
		VCL_UNREFERENCED_PARAMETER(yRes);
		VCL_UNREFERENCED_PARAMETER(zRes);
	}

	void WaveletStack3D::Decompose
	(
		const float* input,
		float*& lowFreq,
		float*& highFreq,
		float*& halfSize,
		int xRes,
		int yRes,
		int zRes
	)
	{
		// init the temp arrays
		float* temp1 = lowFreq;
		float* temp2 = highFreq;

		memset(temp1, 0, xRes*yRes*zRes*sizeof(float));
		memset(temp2, 0, xRes*yRes*zRes*sizeof(float));
		
		DownsampleX(temp1, input, xRes, yRes, zRes);
		DownsampleY(temp2, temp1, xRes, yRes, zRes);
		DownsampleZ(temp1, temp2, xRes, yRes, zRes);

		// copy out the downsampled image
		int index = 0;
		for (int z = 0; z < zRes / 2; z++)
			for (int y = 0; y < yRes / 2; y++)
				for (int x = 0; x < xRes / 2; x++, index++)
					halfSize[index] = temp1[x + y * xRes + z * xRes * yRes];

		UpsampleZ(temp2, temp1, xRes, yRes, zRes);
		UpsampleY(temp1, temp2, xRes, yRes, zRes);
		UpsampleX(temp2, temp1, xRes, yRes, zRes);
		

		// final low frequency component is in temp2
		lowFreq = temp2;
		highFreq = temp1;

		/*const int BND = 0;
		index = 0;
		for (int z = 0; z < zRes; z++) 
		{
			for (int y = 0; y < yRes; y++)
			{
				for (int x = 0; x < xRes; x++, index++)
				{
					// brute force reset of boundaries
					if(z>=zRes-1-BND || x>=xRes-1-BND || y>=yRes-1-BND || z <= BND || y <= BND || x <= BND)
					{
						highFreq[index] = 0.; 
					} else {
						highFreq[index] = input[index] - lowFreq[index];
					}
				}
			}
		}*/
		index = 0;
		for (int z = 0; z < zRes; z++) 
		{
			for (int y = 0; y < yRes; y++)
			{
				for (int x = 0; x < xRes; x++, index++)
				{
					highFreq[index] = input[index] - lowFreq[index];
				}
			}
		}
	}

	void WaveletStack3D::CreateStack(float* data, int levels)
	{
		int max_levels = (int)(log((float)mResX) / log(2.0f));

		if (levels < 0 || levels > max_levels) levels = max_levels;

		if (levels > (int) mStack.size())
		{
			mStack.resize(levels, NULL);
			mCoeff.resize(levels, NULL);
		}

		memcpy(mBufferInput, data, mResX*mResY*mResZ*sizeof(float));

		int currentXRes = mResX;
		int currentYRes = mResY;
		int currentZRes = mResZ;

		// Create the image stack
		for (int x = 0; x < levels; x++)
		{
			if (mStack[x] == NULL) mStack[x] = new float[currentXRes * currentYRes * currentZRes];
			if (mCoeff[x] == NULL) mCoeff[x] = new float[currentXRes * currentYRes * currentZRes];

			Decompose(mBufferInput, mBufferLowFreq, mBufferHighFreq, mBufferHalfSize, currentXRes, currentYRes, currentZRes);

			// Save downsampled image to _coeffs and _stack
			memcpy(mCoeff[x], mBufferHighFreq, currentXRes * currentYRes * currentZRes * sizeof(float));
			memcpy(mStack[x], mBufferHighFreq, currentXRes * currentYRes * currentZRes * sizeof(float));

			// copy leftovers to input for next iteration
			memcpy(mBufferInput, mBufferHalfSize, currentXRes * currentYRes * currentZRes * sizeof(float));

			// upsample the image and save it to stack
			int doubleX = currentXRes;
			int doubleY = currentYRes;
			int doubleZ = currentZRes;
			for (int y = 0; y < x; y++)
			{
				DoubleSize(mStack[x], doubleX, doubleY, doubleZ);
				doubleX *= 2;
				doubleY *= 2;
				doubleZ *= 2;
			}

			currentXRes /= 2;
			currentYRes /= 2;
			currentZRes /= 2;
		}
	}

	void WaveletStack3D::CreateSingleLayer(float* data)
	{
		if (1 > mStack.size())
		{
			mStack.resize(1, NULL);
		}

		if (mStack[0] == NULL) mStack[0] = new float[mResX * mResY * mResZ];

		Decompose(data, mBufferLowFreq, mBufferHighFreq, mBufferHalfSize, mResX, mResY, mResZ);

		// Save downsampled image to the stack
		memcpy(mStack[0], mBufferHighFreq, mResX*mResY*mResZ*sizeof(float));
	}
}}
