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
#define VCL_UTIL_WAVELETNOISE_INST
#include <vcl/util/waveletnoise.h>

 // C++ standard library
#include <cmath>
#include <random>

// GSL
#include <gsl/gsl>

namespace Vcl { namespace Util
{
	template<int N> WaveletNoise<N>::WaveletNoise()
	{
		static_assert(N >= 0, "N >= 0");
		static_assert(N % 2 == 0, "N is even");

		// ISO C++ randorm number generator
		std::random_device rd;
		std::mt19937 twister;
		twister.seed(rd());
		std::normal_distribution<float> normal;

		int n3 = N*N*N;

		_noiseTileData.reserve(n3);
		std::vector<float> temp1(n3, 0);
		std::vector<float> temp2(n3, 0);

		// Step 1. Fill the tile with random numbers in the range -1 to 1.
		for (int i = 0; i < n3; i++)
		{
			_noiseTileData.push_back(normal(twister));
		}

		// Steps 2 and 3. Downsample and upsample the tile
		for (int iy = 0; iy < N; iy++)
		{
			for (int iz = 0; iz < N; iz++)
			{
				const int i = iy * N + iz*N*N;
				downsample(&_noiseTileData[i], &temp1[i], N, 1);
				upsample(&temp1[i], &temp2[i], N, 1);
			}
		}
		for (int ix = 0; ix < N; ix++)
		{
			for (int iz = 0; iz < N; iz++)
			{
				const int i = ix + iz*N*N;
				downsample(&temp2[i], &temp1[i], N, N);
				upsample(&temp1[i], &temp2[i], N, N);
			}
		}
		for (int ix = 0; ix < N; ix++)
		{
			for (int iy = 0; iy < N; iy++)
			{
				const int i = ix + iy*N;
				downsample(&temp2[i], &temp1[i], N, N*N);
				upsample(&temp1[i], &temp2[i], N, N*N);
			}
		}

		// Step 4. Subtract out the coarse-scale contribution
		for (int i = 0; i < n3; i++)
		{
			_noiseTileData[i] -= temp2[i];
		}

		// Avoid even/odd variance difference by adding odd-offset version of noise to itself.
		int offset = N / 2;
		if (offset % 2 == 0) offset++;

		int icnt = 0;
		for (int ix = 0; ix < N; ix++)
		{
			for (int iy = 0; iy < N; iy++)
			{
				for (int iz = 0; iz < N; iz++)
				{
					temp1[icnt] = _noiseTileData[fast_modulo<N>(ix + offset) + fast_modulo<N>(iy + offset)*N + fast_modulo<N>(iz + offset)*N*N];
					icnt++;
				}
			}
		}

		_min =  std::numeric_limits<float>::max();
		_max = -std::numeric_limits<float>::max();

		for (int i = 0; i < n3; i++)
		{
			_noiseTileData[i] += temp1[i];

			_min = std::min(_noiseTileData[i], _min);
			_max = std::max(_noiseTileData[i], _max);
		}
	}

	template<int N> WaveletNoise<N>::~WaveletNoise()
	{
	}

	template<int N> float WaveletNoise<N>::evaluate(const float p[3]) const
	{
		int f[3], c[3], mid[3];
		float w[3][3], t, result = 0;

		/* Evaluate quadratic B-spline basis functions */
		for (int i = 0; i < 3; i++)
		{
			float tmp = ceil(p[i] - 0.5f);
			mid[i] = (int)tmp;
			t = tmp - (p[i] - 0.5f);

			w[i][0] = t * t / 2.0f;
			w[i][2] = (1 - t) * (1 - t) / 2.0f;
			w[i][1] = 1.0f - w[i][0] - w[i][2];
		}

		/* Loop over the noise coefficients within the bound */
		for (f[2] = -1; f[2] <= 1; f[2]++)
		{
			for (f[1] = -1; f[1] <= 1; f[1]++)
			{
				for (f[0] = -1; f[0] <= 1; f[0]++)
				{
					float weight = 1.0f;
					for (int i = 0; i < 3; i++)
					{
						c[i] = fast_modulo<N>(mid[i] + f[i]);
						weight *= w[i][f[i] + 1];
					}

					result += weight * _noiseTileData[c[2] * N*N + c[1] * N + c[0]];
				}
			}
		}

		return result;
	}

	template<int N> float WaveletNoise<N>::evaluate(const float p[3], float normal[3]) const
	{
		int c[3], minimum[3], maximum[3];
		float support, result = 0.0f;

		/* Bound the support of the basis functions for this projection direction */
		for (int i = 0; i < 3; i++)
		{
			support = 3.0f * fabs(normal[i]) + 3.0f * sqrtf((1.0f - normal[i] * normal[i]) / 2.0f);
			minimum[i] = (int)ceil(p[i] - support);
			maximum[i] = (int)floor(p[i] + support);
		}

		/* Loop over the noise coefficients within the bound */
		for (c[2] = minimum[2]; c[2] <= maximum[2]; c[2]++)
		{
			for (c[1] = minimum[1]; c[1] <= maximum[1]; c[1]++)
			{
				for (c[0] = minimum[0]; c[0] <= maximum[0]; c[0]++)
				{
					float t, t1, t2, t3, dot = 0.0f, weight = 1.0f;

					/* Dot the normal with the vector from c to p */
					for (int i = 0; i < 3; i++) { dot += normal[i] * (p[i] - (float)c[i]); }

					/* Evaluate the basis function at c moved halfway to p along the normal */
					for (int i = 0; i < 3; i++)
					{
						t = ((float)c[i] + normal[i] * dot / 2.0f) - (p[i] - 1.5f); t1 = t - 1.0f; t2 = 2.0f - t; t3 = 3.0f - t;
						weight *= (t <= 0 || t >= 3) ? 0 : (t<1.0f) ? t*t / 2.0f : (t<2.0f) ? 1.0f - (t1*t1 + t2*t2) / 2.0f : t3*t3 / 2.0f;
					}

					/* Evaluate noise by weighting noise coefficients by basis function values */
					result += weight * _noiseTileData[fast_modulo<N>(c[2])*N*N + fast_modulo<N>(c[1])*N + fast_modulo<N>(c[0])];
				}
			}
		}

		return result;
	}

	template<int N> float WaveletNoise<N>::evaluate(const float p[3], float s, float normal[3], int first_band, int nr_bands, float *w) const
	{
		float q[3], result = 0, variance = 0;

		for (int b = 0; b < nr_bands && s + (float)first_band + (float)b < 0; b++)
		{
			for (int i = 0; i <= 2; i++)
			{
				q[i] = 2.0f * p[i] * pow(2.0f, first_band + b);
			}
			result += (normal) ? w[b] * evaluate(q, normal) : w[b] * evaluate(q);
		}

		for (int b = 0; b < nr_bands; b++)
		{
			variance += w[b] * w[b];
		}

		// Adjust the noise so it has a variance of 1
		if (variance)
		{
			result /= sqrtf(variance * ((normal) ? 0.296f : 0.210f));
		}

		return result;
	}

#define ADD_WEIGHTED(x,y,z)\
	weight = 1.0f;\
	xC = fast_modulo<N>(midX + (x));\
	weight *= w[0][(x) + 1];\
	yC = fast_modulo<N>(midY + (y));\
	weight *= w[1][(y) + 1];\
	zC = fast_modulo<N>(midZ + (z));\
	weight *= w[2][(z) + 1];\
	result += weight * _noiseTileData[(zC * N + yC) * N + xC];

	template<int N> float WaveletNoise<N>::dx(const float p[3]) const
	{
		float w[3][3], t, result = 0;

		// Evaluate quadratic B-spline basis functions
		int midX = (int)ceil(p[0] - 0.5f);
		t = midX - (p[0] - 0.5f);
		w[0][0] = -t;
		w[0][2] = (1.f - t);
		w[0][1] = 2.0f * t - 1.0f;

		int midY = (int)ceil(p[1] - 0.5f);
		t = midY - (p[1] - 0.5f);
		w[1][0] = t * t * 0.5f;
		w[1][2] = (1.f - t) * (1.f - t) *0.5f;
		w[1][1] = 1.f - w[1][0] - w[1][2];

		int midZ = (int)ceil(p[2] - 0.5f);
		t = midZ - (p[2] - 0.5f);
		w[2][0] = t * t * 0.5f;
		w[2][2] = (1.f - t) * (1.f - t) *0.5f;
		w[2][1] = 1.f - w[2][0] - w[2][2];

		// Evaluate noise by weighting noise coefficients by basis function values
		int xC, yC, zC;
		float weight = 1;

		ADD_WEIGHTED(-1, -1, -1); ADD_WEIGHTED(0, -1, -1); ADD_WEIGHTED(1, -1, -1);
		ADD_WEIGHTED(-1, 0, -1); ADD_WEIGHTED(0, 0, -1); ADD_WEIGHTED(1, 0, -1);
		ADD_WEIGHTED(-1, 1, -1); ADD_WEIGHTED(0, 1, -1); ADD_WEIGHTED(1, 1, -1);

		ADD_WEIGHTED(-1, -1, 0);  ADD_WEIGHTED(0, -1, 0);  ADD_WEIGHTED(1, -1, 0);
		ADD_WEIGHTED(-1, 0, 0);  ADD_WEIGHTED(0, 0, 0);  ADD_WEIGHTED(1, 0, 0);
		ADD_WEIGHTED(-1, 1, 0);  ADD_WEIGHTED(0, 1, 0);  ADD_WEIGHTED(1, 1, 0);

		ADD_WEIGHTED(-1, -1, 1);  ADD_WEIGHTED(0, -1, 1);  ADD_WEIGHTED(1, -1, 1);
		ADD_WEIGHTED(-1, 0, 1);  ADD_WEIGHTED(0, 0, 1);  ADD_WEIGHTED(1, 0, 1);
		ADD_WEIGHTED(-1, 1, 1);  ADD_WEIGHTED(0, 1, 1);  ADD_WEIGHTED(1, 1, 1);

		return result;
	}

	template<int N> float WaveletNoise<N>::dy(const float p[3]) const
	{
		float w[3][3], t, result = 0;

		// Evaluate quadratic B-spline basis functions
		int midX = (int)ceil(p[0] - 0.5f);
		t = midX - (p[0] - 0.5f);
		w[0][0] = t * t * 0.5f;
		w[0][2] = (1.f - t) * (1.f - t) *0.5f;
		w[0][1] = 1.f - w[0][0] - w[0][2];

		int midY = (int)ceil(p[1] - 0.5f);
		t = midY - (p[1] - 0.5f);
		w[1][0] = -t;
		w[1][2] = (1.0f - t);
		w[1][1] = 2.0f * t - 1.0f;

		int midZ = (int)ceil(p[2] - 0.5f);
		t = midZ - (p[2] - 0.5f);
		w[2][0] = t * t * 0.5f;
		w[2][2] = (1.f - t) * (1.f - t) *0.5f;
		w[2][1] = 1.f - w[2][0] - w[2][2];

		// Evaluate noise by weighting noise coefficients by basis function values
		int xC, yC, zC;
		float weight = 1;

		ADD_WEIGHTED(-1, -1, -1); ADD_WEIGHTED(0, -1, -1); ADD_WEIGHTED(1, -1, -1);
		ADD_WEIGHTED(-1,  0, -1); ADD_WEIGHTED(0,  0, -1); ADD_WEIGHTED(1,  0, -1);
		ADD_WEIGHTED(-1,  1, -1); ADD_WEIGHTED(0,  1, -1); ADD_WEIGHTED(1,  1, -1);

		ADD_WEIGHTED(-1, -1, 0);  ADD_WEIGHTED(0, -1, 0);  ADD_WEIGHTED(1, -1, 0);
		ADD_WEIGHTED(-1,  0, 0);  ADD_WEIGHTED(0,  0, 0);  ADD_WEIGHTED(1,  0, 0);
		ADD_WEIGHTED(-1,  1, 0);  ADD_WEIGHTED(0,  1, 0);  ADD_WEIGHTED(1,  1, 0);

		ADD_WEIGHTED(-1, -1, 1);  ADD_WEIGHTED(0, -1, 1);  ADD_WEIGHTED(1, -1, 1);
		ADD_WEIGHTED(-1,  0, 1);  ADD_WEIGHTED(0,  0, 1);  ADD_WEIGHTED(1,  0, 1);
		ADD_WEIGHTED(-1,  1, 1);  ADD_WEIGHTED(0,  1, 1);  ADD_WEIGHTED(1,  1, 1);

		return result;
	}

	template<int N> float WaveletNoise<N>::dz(const float p[3]) const
	{
		float w[3][3], t, result = 0;

		// Evaluate quadratic B-spline basis functions
		int midX = (int)ceil(p[0] - 0.5f);
		t = midX - (p[0] - 0.5f);
		w[0][0] = t * t * 0.5f;
		w[0][2] = (1.f - t) * (1.f - t) *0.5f;
		w[0][1] = 1.f - w[0][0] - w[0][2];

		int midY = (int)ceil(p[1] - 0.5f);
		t = midY - (p[1] - 0.5f);
		w[1][0] = t * t * 0.5f;
		w[1][2] = (1.f - t) * (1.f - t) *0.5f;
		w[1][1] = 1.f - w[1][0] - w[1][2];

		int midZ = (int)ceil(p[2] - 0.5f);
		t = midZ - (p[2] - 0.5f);
		w[2][0] = -t;
		w[2][2] = (1.0f - t);
		w[2][1] = 2.0f * t - 1.0f;

		// Evaluate noise by weighting noise coefficients by basis function values
		int xC, yC, zC;
		float weight = 1;

		ADD_WEIGHTED(-1, -1, -1); ADD_WEIGHTED(0, -1, -1); ADD_WEIGHTED(1, -1, -1);
		ADD_WEIGHTED(-1, 0, -1); ADD_WEIGHTED(0, 0, -1); ADD_WEIGHTED(1, 0, -1);
		ADD_WEIGHTED(-1, 1, -1); ADD_WEIGHTED(0, 1, -1); ADD_WEIGHTED(1, 1, -1);

		ADD_WEIGHTED(-1, -1, 0);  ADD_WEIGHTED(0, -1, 0);  ADD_WEIGHTED(1, -1, 0);
		ADD_WEIGHTED(-1, 0, 0);  ADD_WEIGHTED(0, 0, 0);  ADD_WEIGHTED(1, 0, 0);
		ADD_WEIGHTED(-1, 1, 0);  ADD_WEIGHTED(0, 1, 0);  ADD_WEIGHTED(1, 1, 0);

		ADD_WEIGHTED(-1, -1, 1);  ADD_WEIGHTED(0, -1, 1);  ADD_WEIGHTED(1, -1, 1);
		ADD_WEIGHTED(-1, 0, 1);  ADD_WEIGHTED(0, 0, 1);  ADD_WEIGHTED(1, 0, 1);
		ADD_WEIGHTED(-1, 1, 1);  ADD_WEIGHTED(0, 1, 1);  ADD_WEIGHTED(1, 1, 1);

		return result;
	}

#undef ADD_WEIGHTED

	template<int N> void WaveletNoise<N>::dxDyDz(const float p[3], float final[3][3]) const
	{
		gsl::span<const float> data{ _noiseTileData.data(), static_cast<ptrdiff_t>(_noiseTileData.size()) };

		float w[3][3];
		float dw[3][3];
		float result1 = 0;
		float result2 = 0;
		float result3 = 0;
		float weight;

#define ADD_WEXT_INDEX(x,y,z) ( ((midZ+(z)) <<16) + ((midY+(y)) <<8)+ (midX+(x)) )

#define ADD_WEIGHTED_EXTDX(x,y,z)\
	  weight = dw[0][(x) + 1] * w[1][(y) + 1] * w[2][(z) + 1] ; \
	  result2 += data[ADD_WEXT_INDEX(x,y,z) +32] *weight;\
	  result3 += data[ADD_WEXT_INDEX(x,y,z) +64] *weight;

#define ADD_WEIGHTED_EXTDY(x,y,z)\
	  weight = w[0][(x) + 1] * dw[1][(y) + 1] * w[2][(z) + 1] ; \
	  result1 += data[ADD_WEXT_INDEX(x,y,z) +0 ] *weight;\
	  result3 += data[ADD_WEXT_INDEX(x,y,z) +64] *weight;

#define ADD_WEIGHTED_EXTDZ(x,y,z)\
	  weight = w[0][(x) + 1] * w[1][(y) + 1] * dw[2][(z) + 1] ; \
	  result1 += data[ADD_WEXT_INDEX(x,y,z) +0 ] *weight;\
	  result2 += data[ADD_WEXT_INDEX(x,y,z) +32] *weight;

		int midX = (int)ceil(p[0] - 0.5f);
		int midY = (int)ceil(p[1] - 0.5f);
		int midZ = (int)ceil(p[2] - 0.5f);

		float t0 = midX - (p[0] - 0.5f);
		float t1 = midY - (p[1] - 0.5f);
		float t2 = midZ - (p[2] - 0.5f);

		midX = fast_modulo<N>(midX) + 1;
		midY = fast_modulo<N>(midY) + 1;
		midZ = fast_modulo<N>(midZ) + 1;

		///////////////////////////////////////////////////////////////////////////////////////
		// evaluate splines
		///////////////////////////////////////////////////////////////////////////////////////
		dw[0][0] = -t0;
		dw[0][2] = (1.f - t0);
		dw[0][1] = 2.0f * t0 - 1.0f;

		dw[1][0] = -t1;
		dw[1][2] = (1.0f - t1);
		dw[1][1] = 2.0f * t1 - 1.0f;

		dw[2][0] = -t2;
		dw[2][2] = (1.0f - t2);
		dw[2][1] = 2.0f * t2 - 1.0f;

		w[0][0] = t0 * t0 * 0.5f;
		w[0][2] = (1.f - t0) * (1.f - t0) *0.5f;
		w[0][1] = 1.f - w[0][0] - w[0][2];

		w[1][0] = t1 * t1 * 0.5f;
		w[1][2] = (1.f - t1) * (1.f - t1) *0.5f;
		w[1][1] = 1.f - w[1][0] - w[1][2];

		w[2][0] = t2 * t2 * 0.5f;
		w[2][2] = (1.f - t2) * (1.f - t2) *0.5f;
		w[2][1] = 1.f - w[2][0] - w[2][2];

		///////////////////////////////////////////////////////////////////////////////////////
		// x derivative
		///////////////////////////////////////////////////////////////////////////////////////
		result2 = result3 = 0.0f;
		ADD_WEIGHTED_EXTDX(-1, -1, -1); ADD_WEIGHTED_EXTDX(0, -1, -1); ADD_WEIGHTED_EXTDX(1, -1, -1);
		ADD_WEIGHTED_EXTDX(-1, 0, -1); ADD_WEIGHTED_EXTDX(0, 0, -1); ADD_WEIGHTED_EXTDX(1, 0, -1);
		ADD_WEIGHTED_EXTDX(-1, 1, -1); ADD_WEIGHTED_EXTDX(0, 1, -1); ADD_WEIGHTED_EXTDX(1, 1, -1);

		ADD_WEIGHTED_EXTDX(-1, -1, 0);  ADD_WEIGHTED_EXTDX(0, -1, 0);  ADD_WEIGHTED_EXTDX(1, -1, 0);
		ADD_WEIGHTED_EXTDX(-1, 0, 0);  ADD_WEIGHTED_EXTDX(0, 0, 0);  ADD_WEIGHTED_EXTDX(1, 0, 0);
		ADD_WEIGHTED_EXTDX(-1, 1, 0);  ADD_WEIGHTED_EXTDX(0, 1, 0);  ADD_WEIGHTED_EXTDX(1, 1, 0);

		ADD_WEIGHTED_EXTDX(-1, -1, 1);  ADD_WEIGHTED_EXTDX(0, -1, 1);  ADD_WEIGHTED_EXTDX(1, -1, 1);
		ADD_WEIGHTED_EXTDX(-1, 0, 1);  ADD_WEIGHTED_EXTDX(0, 0, 1);  ADD_WEIGHTED_EXTDX(1, 0, 1);
		ADD_WEIGHTED_EXTDX(-1, 1, 1);  ADD_WEIGHTED_EXTDX(0, 1, 1);  ADD_WEIGHTED_EXTDX(1, 1, 1);
		final[1][0] = result2;
		final[2][0] = result3;

		///////////////////////////////////////////////////////////////////////////////////////
		// y derivative
		///////////////////////////////////////////////////////////////////////////////////////
		result1 = result3 = 0.0f;
		ADD_WEIGHTED_EXTDY(-1, -1, -1); ADD_WEIGHTED_EXTDY(0, -1, -1); ADD_WEIGHTED_EXTDY(1, -1, -1);
		ADD_WEIGHTED_EXTDY(-1, 0, -1); ADD_WEIGHTED_EXTDY(0, 0, -1); ADD_WEIGHTED_EXTDY(1, 0, -1);
		ADD_WEIGHTED_EXTDY(-1, 1, -1); ADD_WEIGHTED_EXTDY(0, 1, -1); ADD_WEIGHTED_EXTDY(1, 1, -1);

		ADD_WEIGHTED_EXTDY(-1, -1, 0);  ADD_WEIGHTED_EXTDY(0, -1, 0);  ADD_WEIGHTED_EXTDY(1, -1, 0);
		ADD_WEIGHTED_EXTDY(-1, 0, 0);  ADD_WEIGHTED_EXTDY(0, 0, 0);  ADD_WEIGHTED_EXTDY(1, 0, 0);
		ADD_WEIGHTED_EXTDY(-1, 1, 0);  ADD_WEIGHTED_EXTDY(0, 1, 0);  ADD_WEIGHTED_EXTDY(1, 1, 0);

		ADD_WEIGHTED_EXTDY(-1, -1, 1);  ADD_WEIGHTED_EXTDY(0, -1, 1);  ADD_WEIGHTED_EXTDY(1, -1, 1);
		ADD_WEIGHTED_EXTDY(-1, 0, 1);  ADD_WEIGHTED_EXTDY(0, 0, 1);  ADD_WEIGHTED_EXTDY(1, 0, 1);
		ADD_WEIGHTED_EXTDY(-1, 1, 1);  ADD_WEIGHTED_EXTDY(0, 1, 1);  ADD_WEIGHTED_EXTDY(1, 1, 1);
		final[0][1] = result1;
		final[2][1] = result3;

		///////////////////////////////////////////////////////////////////////////////////////
		// z derivative
		///////////////////////////////////////////////////////////////////////////////////////
		result1 = result2 = 0.0f;
		ADD_WEIGHTED_EXTDZ(-1, -1, -1); ADD_WEIGHTED_EXTDZ(0, -1, -1); ADD_WEIGHTED_EXTDZ(1, -1, -1);
		ADD_WEIGHTED_EXTDZ(-1, 0, -1); ADD_WEIGHTED_EXTDZ(0, 0, -1); ADD_WEIGHTED_EXTDZ(1, 0, -1);
		ADD_WEIGHTED_EXTDZ(-1, 1, -1); ADD_WEIGHTED_EXTDZ(0, 1, -1); ADD_WEIGHTED_EXTDZ(1, 1, -1);

		ADD_WEIGHTED_EXTDZ(-1, -1, 0);  ADD_WEIGHTED_EXTDZ(0, -1, 0);  ADD_WEIGHTED_EXTDZ(1, -1, 0);
		ADD_WEIGHTED_EXTDZ(-1, 0, 0);  ADD_WEIGHTED_EXTDZ(0, 0, 0);  ADD_WEIGHTED_EXTDZ(1, 0, 0);
		ADD_WEIGHTED_EXTDZ(-1, 1, 0);  ADD_WEIGHTED_EXTDZ(0, 1, 0);  ADD_WEIGHTED_EXTDZ(1, 1, 0);

		ADD_WEIGHTED_EXTDZ(-1, -1, 1);  ADD_WEIGHTED_EXTDZ(0, -1, 1);  ADD_WEIGHTED_EXTDZ(1, -1, 1);
		ADD_WEIGHTED_EXTDZ(-1, 0, 1);  ADD_WEIGHTED_EXTDZ(0, 0, 1);  ADD_WEIGHTED_EXTDZ(1, 0, 1);
		ADD_WEIGHTED_EXTDZ(-1, 1, 1);  ADD_WEIGHTED_EXTDZ(0, 1, 1);  ADD_WEIGHTED_EXTDZ(1, 1, 1);
		final[0][2] = result1;
		final[1][2] = result2;
#undef ADD_WEXT_INDEX
#undef ADD_WEIGHTED_EXTDX
#undef ADD_WEIGHTED_EXTDY
#undef ADD_WEIGHTED_EXTDZ
	}

	template<int N> void WaveletNoise<N>::velocity(const float p[3], float v[3]) const
	{
		const float p1[3] = { p[0] + N / 2,	p[1],		    p[2]         };
		const float p2[3] = { p[0],			p[1] + N / 2,	p[2]         };
		const float p3[3] = { p[0],			p[1],		    p[2] + N / 2 };

		const float f1y = dy(p1);
		const float f1z = dz(p1);

		const float f2x = dx(p2);
		const float f2z = dz(p2);

		const float f3x = dx(p3);
		const float f3y = dy(p3);

		v[0] = f3y - f2z;
		v[1] = f1z - f3x;
		v[2] = f2x - f1y;
	}

	template<int N> void WaveletNoise<N>::downsample(float* from, float* to, int n, int stride)
	{
		const float* const a = &ACoeffs[16];

		for (int i = 0; i < n / 2; i++)
		{
			to[i * stride] = 0;
			for (int k = 2 * i - 16; k <= 2 * i + 16; k++)
			{
				to[i * stride] += a[k - 2 * i] * from[fast_modulo<N>(k) * stride];
			}
		}
	}

	template<int N> void WaveletNoise<N>::upsample(float* from, float* to, int n, int stride)
	{
		const float* const p = &PCoeffs[2];

		for (int i = 0; i < n; i++)
		{
			to[i * stride] = 0;
			for (int k = i / 2; k <= i / 2 + 1; k++)
			{
				to[i * stride] += p[i - 2 * k] * from[fast_modulo<N>(k) * stride];
			}
		}
	}

	template<int N> const float WaveletNoise<N>::ACoeffs[] =
	{
		0.000334f,-0.001528f, 0.000410f, 0.003545f,-0.000938f,-0.008233f, 0.002172f, 0.019120f,
		-0.005040f,-0.044412f, 0.011655f, 0.103311f,-0.025936f,-0.243780f, 0.033979f, 0.655340f,
		0.655340f, 0.033979f,-0.243780f,-0.025936f, 0.103311f, 0.011655f,-0.044412f,-0.005040f,
		0.019120f, 0.002172f,-0.008233f,-0.000938f, 0.003546f, 0.000410f,-0.001528f, 0.000334f
	};
	template<int N> const float WaveletNoise<N>::PCoeffs[] = { 0.25f, 0.75f, 0.75f, 0.25f };

	template class WaveletNoise<32>;
	template class WaveletNoise<64>;
	template class WaveletNoise<128>;
}}
