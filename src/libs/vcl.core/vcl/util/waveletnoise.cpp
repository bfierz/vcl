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
#include <vcl/util/waveletnoise_modulo.h>

 // C++ standard library
#include <cmath>

// Disable the core-guideline checker until this file is refactored
#if defined(VCL_COMPILER_MSVC) && defined(VCL_CHECK_CORE_GUIDELINES) && (_MSC_VER >= 1910)
#	pragma warning(push, 1)
//#	pragma warning(disable: ALL_CPPCORECHECK_WARNINGS)
#	pragma warning(disable: 26446)
#	pragma warning(disable: 26451)
#	pragma warning(disable: 26482)
#endif

namespace
{
	std::unique_ptr<std::mt19937> make_twister(unsigned int seed)
	{
		auto twister = std::make_unique<std::mt19937>();
		twister->seed(seed);
		return twister;
	}
}

namespace Vcl { namespace Util
{
	template<int N>
	WaveletNoise<N>::WaveletNoise()
		: WaveletNoise(std::random_device{}())
	{
	}

	template<int N>
	WaveletNoise<N>::WaveletNoise(unsigned int seed)
		: WaveletNoise(*make_twister(seed))
	{
	}

	template<int N>
	WaveletNoise<N>::WaveletNoise(std::mt19937& rnd_gen)
	{
		static_assert(N >= 0, "N >= 0");
		static_assert(N % 2 == 0, "N is even");

		std::normal_distribution<float> normal;

		const int n3 = N*N*N;

		_noiseTileData.reserve(n3);
		std::vector<float> temp1(n3, 0);
		std::vector<float> temp2(n3, 0);

		// Step 1. Fill the tile with random numbers in the range -1 to 1.
		std::generate_n(std::back_inserter(_noiseTileData), n3, [&normal, &rnd_gen]()
		{
			return normal(rnd_gen);
		});

		// Steps 2 and 3. Downsample and upsample the tile
		for (int iy = 0; iy < N; iy++)
		{
			for (int iz = 0; iz < N; iz++)
			{
				const int i = iy * N + iz*N*N;
				downsample(gsl::make_span(&_noiseTileData[i], N), gsl::make_span(&temp1[i], N), N, 1);
				upsample(  gsl::make_span(&temp1[i], N), gsl::make_span(&temp2[i], N), N, 1);
			}
		}
		for (int ix = 0; ix < N; ix++)
		{
			for (int iz = 0; iz < N; iz++)
			{
				const int i = ix + iz*N*N;
				downsample(gsl::make_span(&temp2[i], N + N*N), gsl::make_span(&temp1[i], N + N*N), N, N);
				upsample(  gsl::make_span(&temp1[i], N + N*N), gsl::make_span(&temp2[i], N + N*N), N, N);
			}
		}
		for (int ix = 0; ix < N; ix++)
		{
			for (int iy = 0; iy < N; iy++)
			{
				const int i = ix + iy*N;
				downsample(gsl::make_span(&temp2[i], N*N*N), gsl::make_span(&temp1[i], N*N*N), N, N*N);
				upsample(  gsl::make_span(&temp1[i], N*N*N), gsl::make_span(&temp2[i], N*N*N), N, N*N);
			}
		}

		// Step 4. Subtract out the coarse-scale contribution
		for (int i = 0; i < n3; i++)
		{
			_noiseTileData[i] -= temp2[i];
		}

		// Avoid even/odd variance difference by adding odd-offset version of noise to itself.
		int offset = N / 2;
		if (offset % 2 == 0) { offset++; }

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

	template<int N>
	float WaveletNoise<N>::evaluate(const Vec3& p) const
	{
		float result = 0;

		// Evaluate quadratic B-spline basis functions
		Mat33 w;
		int mid[3];
		evaluateQuadraticSplineBasis(p[0], w[0], mid[0]);
		evaluateQuadraticSplineBasis(p[1], w[1], mid[1]);
		evaluateQuadraticSplineBasis(p[2], w[2], mid[2]);
		
		// Loop over the noise coefficients within the bound
		for (ptrdiff_t f2 = -1; f2 <= 1; f2++)
		{
			for (ptrdiff_t f1 = -1; f1 <= 1; f1++)
			{
				for (ptrdiff_t f0 = -1; f0 <= 1; f0++)
				{
					float weight = 1.0f;
					const ptrdiff_t c0 = fast_modulo<N>(mid[0] + f0);
					weight *= w[0][f0 + 1];
					const ptrdiff_t c1 = fast_modulo<N>(mid[1] + f1);
					weight *= w[1][f1 + 1];
					const ptrdiff_t c2 = fast_modulo<N>(mid[2] + f2);
					weight *= w[2][f2 + 1];

					result += weight * _noiseTileData[c2 * N*N + c1 * N + c0];
				}
			}
		}

		return result;
	}

	template<int N>
	float WaveletNoise<N>::evaluate(const Vec3& p, const Vec3& normal) const
	{
		int c[3], minimum[3], maximum[3];
		float result = 0.0f;

		// Bound the support of the basis functions for this projection direction
		for (int i = 0; i < 3; i++)
		{
			const float support = 3.0f * fabs(normal[i]) + 3.0f * sqrtf((1.0f - normal[i] * normal[i]) / 2.0f);
			minimum[i] = static_cast<int>(ceil(p[i] - support));
			maximum[i] = static_cast<int>(floor(p[i] + support));
		}

		// Loop over the noise coefficients within the bound
		for (c[2] = minimum[2]; c[2] <= maximum[2]; c[2]++)
		{
			for (c[1] = minimum[1]; c[1] <= maximum[1]; c[1]++)
			{
				for (c[0] = minimum[0]; c[0] <= maximum[0]; c[0]++)
				{
					float t, t1, t2, t3, dot = 0.0f, weight = 1.0f;

					// Dot the normal with the vector from c to p
					for (int i = 0; i < 3; i++) { dot += normal[i] * (p[i] - static_cast<float>(c[i])); }

					// Evaluate the basis function at c moved halfway to p along the normal
					for (int i = 0; i < 3; i++)
					{
						t = (static_cast<float>(c[i]) + normal[i] * dot / 2.0f) - (p[i] - 1.5f); t1 = t - 1.0f; t2 = 2.0f - t; t3 = 3.0f - t;
						weight *= (t <= 0 || t >= 3) ? 0 : (t<1.0f) ? t*t / 2.0f : (t<2.0f) ? 1.0f - (t1*t1 + t2*t2) / 2.0f : t3*t3 / 2.0f;
					}

					// Evaluate noise by weighting noise coefficients by basis function values
					result += weight * _noiseTileData[fast_modulo<N>(c[2])*N*N + fast_modulo<N>(c[1])*N + fast_modulo<N>(c[0])];
				}
			}
		}

		return result;
	}

	template<int N>
	float WaveletNoise<N>::evaluate(const Vec3& p, float s, const Vec3* normal, int first_band, int nr_bands, gsl::span<const float> w) const
	{
		float result = 0;
		for (int b = 0; b < nr_bands && s + static_cast<float>(first_band + b) < 0; b++)
		{
			Vec3 q;
			for (int i = 0; i <= 2; i++)
			{
				q[i] = 2.0f * p[i] * pow(2.0f, first_band + b);
			}
			result += (normal != nullptr) ? w[b] * evaluate(q, *normal) : w[b] * evaluate(q);
		}

		float variance = 0;
		for (int b = 0; b < nr_bands; b++)
		{
			variance += w[b] * w[b];
		}

		// Adjust the noise so it has a variance of 1
		if (variance > 0)
		{
			result /= sqrtf(variance * ((normal != nullptr) ? 0.296f : 0.210f));
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

	template<int N>
	float WaveletNoise<N>::dx(const Vec3& p) const
	{
		// Evaluate quadratic B-spline basis functions
		Mat33 w;
		int mid[3];
		evaluateDQuadraticSplineBasis(p[0], w[0], mid[0]);
		evaluateQuadraticSplineBasis (p[1], w[1], mid[1]);
		evaluateQuadraticSplineBasis (p[2], w[2], mid[2]);
		
		// Evaluate noise by weighting noise coefficients by basis function values
		int xC = 0, yC = 0, zC = 0;
		float weight = 1;
		float result = 0;
		const int midX = mid[0];
		const int midY = mid[1];
		const int midZ = mid[2];

		// clang-format off
		ADD_WEIGHTED(-1,-1,-1); ADD_WEIGHTED(0,-1,-1); ADD_WEIGHTED(1,-1,-1);
		ADD_WEIGHTED(-1, 0,-1); ADD_WEIGHTED(0, 0,-1); ADD_WEIGHTED(1, 0,-1);
		ADD_WEIGHTED(-1, 1,-1); ADD_WEIGHTED(0, 1,-1); ADD_WEIGHTED(1, 1,-1);

		ADD_WEIGHTED(-1,-1, 0); ADD_WEIGHTED(0,-1, 0); ADD_WEIGHTED(1,-1, 0);
		ADD_WEIGHTED(-1, 0, 0); ADD_WEIGHTED(0, 0, 0); ADD_WEIGHTED(1, 0, 0);
		ADD_WEIGHTED(-1, 1, 0); ADD_WEIGHTED(0, 1, 0); ADD_WEIGHTED(1, 1, 0);

		ADD_WEIGHTED(-1,-1, 1); ADD_WEIGHTED(0,-1, 1); ADD_WEIGHTED(1,-1, 1);
		ADD_WEIGHTED(-1, 0, 1); ADD_WEIGHTED(0, 0, 1); ADD_WEIGHTED(1, 0, 1);
		ADD_WEIGHTED(-1, 1, 1); ADD_WEIGHTED(0, 1, 1); ADD_WEIGHTED(1, 1, 1);
		// clang-format on

		return result;
	}

	template<int N>
	float WaveletNoise<N>::dy(const Vec3& p) const
	{
		// Evaluate quadratic B-spline basis functions
		Mat33 w;
		int mid[3];
		evaluateQuadraticSplineBasis (p[0], w[0], mid[0]);
		evaluateDQuadraticSplineBasis(p[1], w[1], mid[1]);
		evaluateQuadraticSplineBasis (p[2], w[2], mid[2]);

		// Evaluate noise by weighting noise coefficients by basis function values
		int xC = 0, yC = 0, zC = 0;
		float weight = 1;
		float result = 0;
		const int midX = mid[0];
		const int midY = mid[1];
		const int midZ = mid[2];

		// clang-format off
		ADD_WEIGHTED(-1,-1,-1); ADD_WEIGHTED(0,-1,-1); ADD_WEIGHTED(1,-1,-1);
		ADD_WEIGHTED(-1, 0,-1); ADD_WEIGHTED(0, 0,-1); ADD_WEIGHTED(1, 0,-1);
		ADD_WEIGHTED(-1, 1,-1); ADD_WEIGHTED(0, 1,-1); ADD_WEIGHTED(1, 1,-1);

		ADD_WEIGHTED(-1,-1, 0); ADD_WEIGHTED(0,-1, 0); ADD_WEIGHTED(1,-1, 0);
		ADD_WEIGHTED(-1, 0, 0); ADD_WEIGHTED(0, 0, 0); ADD_WEIGHTED(1, 0, 0);
		ADD_WEIGHTED(-1, 1, 0); ADD_WEIGHTED(0, 1, 0); ADD_WEIGHTED(1, 1, 0);

		ADD_WEIGHTED(-1,-1, 1); ADD_WEIGHTED(0,-1, 1); ADD_WEIGHTED(1,-1, 1);
		ADD_WEIGHTED(-1, 0, 1); ADD_WEIGHTED(0, 0, 1); ADD_WEIGHTED(1, 0, 1);
		ADD_WEIGHTED(-1, 1, 1); ADD_WEIGHTED(0, 1, 1); ADD_WEIGHTED(1, 1, 1);
		// clang-format on

		return result;
	}

	template<int N>
	float WaveletNoise<N>::dz(const Vec3& p) const
	{
		// Evaluate quadratic B-spline basis functions
		Mat33 w;
		int mid[3];
		evaluateQuadraticSplineBasis (p[0], w[0], mid[0]);
		evaluateQuadraticSplineBasis (p[1], w[1], mid[1]);
		evaluateDQuadraticSplineBasis(p[2], w[2], mid[2]);

		// Evaluate noise by weighting noise coefficients by basis function values
		int xC = 0, yC = 0, zC = 0;
		float weight = 1;
		float result = 0;
		const int midX = mid[0];
		const int midY = mid[1];
		const int midZ = mid[2];

		// clang-format off
		ADD_WEIGHTED(-1,-1,-1); ADD_WEIGHTED(0,-1,-1); ADD_WEIGHTED(1,-1,-1);
		ADD_WEIGHTED(-1, 0,-1); ADD_WEIGHTED(0, 0,-1); ADD_WEIGHTED(1, 0,-1);
		ADD_WEIGHTED(-1, 1,-1); ADD_WEIGHTED(0, 1,-1); ADD_WEIGHTED(1, 1,-1);

		ADD_WEIGHTED(-1,-1, 0); ADD_WEIGHTED(0,-1, 0); ADD_WEIGHTED(1,-1, 0);
		ADD_WEIGHTED(-1, 0, 0); ADD_WEIGHTED(0, 0, 0); ADD_WEIGHTED(1, 0, 0);
		ADD_WEIGHTED(-1, 1, 0); ADD_WEIGHTED(0, 1, 0); ADD_WEIGHTED(1, 1, 0);

		ADD_WEIGHTED(-1,-1, 1); ADD_WEIGHTED(0,-1, 1); ADD_WEIGHTED(1,-1, 1);
		ADD_WEIGHTED(-1, 0, 1); ADD_WEIGHTED(0, 0, 1); ADD_WEIGHTED(1, 0, 1);
		ADD_WEIGHTED(-1, 1, 1); ADD_WEIGHTED(0, 1, 1); ADD_WEIGHTED(1, 1, 1);
		// clang-format on

		return result;
	}

#undef ADD_WEIGHTED

	template<int N>
	void WaveletNoise<N>::dxDyDz(const Vec3& p, Mat33& final) const
	{
		const gsl::span<const float> data = _noiseTileData;

		float result1 = 0;
		float result2 = 0;
		float result3 = 0;
		float weight = 1;

#define ADD_WEXT_INDEX(x,y,z) ( (fast_modulo<N>(midZ+(z))*N*N) + (fast_modulo<N>(midY+(y))*N)+ fast_modulo<N>(midX+(x)) )

#define ADD_WEIGHTED_EXTDX(x,y,z)\
	  weight = dw[0][(x) + 1] * w[1][(y) + 1] * w[2][(z) + 1] ; \
	  result2 += data[ADD_WEXT_INDEX(x,y,z)] * weight;\
	  result3 += data[ADD_WEXT_INDEX(x,y,z)] * weight;

#define ADD_WEIGHTED_EXTDY(x,y,z)\
	  weight = w[0][(x) + 1] * dw[1][(y) + 1] * w[2][(z) + 1] ; \
	  result1 += data[ADD_WEXT_INDEX(x,y,z)] * weight;\
	  result3 += data[ADD_WEXT_INDEX(x,y,z)] * weight;

#define ADD_WEIGHTED_EXTDZ(x,y,z)\
	  weight = w[0][(x) + 1] * w[1][(y) + 1] * dw[2][(z) + 1] ; \
	  result1 += data[ADD_WEXT_INDEX(x,y,z)] * weight;\
	  result2 += data[ADD_WEXT_INDEX(x,y,z)] * weight;

		const float midXf = ceil(p[0] - 0.5f);
		const float midYf = ceil(p[1] - 0.5f);
		const float midZf = ceil(p[2] - 0.5f);

		const float t0 = midXf - (p[0] - 0.5f);
		const float t1 = midYf - (p[1] - 0.5f);
		const float t2 = midZf - (p[2] - 0.5f);

		auto midX = static_cast<const int>(midXf);
		auto midY = static_cast<const int>(midYf);
		auto midZ = static_cast<const int>(midZf);

		///////////////////////////////////////////////////////////////////////////////////////
		// Evaluate splines
		///////////////////////////////////////////////////////////////////////////////////////
		Mat33 w, dw;
		evaluateQuadraticSplineBasisImpl(t0, w[0]);
		evaluateQuadraticSplineBasisImpl(t1, w[1]);
		evaluateQuadraticSplineBasisImpl(t2, w[2]);
		evaluateDQuadraticSplineBasisImpl(t0, dw[0]);
		evaluateDQuadraticSplineBasisImpl(t1, dw[1]);
		evaluateDQuadraticSplineBasisImpl(t2, dw[2]);

		// clang-format off
		///////////////////////////////////////////////////////////////////////////////////////
		// x derivative
		///////////////////////////////////////////////////////////////////////////////////////
		result2 = result3 = 0.0f;
		ADD_WEIGHTED_EXTDX(-1,-1,-1); ADD_WEIGHTED_EXTDX(0,-1,-1); ADD_WEIGHTED_EXTDX(1,-1,-1);
		ADD_WEIGHTED_EXTDX(-1, 0,-1); ADD_WEIGHTED_EXTDX(0, 0,-1); ADD_WEIGHTED_EXTDX(1, 0,-1);
		ADD_WEIGHTED_EXTDX(-1, 1,-1); ADD_WEIGHTED_EXTDX(0, 1,-1); ADD_WEIGHTED_EXTDX(1, 1,-1);

		ADD_WEIGHTED_EXTDX(-1,-1, 0); ADD_WEIGHTED_EXTDX(0,-1, 0); ADD_WEIGHTED_EXTDX(1,-1, 0);
		ADD_WEIGHTED_EXTDX(-1, 0, 0); ADD_WEIGHTED_EXTDX(0, 0, 0); ADD_WEIGHTED_EXTDX(1, 0, 0);
		ADD_WEIGHTED_EXTDX(-1, 1, 0); ADD_WEIGHTED_EXTDX(0, 1, 0); ADD_WEIGHTED_EXTDX(1, 1, 0);

		ADD_WEIGHTED_EXTDX(-1,-1, 1); ADD_WEIGHTED_EXTDX(0,-1, 1); ADD_WEIGHTED_EXTDX(1,-1, 1);
		ADD_WEIGHTED_EXTDX(-1, 0, 1); ADD_WEIGHTED_EXTDX(0, 0, 1); ADD_WEIGHTED_EXTDX(1, 0, 1);
		ADD_WEIGHTED_EXTDX(-1, 1, 1); ADD_WEIGHTED_EXTDX(0, 1, 1); ADD_WEIGHTED_EXTDX(1, 1, 1);
		final[1][0] = result2;
		final[2][0] = result3;

		///////////////////////////////////////////////////////////////////////////////////////
		// y derivative
		///////////////////////////////////////////////////////////////////////////////////////
		result1 = result3 = 0.0f;
		ADD_WEIGHTED_EXTDY(-1,-1,-1); ADD_WEIGHTED_EXTDY(0,-1,-1); ADD_WEIGHTED_EXTDY(1,-1,-1);
		ADD_WEIGHTED_EXTDY(-1, 0,-1); ADD_WEIGHTED_EXTDY(0, 0,-1); ADD_WEIGHTED_EXTDY(1, 0,-1);
		ADD_WEIGHTED_EXTDY(-1, 1,-1); ADD_WEIGHTED_EXTDY(0, 1,-1); ADD_WEIGHTED_EXTDY(1, 1,-1);

		ADD_WEIGHTED_EXTDY(-1,-1, 0); ADD_WEIGHTED_EXTDY(0,-1, 0); ADD_WEIGHTED_EXTDY(1,-1, 0);
		ADD_WEIGHTED_EXTDY(-1, 0, 0); ADD_WEIGHTED_EXTDY(0, 0, 0); ADD_WEIGHTED_EXTDY(1, 0, 0);
		ADD_WEIGHTED_EXTDY(-1, 1, 0); ADD_WEIGHTED_EXTDY(0, 1, 0); ADD_WEIGHTED_EXTDY(1, 1, 0);

		ADD_WEIGHTED_EXTDY(-1,-1, 1); ADD_WEIGHTED_EXTDY(0,-1, 1); ADD_WEIGHTED_EXTDY(1,-1, 1);
		ADD_WEIGHTED_EXTDY(-1, 0, 1); ADD_WEIGHTED_EXTDY(0, 0, 1); ADD_WEIGHTED_EXTDY(1, 0, 1);
		ADD_WEIGHTED_EXTDY(-1, 1, 1); ADD_WEIGHTED_EXTDY(0, 1, 1); ADD_WEIGHTED_EXTDY(1, 1, 1);
		final[0][1] = result1;
		final[2][1] = result3;

		///////////////////////////////////////////////////////////////////////////////////////
		// z derivative
		///////////////////////////////////////////////////////////////////////////////////////
		result1 = result2 = 0.0f;
		ADD_WEIGHTED_EXTDZ(-1,-1,-1); ADD_WEIGHTED_EXTDZ(0,-1,-1); ADD_WEIGHTED_EXTDZ(1,-1,-1);
		ADD_WEIGHTED_EXTDZ(-1, 0,-1); ADD_WEIGHTED_EXTDZ(0, 0,-1); ADD_WEIGHTED_EXTDZ(1, 0,-1);
		ADD_WEIGHTED_EXTDZ(-1, 1,-1); ADD_WEIGHTED_EXTDZ(0, 1,-1); ADD_WEIGHTED_EXTDZ(1, 1,-1);

		ADD_WEIGHTED_EXTDZ(-1,-1, 0); ADD_WEIGHTED_EXTDZ(0,-1, 0); ADD_WEIGHTED_EXTDZ(1,-1, 0);
		ADD_WEIGHTED_EXTDZ(-1, 0, 0); ADD_WEIGHTED_EXTDZ(0, 0, 0); ADD_WEIGHTED_EXTDZ(1, 0, 0);
		ADD_WEIGHTED_EXTDZ(-1, 1, 0); ADD_WEIGHTED_EXTDZ(0, 1, 0); ADD_WEIGHTED_EXTDZ(1, 1, 0);

		ADD_WEIGHTED_EXTDZ(-1,-1, 1); ADD_WEIGHTED_EXTDZ(0,-1, 1); ADD_WEIGHTED_EXTDZ(1,-1, 1);
		ADD_WEIGHTED_EXTDZ(-1, 0, 1); ADD_WEIGHTED_EXTDZ(0, 0, 1); ADD_WEIGHTED_EXTDZ(1, 0, 1);
		ADD_WEIGHTED_EXTDZ(-1, 1, 1); ADD_WEIGHTED_EXTDZ(0, 1, 1); ADD_WEIGHTED_EXTDZ(1, 1, 1);
		final[0][2] = result1;
		final[1][2] = result2;
		// clang-format on
#undef ADD_WEXT_INDEX
#undef ADD_WEIGHTED_EXTDX
#undef ADD_WEIGHTED_EXTDY
#undef ADD_WEIGHTED_EXTDZ
	}

	template<int N>
	typename WaveletNoise<N>::Vec3 WaveletNoise<N>::velocity(const Vec3& p) const
	{
		// clang-format off
		const Vec3 p1 = { p[0] + N/2, p[1]      , p[2]       };
		const Vec3 p2 = { p[0]      , p[1] + N/2, p[2]       };
		const Vec3 p3 = { p[0]      , p[1]      , p[2] + N/2 };
		// clang-format on

		const float f1y = dy(p1);
		const float f1z = dz(p1);

		const float f2x = dx(p2);
		const float f2z = dz(p2);

		const float f3x = dx(p3);
		const float f3y = dy(p3);

		Vec3 v;
		v[0] = f3y - f2z;
		v[1] = f1z - f3x;
		v[2] = f2x - f1y;
		return v;
	}

	template<int N>
	void WaveletNoise<N>::downsample(gsl::span<const float> from, gsl::span<float> to, int n, int stride) noexcept
	{
		const gsl::span<const float> a = ACoeffs;
		for (ptrdiff_t i = 0; i < n / 2; i++)
		{
			to[i * stride] = 0;
			for (ptrdiff_t k = 2 * i - 16; k < 2 * i + 16; k++)
			{
				to[i * stride] += a[16 + k - 2 * i] * from[fast_modulo<N>(k) * stride];
			}
		}
	}

	template<int N>
	void WaveletNoise<N>::upsample(gsl::span<const float> from, gsl::span<float> to, int n, int stride) noexcept
	{
		const gsl::span<const float> p = PCoeffs;
		for (ptrdiff_t i = 0; i < n; i++)
		{
			to[i * stride] = 0;
			for (ptrdiff_t k = i / 2; k <= i / 2 + 1; k++)
			{
				to[i * stride] += p[2 + i - 2 * k] * from[fast_modulo<N/2>(k) * stride];
			}
		}
	}

	template<int N>
	void WaveletNoise<N>::evaluateQuadraticSplineBasis(float p, Vec3& w, int& mid) const noexcept
	{
		const float midf = ceil(p - 0.5f);
		const float t = midf - (p - 0.5f);
		mid = static_cast<int>(midf);

		evaluateQuadraticSplineBasisImpl(t, w);
	}

	template<int N>
	void WaveletNoise<N>::evaluateQuadraticSplineBasisImpl(float t, Vec3& w) const noexcept
	{
		w[0] = t * t / 2.0f;
		w[2] = (1.0f - t) * (1.0f - t) / 2.0f;
		w[1] = 1.0f - w[0] - w[2];
	}

	template<int N>
	void WaveletNoise<N>::evaluateDQuadraticSplineBasis(float p, Vec3& w, int& mid) const noexcept
	{
		const float midf = ceil(p - 0.5f);
		const float t = midf - (p - 0.5f);
		mid = static_cast<int>(midf);

		evaluateDQuadraticSplineBasisImpl(t, w);
	}

	template<int N>
	void WaveletNoise<N>::evaluateDQuadraticSplineBasisImpl(float t, Vec3& w) const noexcept
	{
		w[0] = -t;
		w[2] = (1.0f - t);
		w[1] = 2.0f * t - 1.0f;
	}

	template<int N>
	const std::array<float, 32> WaveletNoise<N>::ACoeffs =
	{
		 0.000334f,-0.001528f, 0.000410f, 0.003545f,-0.000938f,-0.008233f, 0.002172f, 0.019120f,
		-0.005040f,-0.044412f, 0.011655f, 0.103311f,-0.025936f,-0.243780f, 0.033979f, 0.655340f,
		 0.655340f, 0.033979f,-0.243780f,-0.025936f, 0.103311f, 0.011655f,-0.044412f,-0.005040f,
		 0.019120f, 0.002172f,-0.008233f,-0.000938f, 0.003546f, 0.000410f,-0.001528f, 0.000334f
	};
	template<int N>
	const std::array<float, 4> WaveletNoise<N>::PCoeffs = { 0.25f, 0.75f, 0.75f, 0.25f };

	template class WaveletNoise<32>;
	template class WaveletNoise<64>;
	template class WaveletNoise<128>;
}}
#if defined(VCL_COMPILER_MSVC) && defined(VCL_CHECK_CORE_GUIDELINES) && (_MSC_VER >= 1910)
#	pragma warning(pop)
#endif
