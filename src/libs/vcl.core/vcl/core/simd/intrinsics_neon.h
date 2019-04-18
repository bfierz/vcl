/* 
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014 Basil Fierz
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
#include <limits>
#include <cstdint>

#if defined(VCL_VECTORIZE_NEON)
namespace Vcl
{
	float32x4_t vsinq_f32(float32x4_t v);
	float32x4_t vcosq_f32(float32x4_t v);
	float32x4_t vlogq_f32(float32x4_t v);
	float32x4_t vexpq_f32(float32x4_t v);

	float32x4_t vacosq_f32(float32x4_t v);
	float32x4_t vasinq_f32(float32x4_t v);

	float32x4_t vatan2q_f32(float32x4_t y, float32x4_t x);
	float32x4_t vpowq_f32(float32x4_t x, float32x4_t y);

	VCL_STRONG_INLINE uint32x4_t visinfq_f32(float32x4_t x)
	{
		const uint32x4_t sign_mask = vdupq_n_u32(0x7fffffff);
		const uint32x4_t inf = vreinterpretq_u32_f32(vdupq_n_f32(std::numeric_limits<float>::infinity()));

		uint32x4_t r;
		r = vandq_u32(sign_mask, vreinterpretq_u32_f32(x));
		r = vceqq_u32(r, inf);
		return r;
	}

	// From: Eigen/src/Core/arch/NEON/PacketMath.h
	VCL_STRONG_INLINE float32x4_t vdivq_f32(float32x4_t x, float32x4_t y)
	{
		float32x4_t inv, restep, div;

		// NEON does not offer a divide instruction, we have to do a reciprocal approximation
		// However NEON in contrast to other SIMD engines (AltiVec/SSE), offers
		// a reciprocal estimate AND a reciprocal step -which saves a few instructions
		// vrecpeq_f32() returns an estimate to 1/b, which we will finetune with
		// Newton-Raphson and vrecpsq_f32()
		inv = vrecpeq_f32(y);

		// This returns a differential, by which we will have to multiply inv to get a better
		// approximation of 1/b.
		restep = vrecpsq_f32(y, inv);
		inv = vmulq_f32(restep, inv);

		// Do a second step
		restep = vrecpsq_f32(y, inv);
		inv = vmulq_f32(restep, inv);

		// Finally, multiply a by 1/b and get the wanted result of the division.
		div = vmulq_f32(x, inv);

		return div;
	}

	VCL_STRONG_INLINE float32x4_t vrcpq_f32(float32x4_t x)
	{
		float32x4_t inv, restep;

		// NEON does not offer a divide instruction, we have to do a reciprocal approximation
		// However NEON in contrast to other SIMD engines (AltiVec/SSE), offers
		// a reciprocal estimate AND a reciprocal step -which saves a few instructions
		// vrecpeq_f32() returns an estimate to 1/b, which we will finetune with
		// Newton-Raphson and vrecpsq_f32()
		inv = vrecpeq_f32(x);

		// This returns a differential, by which we will have to multiply inv to get a better
		// approximation of 1/b.
		restep = vrecpsq_f32(x, inv);
		inv = vmulq_f32(restep, inv);

		return inv;
	}

	VCL_STRONG_INLINE float32x4_t vrsqrtq_f32(float32x4_t x)
	{
		const float32x4_t q_step_0 = vrsqrteq_f32(x);
		// step
		const float32x4_t q_step_parm0 = vmulq_f32(x, q_step_0);
		const float32x4_t q_step_result0 = vrsqrtsq_f32(q_step_parm0, q_step_0);

		// step
		const float32x4_t q_step_1 = vmulq_f32(q_step_0, q_step_result0);
		const float32x4_t q_step_parm1 = vmulq_f32(x, q_step_1);
		const float32x4_t q_step_result1 = vrsqrtsq_f32(q_step_parm1, q_step_1);

		// take the res
		const float32x4_t q_step_2 = vmulq_f32(q_step_1, q_step_result1);

		return q_step_2;
	}

	float32x4_t vsqrtq_f32(float32x4_t x);

	VCL_STRONG_INLINE uint32x4_t vcneqq_f32(float32x4_t x, float32x4_t y)
	{
		return vmvnq_u32(vceqq_f32(x, y));
	}

	VCL_STRONG_INLINE float32x4_t vsgnq_f32(float32x4_t v)
	{
		return vreinterpretq_f32_u32(vandq_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(v), vdupq_n_u32(0x80000000)), vreinterpretq_u32_f32(vdupq_n_f32(1.0f))), vcneqq_f32(v, vdupq_n_f32(0))));
	}

	VCL_STRONG_INLINE int vmovemaskq_u32(uint32x4_t a)
	{
		static const uint32_t data[4] = { 1, 2, 4, 8 };
		static const uint32x4_t movemask = vld1q_u32(data);
		static const uint32x4_t highbit = vdupq_n_u32(0x80000000);

		uint32x4_t t0 = a;
		uint32x4_t t1 = vtstq_u32(t0, highbit);
		uint32x4_t t2 = vandq_u32(t1, movemask);
		uint32x2_t t3 = vorr_u32(vget_low_u32(t2), vget_high_u32(t2));

		return vget_lane_u32(t3, 0) | vget_lane_u32(t3, 1);
	}

	VCL_STRONG_INLINE int vmovemaskq_f32(float32x4_t a)
	{
		return vmovemaskq_u32(vreinterpretq_u32_f32(a));
	}

	VCL_STRONG_INLINE float32_t vpminq_f32(float32x4_t v)
	{
		float32x2_t tmp = vpmin_f32(vget_low_f32(v), vget_high_f32(v));
		tmp = vpmin_f32(tmp, tmp);
		return vget_lane_f32(tmp, 0);
	}

	VCL_STRONG_INLINE float32_t vpmaxq_f32(float32x4_t v)
	{
		float32x2_t tmp = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
		tmp = vpmax_f32(tmp, tmp);
		return vget_lane_f32(tmp, 0);
	}

	VCL_STRONG_INLINE float32_t vdotq_f32(float32x4_t a, float32x4_t b)
	{
		float32x4_t prod = vmulq_f32(a, b);
		float32x4_t sum1 = vaddq_f32(prod, vrev64q_f32(prod));
		float32x4_t sum2 = vaddq_f32(sum1, vcombine_f32(vget_high_f32(sum1), vget_low_f32(sum1)));
		return vgetq_lane_f32(sum2, 0);
	}

	/*__m128 _mmVCL_floor_ps(__m128 v);

	VCL_STRONG_INLINE __m128i _mmVCL_mullo_epi32(__m128i a, __m128i b)
	{
#ifdef VCL_VECTORIZE_SSE4_1
		return _mm_mullo_epi32(a, b);
#else
		__m128i tmp1 = _mm_mul_epu32(a, b); // mul 2,0
		__m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4)); // mul 3,1 
		return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 2, 0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0))); // shuffle results to [63..0] and pack
#endif
	}

	VCL_STRONG_INLINE float _mmVCL_hmin_ps(__m128 v)
	{
		__m128 data = v;             // [0, 1, 2, 3]
		__m128 low = _mm_movehl_ps(data, data); // [2, 3, 2, 3]
		__m128 low_accum = _mm_min_ps(low, data); // [0|2, 1|3, 2|2, 3|3]
		__m128 elem1 = _mm_shuffle_ps(low_accum, low_accum, _MM_SHUFFLE(1, 1, 1, 1)); // [1|3, 1|3, 1|3, 1|3]
		__m128 accum = _mm_min_ss(low_accum, elem1);
		return _mm_cvtss_f32(accum);
	}

	VCL_STRONG_INLINE float _mmVCL_hmax_ps(__m128 v)
	{
		__m128 data = v;             // [0, 1, 2, 3]
		__m128 high = _mm_movehl_ps(data, data); // [2, 3, 2, 3]
		__m128 high_accum = _mm_max_ps(high, data); // [0|2, 1|3, 2|2, 3|3] 
		__m128 elem1 = _mm_shuffle_ps(high_accum, high_accum, _MM_SHUFFLE(1, 1, 1, 1)); // [1|3, 1|3, 1|3, 1|3]
		__m128 accum = _mm_max_ss(high_accum, elem1);
		return _mm_cvtss_f32(accum);
	}

	VCL_STRONG_INLINE float _mmVCL_extract_ps(__m128 v, int i)
	{
		typedef union
		{
			__m128 x;
			float a[4];
		} F32;

		return F32 {v}.a[i];
	}

	VCL_STRONG_INLINE int _mmVCL_extract_epi32(__m128i v, int i)
	{
		typedef union
		{
			__m128i x;
			int32_t a[4];
		} U32;

		return U32 {v}.a[i];
	}*/
}
#endif // defined(VCL_VECTORIZE_NEON)
