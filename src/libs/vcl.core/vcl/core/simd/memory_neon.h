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

// VCL
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/simd/intrinsics_neon.h>

#if defined VCL_VECTORIZE_NEON
namespace Vcl {
	VCL_STRONG_INLINE float32x4_t gather(float const* base, int32x4_t vindex)
	{
		int idx[4];
		vst1q_s32(idx, vindex);

		float data[4] = {
			base[idx[0]],
			base[idx[1]],
			base[idx[2]],
			base[idx[3]]
		};

		return vld1q_f32(data);
	}

	VCL_STRONG_INLINE VectorScalar<float, 4> gather(float const* base, const VectorScalar<int, 4>& vindex)
	{
		return VectorScalar<float, 4>(gather(base, vindex.get(0)));
	}

	VCL_STRONG_INLINE VectorScalar<float, 8> gather(float const* base, const VectorScalar<int, 8>& vindex)
	{
		return VectorScalar<float, 8>(
			gather(base, vindex.get(0)),
			gather(base, vindex.get(1)));
	}
	VCL_STRONG_INLINE VectorScalar<float, 16> gather(float const* base, const VectorScalar<int, 16>& vindex)
	{
		return VectorScalar<float, 16>(
			gather(base, vindex.get(0)),
			gather(base, vindex.get(1)),
			gather(base, vindex.get(2)),
			gather(base, vindex.get(3)));
	}

	VCL_STRONG_INLINE void load(float4& value, const float* base)
	{
		value = float4{ vld1q_f32(base) };
	}

	VCL_STRONG_INLINE void load(int4& value, const int* base)
	{
		value = int4{ vld1q_s32(base) };
	}

	VCL_STRONG_INLINE void load(
		float32x4_t& x,
		float32x4_t& y,
		const Eigen::Vector2f* base)
	{
		const float* p = base->data();
		float32x4x2_t reg = vld2q_f32(p);

		x = reg.val[0];
		y = reg.val[1];
	}

	VCL_STRONG_INLINE void load(
		float32x4_t& x,
		float32x4_t& y,
		float32x4_t& z,
		const Eigen::Vector3f* base)
	{
		const float* p = base->data();
		float32x4x3_t reg = vld3q_f32(p);

		x = reg.val[0];
		y = reg.val[1];
		z = reg.val[2];
	}

	VCL_STRONG_INLINE void load(
		float32x4_t& x,
		float32x4_t& y,
		float32x4_t& z,
		float32x4_t& w,
		const Eigen::Vector4f* base)
	{
		const float* p = base->data();
		float32x4x4_t reg = vld4q_f32(p);

		x = reg.val[0];
		y = reg.val[1];
		z = reg.val[2];
		w = reg.val[3];
	}

	VCL_STRONG_INLINE void store(
		Eigen::Vector3f* base,
		const float32x4_t& x,
		const float32x4_t& y,
		const float32x4_t& z)
	{
		float* p = base->data();

		float32x4x3_t reg = { x, y, z };
		vst3q_f32(p, reg);
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<float4, 2, 1>& loaded,
		const Eigen::Vector2f* base)
	{
		float32x4_t x0, y0;
		load(x0, y0, base);

		loaded(0) = float4(x0);
		loaded(1) = float4(y0);
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<int4, 2, 1>& loaded,
		const Eigen::Vector2i* base)
	{
		float32x4_t x0, y0;
		load(x0, y0, reinterpret_cast<const Eigen::Vector2f*>(base));

		loaded(0) = int4{ vreinterpretq_s32_f32(x0) };
		loaded(1) = int4{ vreinterpretq_s32_f32(y0) };
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<float4, 3, 1>& loaded,
		const Eigen::Vector3f* base)
	{
		float32x4_t x0, y0, z0;
		load(x0, y0, z0, base);

		loaded(0) = float4(x0);
		loaded(1) = float4(y0);
		loaded(2) = float4(z0);
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<int4, 3, 1>& loaded,
		const Eigen::Vector3i* base)
	{
		float32x4_t x0, y0, z0;
		load(x0, y0, z0, reinterpret_cast<const Eigen::Vector3f*>(base));

		loaded(0) = int4{ vreinterpretq_s32_f32(x0) };
		loaded(1) = int4{ vreinterpretq_s32_f32(y0) };
		loaded(2) = int4{ vreinterpretq_s32_f32(z0) };
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<float4, 4, 1>& loaded,
		const Eigen::Vector4f* base)
	{
		float32x4_t x0, y0, z0, w0;
		load(x0, y0, z0, w0, base);

		loaded(0) = float4(x0);
		loaded(1) = float4(y0);
		loaded(2) = float4(z0);
		loaded(3) = float4(w0);
	}
	VCL_STRONG_INLINE void load(
		Eigen::Matrix<int4, 4, 1>& loaded,
		const Eigen::Vector4i* base)
	{
		float32x4_t x0, y0, z0, w0;
		load(x0, y0, z0, w0, reinterpret_cast<const Eigen::Vector4f*>(base));

		loaded(0) = int4{ vreinterpretq_s32_f32(x0) };
		loaded(1) = int4{ vreinterpretq_s32_f32(y0) };
		loaded(2) = int4{ vreinterpretq_s32_f32(z0) };
		loaded(3) = int4{ vreinterpretq_s32_f32(w0) };
	}

	VCL_STRONG_INLINE void store(
		Eigen::Vector3f* base,
		const Eigen::Matrix<float4, 3, 1>& value)
	{
		store(
			base,
			value(0).get(0),
			value(1).get(0),
			value(2).get(0));
	}

	VCL_STRONG_INLINE void store(
		Eigen::Vector3i* base,
		const Eigen::Matrix<int4, 3, 1>& value)
	{
		store(
			reinterpret_cast<Eigen::Vector3f*>(base),
			vreinterpretq_f32_s32(value(0).get(0)),
			vreinterpretq_f32_s32(value(1).get(0)),
			vreinterpretq_f32_s32(value(2).get(0)));
	}

	VCL_STRONG_INLINE std::array<float4, 2> interleave(const float4& a, const float4& b) noexcept
	{
		float32x2_t a1 = vget_low_f32(a.get(0));
		float32x2_t b1 = vget_low_f32(b.get(0));
		float32x2x2_t r1 = vzip_f32(a1, b1);
		float4 low{ vcombine_f32(r1.val[0], r1.val[1]) };

		float32x2_t a2 = vget_high_f32(a.get(0));
		float32x2_t b2 = vget_high_f32(b.get(0));
		float32x2x2_t r2 = vzip_f32(a2, b2);
		float4 high{ vcombine_f32(r2.val[0], r2.val[1]) };

		return { low, high };
	}

	VCL_STRONG_INLINE std::array<float8, 2> interleave(const float8& a, const float8& b) noexcept
	{
		const auto l0h0 = interleave(float4(a.get(0)), float4(b.get(0)));
		const auto l1h1 = interleave(float4(a.get(1)), float4(b.get(1)));

		return { float8(l0h0[0].get(0), l0h0[1].get(0)), float8(l1h1[0].get(0), l1h1[1].get(0)) };
	}

	VCL_STRONG_INLINE std::array<float16, 2> interleave(const float16& a, const float16& b) noexcept
	{
		const auto l0h0 = interleave(float4(a.get(0)), float4(b.get(0)));
		const auto l1h1 = interleave(float4(a.get(1)), float4(b.get(1)));
		const auto l2h2 = interleave(float4(a.get(2)), float4(b.get(2)));
		const auto l3h3 = interleave(float4(a.get(3)), float4(b.get(3)));

		return { float16(l0h0[0].get(0), l0h0[1].get(0), l1h1[0].get(0), l1h1[1].get(0)), float16(l2h2[0].get(0), l2h2[1].get(0), l3h3[0].get(0), l3h3[1].get(0)) };
	}

	VCL_STRONG_INLINE void load(float8& value, const float* base)
	{
		value = float8{ vld1q_f32(base), vld1q_f32(base + 4) };
	}

	VCL_STRONG_INLINE void load(int8& value, const int* base)
	{
		value = int8{ vld1q_s32(base), vld1q_s32(base + 4) };
	}

	VCL_STRONG_INLINE void load(float16& value, const float* base)
	{
		value = float16{
			vld1q_f32(base + 0), vld1q_f32(base + 4),
			vld1q_f32(base + 8), vld1q_f32(base + 12)
		};
	}

	VCL_STRONG_INLINE void load(int16& value, const int* base)
	{
		value = int16{
			vld1q_s32(base + 0), vld1q_s32(base + 4),
			vld1q_s32(base + 8), vld1q_s32(base + 12)
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<float8, 2, 1>& loaded,
		const Eigen::Vector2f* base)
	{
		float32x4_t x0, x1, y0, y1;
		load(x0, y0, base);
		load(x1, y1, base + 4);

		loaded = {
			float8(x0, x1),
			float8(y0, y1)
		};
	}
	VCL_STRONG_INLINE void load(
		Eigen::Matrix<int8, 2, 1>& loaded,
		const Eigen::Vector2i* base)
	{
		float32x4_t x0, x1, y0, y1;
		load(x0, y0, reinterpret_cast<const Eigen::Vector2f*>(base));
		load(x1, y1, reinterpret_cast<const Eigen::Vector2f*>(base) + 4);

		loaded = {
			int8{ vreinterpretq_s32_f32(x0), vreinterpretq_s32_f32(x1) },
			int8{ vreinterpretq_s32_f32(y0), vreinterpretq_s32_f32(y1) }
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<float8, 3, 1>& loaded,
		const Eigen::Vector3f* base)
	{
		float32x4_t x0, x1, y0, y1, z0, z1;
		load(x0, y0, z0, base);
		load(x1, y1, z1, base + 4);

		loaded = {
			float8(x0, x1),
			float8(y0, y1),
			float8(z0, z1)
		};
	}
	VCL_STRONG_INLINE void load(
		Eigen::Matrix<int8, 3, 1>& loaded,
		const Eigen::Vector3i* base)
	{
		float32x4_t x0, x1, y0, y1, z0, z1;
		load(x0, y0, z0, reinterpret_cast<const Eigen::Vector3f*>(base));
		load(x1, y1, z1, reinterpret_cast<const Eigen::Vector3f*>(base) + 4);

		loaded = {
			int8{ vreinterpretq_s32_f32(x0), vreinterpretq_s32_f32(x1) },
			int8{ vreinterpretq_s32_f32(y0), vreinterpretq_s32_f32(y1) },
			int8{ vreinterpretq_s32_f32(z0), vreinterpretq_s32_f32(z1) }
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<float8, 4, 1>& loaded,
		const Eigen::Vector4f* base)
	{
		float32x4_t x0, x1, y0, y1, z0, z1, w0, w1;
		load(x0, y0, z0, w0, base);
		load(x1, y1, z1, w1, base + 4);

		loaded(0) = float8(x0, x1);
		loaded(1) = float8(y0, y1);
		loaded(2) = float8(z0, z1);
		loaded(3) = float8(w0, w1);
	}
	VCL_STRONG_INLINE void load(
		Eigen::Matrix<int8, 4, 1>& loaded,
		const Eigen::Vector4i* base)
	{
		float32x4_t x0, x1, y0, y1, z0, z1, w0, w1;
		load(x0, y0, z0, w0, reinterpret_cast<const Eigen::Vector4f*>(base));
		load(x1, y1, z1, w1, reinterpret_cast<const Eigen::Vector4f*>(base) + 4);

		loaded(0) = int8{ vreinterpretq_s32_f32(x0), vreinterpretq_s32_f32(x1) };
		loaded(1) = int8{ vreinterpretq_s32_f32(y0), vreinterpretq_s32_f32(y1) };
		loaded(2) = int8{ vreinterpretq_s32_f32(z0), vreinterpretq_s32_f32(z1) };
		loaded(3) = int8{ vreinterpretq_s32_f32(w0), vreinterpretq_s32_f32(w1) };
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<float16, 2, 1>& loaded,
		const Eigen::Vector2f* base)
	{
		float32x4_t x0, x1, x2, x3, y0, y1, y2, y3;
		load(x0, y0, base);
		load(x1, y1, base + 4);
		load(x2, y2, base + 8);
		load(x3, y3, base + 12);

		loaded = {
			float16(x0, x1, x2, x3),
			float16(y0, y1, y2, y3)
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<int16, 2, 1>& loaded,
		const Eigen::Vector2i* base)
	{
		float32x4_t x0, x1, x2, x3, y0, y1, y2, y3;
		load(x0, y0, reinterpret_cast<const Eigen::Vector2f*>(base));
		load(x1, y1, reinterpret_cast<const Eigen::Vector2f*>(base) + 4);
		load(x2, y2, reinterpret_cast<const Eigen::Vector2f*>(base) + 8);
		load(x3, y3, reinterpret_cast<const Eigen::Vector2f*>(base) + 12);

		loaded = {
			int16{ vreinterpretq_s32_f32(x0), vreinterpretq_s32_f32(x1), vreinterpretq_s32_f32(x2), vreinterpretq_s32_f32(x3) },
			int16{ vreinterpretq_s32_f32(y0), vreinterpretq_s32_f32(y1), vreinterpretq_s32_f32(y2), vreinterpretq_s32_f32(y3) }
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<float16, 3, 1>& loaded,
		const Eigen::Vector3f* base)
	{
		float32x4_t x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
		load(x0, y0, z0, base);
		load(x1, y1, z1, base + 4);
		load(x2, y2, z2, base + 8);
		load(x3, y3, z3, base + 12);

		loaded = {
			float16(x0, x1, x2, x3),
			float16(y0, y1, y2, y3),
			float16(z0, z1, z2, z3)
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<int16, 3, 1>& loaded,
		const Eigen::Vector3i* base)
	{
		float32x4_t x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
		load(x0, y0, z0, reinterpret_cast<const Eigen::Vector3f*>(base));
		load(x1, y1, z1, reinterpret_cast<const Eigen::Vector3f*>(base) + 4);
		load(x2, y2, z2, reinterpret_cast<const Eigen::Vector3f*>(base) + 8);
		load(x3, y3, z3, reinterpret_cast<const Eigen::Vector3f*>(base) + 12);

		loaded = {
			int16{ vreinterpretq_s32_f32(x0), vreinterpretq_s32_f32(x1), vreinterpretq_s32_f32(x2), vreinterpretq_s32_f32(x3) },
			int16{ vreinterpretq_s32_f32(y0), vreinterpretq_s32_f32(y1), vreinterpretq_s32_f32(y2), vreinterpretq_s32_f32(y3) },
			int16{ vreinterpretq_s32_f32(z0), vreinterpretq_s32_f32(z1), vreinterpretq_s32_f32(z2), vreinterpretq_s32_f32(z3) }
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<float16, 4, 1>& loaded,
		const Eigen::Vector4f* base)
	{
		float32x4_t x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3, w0, w1, w2, w3;
		load(x0, y0, z0, w0, base);
		load(x1, y1, z1, w1, base + 4);
		load(x2, y2, z2, w2, base + 8);
		load(x3, y3, z3, w3, base + 12);

		loaded = {
			float16(x0, x1, x2, x3),
			float16(y0, y1, y2, y3),
			float16(z0, z1, z2, z3),
			float16(w0, w1, w2, w3)
		};
	}

	VCL_STRONG_INLINE void load(
		Eigen::Matrix<int16, 4, 1>& loaded,
		const Eigen::Vector4i* base)
	{
		float32x4_t x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3, w0, w1, w2, w3;
		load(x0, y0, z0, w0, reinterpret_cast<const Eigen::Vector4f*>(base));
		load(x1, y1, z1, w1, reinterpret_cast<const Eigen::Vector4f*>(base) + 4);
		load(x2, y2, z2, w2, reinterpret_cast<const Eigen::Vector4f*>(base) + 8);
		load(x3, y3, z3, w3, reinterpret_cast<const Eigen::Vector4f*>(base) + 12);

		loaded = {
			int16{ vreinterpretq_s32_f32(x0), vreinterpretq_s32_f32(x1), vreinterpretq_s32_f32(x2), vreinterpretq_s32_f32(x3) },
			int16{ vreinterpretq_s32_f32(y0), vreinterpretq_s32_f32(y1), vreinterpretq_s32_f32(y2), vreinterpretq_s32_f32(y3) },
			int16{ vreinterpretq_s32_f32(z0), vreinterpretq_s32_f32(z1), vreinterpretq_s32_f32(z2), vreinterpretq_s32_f32(z3) },
			int16{ vreinterpretq_s32_f32(w0), vreinterpretq_s32_f32(w1), vreinterpretq_s32_f32(w2), vreinterpretq_s32_f32(w3) }
		};
	}

	VCL_STRONG_INLINE void store(
		Eigen::Vector3f* base,
		const Eigen::Matrix<float8, 3, 1>& value)
	{
		store(base + 0, value(0).get(0), value(1).get(0), value(2).get(0));
		store(base + 4, value(0).get(1), value(1).get(1), value(2).get(1));
	}

	VCL_STRONG_INLINE void store(
		Eigen::Vector3i* base,
		const Eigen::Matrix<int8, 3, 1>& value)
	{
		store(
			reinterpret_cast<Eigen::Vector3f*>(base) + 0,
			vreinterpretq_f32_s32(value(0).get(0)),
			vreinterpretq_f32_s32(value(1).get(0)),
			vreinterpretq_f32_s32(value(2).get(0)));

		store(
			reinterpret_cast<Eigen::Vector3f*>(base) + 4,
			vreinterpretq_f32_s32(value(0).get(1)),
			vreinterpretq_f32_s32(value(1).get(1)),
			vreinterpretq_f32_s32(value(2).get(1)));
	}

	VCL_STRONG_INLINE void store(
		Eigen::Vector3f* base,
		const Eigen::Matrix<float16, 3, 1>& value)
	{
		store(base + 0, value(0).get(0), value(1).get(0), value(2).get(0));
		store(base + 4, value(0).get(1), value(1).get(1), value(2).get(1));
		store(base + 8, value(0).get(2), value(1).get(2), value(2).get(2));
		store(base + 12, value(0).get(3), value(1).get(3), value(2).get(3));
	}

	VCL_STRONG_INLINE void store(
		Eigen::Vector3i* base,
		const Eigen::Matrix<int16, 3, 1>& value)
	{
		store(
			reinterpret_cast<Eigen::Vector3f*>(base) + 0,
			vreinterpretq_f32_s32(value(0).get(0)),
			vreinterpretq_f32_s32(value(1).get(0)),
			vreinterpretq_f32_s32(value(2).get(0)));

		store(
			reinterpret_cast<Eigen::Vector3f*>(base) + 4,
			vreinterpretq_f32_s32(value(0).get(1)),
			vreinterpretq_f32_s32(value(1).get(1)),
			vreinterpretq_f32_s32(value(2).get(1)));
		store(
			reinterpret_cast<Eigen::Vector3f*>(base) + 8,
			vreinterpretq_f32_s32(value(0).get(2)),
			vreinterpretq_f32_s32(value(1).get(2)),
			vreinterpretq_f32_s32(value(2).get(2)));

		store(
			reinterpret_cast<Eigen::Vector3f*>(base) + 12,
			vreinterpretq_f32_s32(value(0).get(3)),
			vreinterpretq_f32_s32(value(1).get(3)),
			vreinterpretq_f32_s32(value(2).get(3)));
	}
}
#endif // defined VCL_VECTORIZE_NEON
