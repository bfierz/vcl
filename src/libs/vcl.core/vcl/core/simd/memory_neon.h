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

namespace Vcl
{
#if defined VCL_VECTORIZE_NEON
	VCL_STRONG_INLINE float32x4_t gather(float const* base, uint32x4_t vindex)
	{
		float VCL_ALIGN(16) data[4] =
		{
			base[vindex.n128_i32[0]],
			base[vindex.n128_i32[1]],
			base[vindex.n128_i32[2]],
			base[vindex.n128_i32[3]]
		};

		return vld1q_f32(data);
	}

	VCL_STRONG_INLINE VectorScalar<float, 4> gather(float const * base, const VectorScalar<int, 4>& vindex)
	{
		return VectorScalar<float, 4>(gather(base, vindex.get(0)));
	}

	VCL_STRONG_INLINE VectorScalar<float, 8> gather(float const * base, const VectorScalar<int, 8>& vindex)
	{
		return VectorScalar<float, 8>
		(
			gather(base, vindex.get(0)),
			gather(base, vindex.get(1))
		);
	}
	VCL_STRONG_INLINE VectorScalar<float, 16> gather(float const * base, const VectorScalar<int, 16>& vindex)
	{
		return VectorScalar<float, 16>
		(
			gather(base, vindex.get(0)),
			gather(base, vindex.get(1)),
			gather(base, vindex.get(2)),
			gather(base, vindex.get(3))
		);
	}

	VCL_STRONG_INLINE void load(float4& value, const float* base)
	{
		value = float4{ vld1q_f32(base) };
	}

	// https://software.intel.com/en-us/articles/3d-vector-normalization-using-256-bit-intel-advanced-vector-extensions-intel-avx
	VCL_STRONG_INLINE void load
	(
		float32x4_t& x, float32x4_t& y, float32x4_t& z,
		const Eigen::Vector3f* base
	)
	{
		const float* p = base->data();
		float32x4x3_t reg = vld3q_f32(p);

		x = reg.val[0];
		y = reg.val[1];
		z = reg.val[2];
	}
	
	VCL_STRONG_INLINE void load
	(
		float32x4_t& x, float32x4_t& y, float32x4_t& z, float32x4_t& w,
		const Eigen::Vector4f* base
	)
	{
		const float* p = base->data();
		float32x4_t m0 = vld1q_f32(p + 0);
		float32x4_t m1 = vld1q_f32(p + 4);
		float32x4_t m2 = vld1q_f32(p + 8);
		float32x4_t m3 = vld1q_f32(p + 12);

		vtrnq_f32(m0, m1);
		vtrnq_f32(m2, m3);

		x = vcombine_f32(vget_low_f32 (m0), vget_high_f32(m2));
		y = vcombine_f32(vget_low_f32 (m1), vget_high_f32(m3));
		z = vcombine_f32(vget_high_f32(m0), vget_low_f32 (m2));
		w = vcombine_f32(vget_high_f32(m0), vget_low_f32 (m2));
	}

	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3f* base,
		const float32x4_t& x, const float32x4_t& y, const float32x4_t& z
	)
	{
		float* p = base->data();

		float32x4x3_t reg = { x, y, z };
		vst3q_f32(p, reg);
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<float4, 3, 1>& loaded,
		const Eigen::Vector3f* base
	)
	{
		float32x4_t x0, y0, z0;
		load(x0, y0, z0, base);

		loaded(0) = float4(x0);
		loaded(1) = float4(y0);
		loaded(2) = float4(z0);
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<int4, 3, 1>& loaded,
		const Eigen::Vector3i* base
	)
	{
		float32x4_t x0, y0, z0;
		load(x0, y0, z0, reinterpret_cast<const Eigen::Vector3f*>(base));

		loaded(0) = int4{ vreinterpret_s32_f32(x0) };
		loaded(1) = int4{ vreinterpret_s32_f32(y0) };
		loaded(2) = int4{ vreinterpret_s32_f32(z0) };
	}
	
	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<float4, 4, 1>& loaded,
		const Eigen::Vector4f* base
	)
	{
		float32x4_t x0, y0, z0, w0;
		load(x0, y0, z0, w0, base);

		loaded(0) = float4(x0);
		loaded(1) = float4(y0);
		loaded(2) = float4(z0);
		loaded(3) = float4(w0);
	}
	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<int4, 4, 1>& loaded,
		const Eigen::Vector4i* base
	)
	{
		float32x4_t x0, y0, z0, w0;
		load(x0, y0, z0, w0, reinterpret_cast<const Eigen::Vector4f*>(base));

		loaded(0) = int4{ vreinterpret_s32_f32(x0) };
		loaded(1) = int4{ vreinterpret_s32_f32(y0) };
		loaded(2) = int4{ vreinterpret_s32_f32(z0) };
		loaded(3) = int4{ vreinterpret_s32_f32(w0) };
	}
	
	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3f* base,
		const Eigen::Matrix<float4, 3, 1>& value
	)
	{
		store
		(
			base,
			value(0).get(0),
			value(1).get(0),
			value(2).get(0)
		);
	}
	
	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3i* base,
		const Eigen::Matrix<int4, 3, 1>& value
	)
	{
		store
		(
			reinterpret_cast<Eigen::Vector3f*>(base),
			vreinterpret_f32_s32(value(0).get(0)),
			vreinterpret_f32_s32(value(1).get(0)),
			vreinterpret_f32_s32(value(2).get(0))
		);
	}

	VCL_STRONG_INLINE void load(float8& value, const float* base)
	{
		value = float8{ vld1q_f32(base), vld1q_f32(base + 4) };
	}

	VCL_STRONG_INLINE void load(float16& value, const float* base)
	{
		value = float16
		{
			vld1q_f32(base + 0), vld1q_f32(base +  4),
			vld1q_f32(base + 8), vld1q_f32(base + 12)
		};
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<float8, 3, 1>& loaded,
		const Eigen::Vector3f* base
	)
	{
		float32x4_t x0, x1, y0, y1, z0, z1;
		load(x0, y0, z0, base);
		load(x1, y1, z1, base + 4);

		loaded =
		{
			float8(x0, x1),
			float8(y0, y1),
			float8(z0, z1)
		};
	}
	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<int8, 3, 1>& loaded,
		const Eigen::Vector3i* base
	)
	{
		float32x4_t x0, x1, y0, y1, z0, z1;
		load(x0, y0, z0, reinterpret_cast<const Eigen::Vector3f*>(base));
		load(x1, y1, z1, reinterpret_cast<const Eigen::Vector3f*>(base) +4);

		loaded =
		{
			int8{ vreinterpret_s32_f32(x0), vreinterpret_s32_f32(x1) },
			int8{ vreinterpret_s32_f32(y0), vreinterpret_s32_f32(y1) },
			int8{ vreinterpret_s32_f32(z0), vreinterpret_s32_f32(z1) }
		};
	}
	
	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<float8, 4, 1>& loaded,
		const Eigen::Vector4f* base
	)
	{
		float32x4_t x0, x1, y0, y1, z0, z1, w0, w1;
		load(x0, y0, z0, w0, base);
		load(x1, y1, z1, w1, base + 4);

		loaded(0) = float8(x0, x1);
		loaded(1) = float8(y0, y1);
		loaded(2) = float8(z0, z1);
		loaded(3) = float8(w0, w1);
	}
	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<int8, 4, 1>& loaded,
		const Eigen::Vector4i* base
	)
	{
		float32x4_t x0, x1, y0, y1, z0, z1, w0, w1;
		load(x0, y0, z0, w0, reinterpret_cast<const Eigen::Vector4f*>(base));
		load(x1, y1, z1, w1, reinterpret_cast<const Eigen::Vector4f*>(base)+4);

		loaded(0) = int8{ vreinterpret_s32_f32(x0), vreinterpret_s32_f32(x1) };
		loaded(1) = int8{ vreinterpret_s32_f32(y0), vreinterpret_s32_f32(y1) };
		loaded(2) = int8{ vreinterpret_s32_f32(z0), vreinterpret_s32_f32(z1) };
		loaded(3) = int8{ vreinterpret_s32_f32(w0), vreinterpret_s32_f32(w1) };
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<float16, 3, 1>& loaded,
		const Eigen::Vector3f* base
	)
	{
		float32x4_t x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
		load(x0, y0, z0, base);
		load(x1, y1, z1, base + 4);
		load(x2, y2, z2, base + 8);
		load(x3, y3, z3, base + 12);

		loaded =
		{
			float16(x0, x1, x2, x3),
			float16(y0, y1, y2, y3),
			float16(z0, z1, z2, z3)
		};
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<int16, 3, 1>& loaded,
		const Eigen::Vector3i* base
	)
	{
		float32x4_t x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
		load(x0, y0, z0, reinterpret_cast<const Eigen::Vector3f*>(base));
		load(x1, y1, z1, reinterpret_cast<const Eigen::Vector3f*>(base) + 4);
		load(x2, y2, z2, reinterpret_cast<const Eigen::Vector3f*>(base) + 8);
		load(x3, y3, z3, reinterpret_cast<const Eigen::Vector3f*>(base) + 12);

		loaded =
		{
			int16{ vreinterpret_s32_f32(x0), vreinterpret_s32_f32(x1), vreinterpret_s32_f32(x2), vreinterpret_s32_f32(x3) },
			int16{ vreinterpret_s32_f32(y0), vreinterpret_s32_f32(y1), vreinterpret_s32_f32(y2), vreinterpret_s32_f32(y3) },
			int16{ vreinterpret_s32_f32(z0), vreinterpret_s32_f32(z1), vreinterpret_s32_f32(z2), vreinterpret_s32_f32(z3) }
		};
	}
	
	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<float16, 4, 1>& loaded,
		const Eigen::Vector4f* base
	)
	{
		float32x4_t x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3, w0, w1, w2, w3;
		load(x0, y0, z0, w0, base);
		load(x1, y1, z1, w1, base + 4);
		load(x2, y2, z2, w2, base + 8);
		load(x3, y3, z3, w3, base + 12);

		loaded =
		{
			float16(x0, x1, x2, x3),
			float16(y0, y1, y2, y3),
			float16(z0, z1, z2, z3),
			float16(w0, w1, w2, w3)
		};
	}

	VCL_STRONG_INLINE void load
	(
		Eigen::Matrix<int16, 4, 1>& loaded,
		const Eigen::Vector4i* base
	)
	{
		float32x4_t x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3, w0, w1, w2, w3;
		load(x0, y0, z0, w0, reinterpret_cast<const Eigen::Vector4f*>(base));
		load(x1, y1, z1, w1, reinterpret_cast<const Eigen::Vector4f*>(base) + 4);
		load(x2, y2, z2, w2, reinterpret_cast<const Eigen::Vector4f*>(base) + 8);
		load(x3, y3, z3, w3, reinterpret_cast<const Eigen::Vector4f*>(base) + 12);

		loaded =
		{
			int16{ vreinterpret_s32_f32(x0), vreinterpret_s32_f32(x1), vreinterpret_s32_f32(x2), vreinterpret_s32_f32(x3) },
			int16{ vreinterpret_s32_f32(y0), vreinterpret_s32_f32(y1), vreinterpret_s32_f32(y2), vreinterpret_s32_f32(y3) },
			int16{ vreinterpret_s32_f32(z0), vreinterpret_s32_f32(z1), vreinterpret_s32_f32(z2), vreinterpret_s32_f32(z3) },
			int16{ vreinterpret_s32_f32(w0), vreinterpret_s32_f32(w1), vreinterpret_s32_f32(w2), vreinterpret_s32_f32(w3) }
		};
	}

	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3f* base,
		const Eigen::Matrix<float8, 3, 1>& value
	)
	{
		store(base + 0, value(0).get(0), value(1).get(0), value(2).get(0));
		store(base + 4, value(0).get(1), value(1).get(1), value(2).get(1));
	}
	
	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3i* base,
		const Eigen::Matrix<int8, 3, 1>& value
	)
	{
		store
		(
			reinterpret_cast<Eigen::Vector3f*>(base) + 0,
			vreinterpret_f32_s32(value(0).get(0)),
			vreinterpret_f32_s32(value(1).get(0)),
			vreinterpret_f32_s32(value(2).get(0))
		);
		
		store
		(
			reinterpret_cast<Eigen::Vector3f*>(base) + 4,
			vreinterpret_f32_s32(value(0).get(1)),
			vreinterpret_f32_s32(value(1).get(1)),
			vreinterpret_f32_s32(value(2).get(1))
		);
	}
	
	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3f* base,
		const Eigen::Matrix<float16, 3, 1>& value
	)
	{
		store(base +  0, value(0).get(0),value(1).get(0), value(2).get(0));
		store(base +  4, value(0).get(1),value(1).get(1), value(2).get(1));
		store(base +  8, value(0).get(2),value(1).get(2), value(2).get(2));
		store(base + 12, value(0).get(3),value(1).get(3), value(2).get(3));
	}
	
	VCL_STRONG_INLINE void store
	(
		Eigen::Vector3i* base,
		const Eigen::Matrix<int16, 3, 1>& value
	)
	{
		store
		(
			reinterpret_cast<Eigen::Vector3f*>(base) +  0,
			vreinterpret_f32_s32(value(0).get(0)),
			vreinterpret_f32_s32(value(1).get(0)),
			vreinterpret_f32_s32(value(2).get(0))
		);
		
		store
		(
			reinterpret_cast<Eigen::Vector3f*>(base) +  4,
			vreinterpret_f32_s32(value(0).get(1)),
			vreinterpret_f32_s32(value(1).get(1)),
			vreinterpret_f32_s32(value(2).get(1))
		);
		store
		(
			reinterpret_cast<Eigen::Vector3f*>(base) +  8,
			vreinterpret_f32_s32(value(0).get(2)),
			vreinterpret_f32_s32(value(1).get(2)),
			vreinterpret_f32_s32(value(2).get(2))
		);
		
		store
		(
			reinterpret_cast<Eigen::Vector3f*>(base) + 12,
			vreinterpret_f32_s32(value(0).get(3)),
			vreinterpret_f32_s32(value(1).get(3)),
			vreinterpret_f32_s32(value(2).get(3))
		);
	}
#endif // defined VCL_VECTORIZE_NEON
}
