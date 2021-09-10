/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2016 Basil Fierz
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
#ifndef GLSL_UNIFORMBUFFER
#define GLSL_UNIFORMBUFFER

#ifdef __cplusplus
#	define UNIFORM_BUFFER(loc) struct

namespace std140 {
	struct vec3_u
	{
		float x, y, z;

		vec3_u() = default;
		vec3_u(float x_, float y_, float z_)
		: x(x_), y(y_), z(z_) {}
		vec3_u(const Eigen::Vector3f& v)
		: x(v.x()), y(v.y()), z(v.z()) {}
	};

	struct vec3 : public vec3_u
	{
		using vec3_u::vec3_u;

	private:
		float pad;
	};

	struct vec4
	{
		float x, y, z, w;

		vec4() {}
		vec4(float x_, float y_, float z_, float w_)
		: x(x_), y(y_), z(z_), w(w_) {}
		vec4(const Eigen::Vector4f& v)
		: x(v.x()), y(v.y()), z(v.z()), w(v.w()) {}
	};

	struct mat4
	{
		vec4 cols[4];

		mat4() {}
		mat4(const Eigen::Matrix4f& m)
		{
			cols[0] = vec4{ Eigen::Vector4f{ m.col(0) } };
			cols[1] = vec4{ Eigen::Vector4f{ m.col(1) } };
			cols[2] = vec4{ Eigen::Vector4f{ m.col(2) } };
			cols[3] = vec4{ Eigen::Vector4f{ m.col(3) } };
		}
	};
}

using std140::mat4;
using std140::vec3;
using std140::vec3_u;
using std140::vec4;

#else
#	define vec3_u vec3
#	define UNIFORM_BUFFER(loc) layout(std140, binding = loc) uniform
#endif // __cplusplus

#endif // GLSL_UNIFORMBUFFER
