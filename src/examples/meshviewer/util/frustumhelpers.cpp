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
#include "frustumhelpers.h"

// VCL
#include <vcl/geometry/marchingcubestables.h>

namespace Vcl { namespace Util {
	Eigen::Vector4f computeFrustumSize(const Eigen::Vector4f& frustum)
	{
		// tan(fov / 2)
		float scale = frustum.x();
		float ratio = frustum.y();
		float near_dist = frustum.z();
		float far_dist = frustum.w();

		float near_half_height = scale * near_dist;
		float near_half_width = near_half_height * ratio;

		float far_half_height = scale * far_dist;
		float far_half_width = far_half_height * ratio;

		return { near_half_width, near_half_height, far_half_width, far_half_height };
	}

	Eigen::Vector3f intersectRayPlane(const Eigen::Vector3f& p0, const Eigen::Vector3f& dir, const Eigen::Vector4f& plane)
	{
		Eigen::Vector3f N = plane.segment<3>(0);
		float d = plane.w();

		// Point on plane
		Eigen::Vector3f P = d * N;

		float t = (P - p0).dot(N) / dir.dot(N);
		return p0 + t * dir;
	}

	// https://de.wikipedia.org/wiki/Hessesche_Normalform#Abstand_2
	float computePlaneVertexDistance(const Eigen::Vector4f& eq, const Eigen::Vector3f& v)
	{
		Eigen::Vector3f N = eq.segment<3>(0);
		float d = eq.w();

		// Distance point-plane
		float dist = v.dot(N) - d;

		return dist;
	}

	void computePlaneFrustumIntersection(const Eigen::Vector4f& plane, const Eigen::Matrix4f& View, const Eigen::Vector4f& Frustum, std::vector<Eigen::Vector3f>& vertices)
	{
		Eigen::Vector3f N = plane.segment<3>(0);
		float d = plane.w();

		// Point on plane
		Eigen::Vector3f P = d * N;

		// Transform the plane normal to the view-space
		P = (View * Eigen::Vector4f(P.x(), P.y(), P.z(), 1)).segment<3>(0);
		N = View.block<3, 3>(0, 0) * N;
		d = P.dot(N);

		Eigen::Vector4f transformed_plane{ N.x(), N.y(), N.z(), d };

		// Compute the rays of the frustum from camera point into screen
		Eigen::Vector4f frustum_size = computeFrustumSize(Frustum);

		// Computed frustum points
		Eigen::Vector3f fc[8];

		// Points of the near plane
		Eigen::Vector3f point_on_near = Eigen::Vector3f(0, 0, -Frustum.z());
		fc[0] = point_on_near + Eigen::Vector3f(1, 0, 0) * frustum_size.x() - Eigen::Vector3f(0, 1, 0) * frustum_size.y();
		fc[1] = point_on_near - Eigen::Vector3f(1, 0, 0) * frustum_size.x() - Eigen::Vector3f(0, 1, 0) * frustum_size.y();
		fc[2] = point_on_near - Eigen::Vector3f(1, 0, 0) * frustum_size.x() + Eigen::Vector3f(0, 1, 0) * frustum_size.y();
		fc[3] = point_on_near + Eigen::Vector3f(1, 0, 0) * frustum_size.x() + Eigen::Vector3f(0, 1, 0) * frustum_size.y();

		// Distance of near plane
		float dn0 = computePlaneVertexDistance(transformed_plane, fc[0]);
		float dn1 = computePlaneVertexDistance(transformed_plane, fc[1]);
		float dn2 = computePlaneVertexDistance(transformed_plane, fc[2]);
		float dn3 = computePlaneVertexDistance(transformed_plane, fc[3]);

		// Points of the far plane
		Eigen::Vector3f point_on_far = Eigen::Vector3f(0, 0, -Frustum.w());
		fc[4] = point_on_far + Eigen::Vector3f(1, 0, 0) * frustum_size.z() - Eigen::Vector3f(0, 1, 0) * frustum_size.w();
		fc[5] = point_on_far - Eigen::Vector3f(1, 0, 0) * frustum_size.z() - Eigen::Vector3f(0, 1, 0) * frustum_size.w();
		fc[6] = point_on_far - Eigen::Vector3f(1, 0, 0) * frustum_size.z() + Eigen::Vector3f(0, 1, 0) * frustum_size.w();
		fc[7] = point_on_far + Eigen::Vector3f(1, 0, 0) * frustum_size.z() + Eigen::Vector3f(0, 1, 0) * frustum_size.w();

		// Distance of far plane
		float df0 = computePlaneVertexDistance(transformed_plane, fc[4]);
		float df1 = computePlaneVertexDistance(transformed_plane, fc[5]);
		float df2 = computePlaneVertexDistance(transformed_plane, fc[6]);
		float df3 = computePlaneVertexDistance(transformed_plane, fc[7]);

		// Compute MC table index
		int num_tri_idx = ((df3 >= 0) << 7) | ((df2 >= 0) << 6) | ((df1 >= 0) << 5) | ((df0 >= 0) << 4) |
						  ((dn3 >= 0) << 3) | ((dn2 >= 0) << 2) | ((dn1 >= 0) << 1) | ((dn0 >= 0) << 0);

		// How many triangles are we generating
		int num_tris = Vcl::Geometry::caseToNumPolys[num_tri_idx];

		// List of triangles
		auto tris = (Eigen::Vector4i*)Vcl::Geometry::edgeVertexList + (5 * num_tri_idx);

		// Prepare storage
		vertices.reserve(vertices.size() + 15);

		for (int t = 0; t < num_tris; t++)
		{
			const auto& tri = tris[t];
			for (int i = 0; i < 3; i++)
			{
				if (tri(i) < 4)
				{
					vertices.emplace_back(intersectRayPlane(fc[tri(i)], (fc[(tri(i) + 1) % 4] - fc[tri(i)]).normalized(), transformed_plane));
				} else if (tri(i) < 8)
				{
					vertices.emplace_back(intersectRayPlane(fc[tri(i)], (fc[4 + (tri(i) + 1) % 4] - fc[tri(i)]).normalized(), transformed_plane));
				} else
				{
					vertices.emplace_back(intersectRayPlane(fc[tri(i) - 4], (fc[tri(i) - 8] - fc[tri(i) - 4]).normalized(), transformed_plane));
				}
			}
		}
	}
}}
