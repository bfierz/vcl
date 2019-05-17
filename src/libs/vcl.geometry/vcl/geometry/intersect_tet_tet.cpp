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
#include <vcl/geometry/intersect.h>

namespace Vcl { namespace Geometry
{
	struct HalfSpaceCoordinates
	{
		// For each face of the first tetrahedron
		// store the halfspace each vertex of the
		// second tetrahedron belongs to
		int Masks[4];

		// Vertices coordinates in the affine space
		std::array<double, 4> Coord[4];
	};

	//! Computes the half space of tet points, given the normal \p n
	//! \param[in] diff Points of the tetrahedron relative to a point on the half space separator
	//! \param[in] n Normal defining the half space
	//! \param[out] coord 
	//! \param[out] maskEdges
	//! \returns True, if all points are in the same half space
	inline bool detHalfSpace(const std::array<Eigen::Vector3d, 4>& diff, const Eigen::Vector3d& n, std::array<double, 4>& coord, int& maskEdges)
	{
		maskEdges = 000;
		if ((coord[0] = diff[0].dot(n)) > 0) maskEdges  = 001;
		if ((coord[1] = diff[1].dot(n)) > 0) maskEdges |= 002;
		if ((coord[2] = diff[2].dot(n)) > 0) maskEdges |= 004;
		if ((coord[3] = diff[3].dot(n)) > 0) maskEdges |= 010;

		return maskEdges == 017;
	}

	//! Computes the half space of tet points, given the normal \p n
	//! \param[in] diff Points of the tetrahedron relative to a point on the half space separator
	//! \param[in] n Normal defining the half space
	//! \returns True, if all points are in the same half space
	inline bool detHalfSpace(const std::array<Eigen::Vector3d, 4>& diff, const Eigen::Vector3d& n)
	{
		return ((diff[0].dot(n)>0) &&
				(diff[1].dot(n)>0) &&
				(diff[2].dot(n)>0) &&
				(diff[3].dot(n)>0));
	}

	// EdgeA -------------------------------------------------------

	inline bool EdgeA(HalfSpaceCoordinates& ctx, const int & f0, const int & f1)
	{
		const auto& coord_f0 = ctx.Coord[f0];
		const auto& coord_f1 = ctx.Coord[f1];

		int  maskf0 = ctx.Masks[f0];
		int  maskf1 = ctx.Masks[f1];

		if ((maskf0 | maskf1) != 017) // if there is a vertex of b 
			return false;             // included in (-,-) return false

		maskf0 &= (maskf0 ^ maskf1);  // exclude the vertices in (+,+)
		maskf1 &= (maskf0 ^ maskf1);

		// edge 0: 0--1
		if (((maskf0 & 001) &&		// the vertex 0 of b is in (-,+)
			(maskf1 & 002)) &&		// the vertex 1 of b is in (+,-)
			(((coord_f0[1] * coord_f1[0]) -
			(coord_f0[0] * coord_f1[1]))  >0))
			// the edge of b (0,1) intersect (-,-) (see the paper)
			return false;

		if (((maskf0 & 002) && (maskf1 & 001)) && (((coord_f0[1] * coord_f1[0]) - (coord_f0[0] * coord_f1[1]))  < 0))
			return false;

		// edge 1: 0--2
		if (((maskf0 & 001) && (maskf1 & 004)) && (((coord_f0[2] * coord_f1[0]) - (coord_f0[0] * coord_f1[2]))  > 0))
			return false;

		if (((maskf0 & 004) && (maskf1 & 001)) && (((coord_f0[2] * coord_f1[0]) - (coord_f0[0] * coord_f1[2]))  < 0))
			return false;

		// edge 2: 0--3
		if (((maskf0 & 001) && (maskf1 & 010)) && (((coord_f0[3] * coord_f1[0]) - (coord_f0[0] * coord_f1[3]))  > 0))
			return false;

		if (((maskf0 & 010) && (maskf1 & 001)) && (((coord_f0[3] * coord_f1[0]) - (coord_f0[0] * coord_f1[3]))  < 0))
			return false;

		// edge 3: 1--2
		if (((maskf0 & 002) && (maskf1 & 004)) && (((coord_f0[2] * coord_f1[1]) - (coord_f0[1] * coord_f1[2]))  > 0))
			return false;

		if (((maskf0 & 004) && (maskf1 & 002)) && (((coord_f0[2] * coord_f1[1]) - (coord_f0[1] * coord_f1[2]))  < 0))
			return false;


		// edge 4: 1--3
		if (((maskf0 & 002) && (maskf1 & 010)) && (((coord_f0[3] * coord_f1[1]) - (coord_f0[1] * coord_f1[3]))  > 0))
			return false;

		if (((maskf0 & 010) && (maskf1 & 002)) && (((coord_f0[3] * coord_f1[1]) - (coord_f0[1] * coord_f1[3]))  < 0))
			return false;

		// edge 5: 2--3
		if (((maskf0 & 004) && (maskf1 & 010)) && (((coord_f0[3] * coord_f1[2]) - (coord_f0[2] * coord_f1[3])) > 0))
			return false;

		if (((maskf0 & 010) && (maskf1 & 004)) && (((coord_f0[3] * coord_f1[2]) - (coord_f0[2] * coord_f1[3])) < 0))
			return false;

		// There exists a separting plane supported by the edge shared by f0 and f1
		return true;
	}

	bool tet_a_tet(const std::array<Eigen::Vector3d, 4>& V1, const std::array<Eigen::Vector3d, 4>& V2)
	{
		HalfSpaceCoordinates ctx;
		
		// Points of V2 relative to base of V1
		std::array<Eigen::Vector3d, 4> diff = {
			V2[0] - V1[0],
			V2[1] - V1[0],
			V2[2] - V1[0],
			V2[3] - V1[0]
		};
		const Eigen::Vector3d edge_v1_0 = V1[1] - V1[0];
		const Eigen::Vector3d edge_v1_1 = V1[2] - V1[0];
		Eigen::Vector3d n = edge_v1_0.cross(edge_v1_1);
		if (detHalfSpace(diff, n, ctx.Coord[0], ctx.Masks[0]))
			return false;

		const Eigen::Vector3d edge_v1_2 = V1[3] - V1[0];
		n = edge_v1_2.cross(edge_v1_0);
		if (detHalfSpace(diff, n, ctx.Coord[1], ctx.Masks[1]))
			return false;

		if (EdgeA(ctx, 0, 1)) return false;

		n = edge_v1_1.cross(edge_v1_2);
		if (detHalfSpace(diff, n, ctx.Coord[2], ctx.Masks[2]))
			return false;

		if (EdgeA(ctx, 0, 2)) return false;
		if (EdgeA(ctx, 1, 2)) return false;

		// Points of V2 relative to V1
		diff = {
			V2[0] - V1[1],
			V2[1] - V1[1],
			V2[2] - V1[1],
			V2[3] - V1[1]
		};
		const Eigen::Vector3d edge_v1_4 = V1[3] - V1[1];
		const Eigen::Vector3d edge_v1_3 = V1[2] - V1[1];
		n = edge_v1_4.cross(edge_v1_3);
		if (detHalfSpace(diff, n, ctx.Coord[3], ctx.Masks[3]))
			return false;

		if (EdgeA(ctx, 0, 3)) return false;
		if (EdgeA(ctx, 1, 3)) return false;
		if (EdgeA(ctx, 2, 3)) return false;

		if ((ctx.Masks[0] | ctx.Masks[1] | ctx.Masks[2] | ctx.Masks[3]) != 017)
			return true;


		// from now on, if there is a separating plane it is parallel to a face of b

		// Points of V2 relative to base of V1
		diff = {
			V1[0] - V2[0],
			V1[1] - V2[0],
			V1[2] - V2[0],
			V1[3] - V2[0]
		};
		const Eigen::Vector3d edge_v2_0 = V2[1] - V2[0];
		const Eigen::Vector3d edge_v2_1 = V2[2] - V2[0];
		n = edge_v2_0.cross(edge_v2_1);
		if (detHalfSpace(diff, n))
			return false;

		const Eigen::Vector3d edge_v2_2 = V2[3] - V2[0];
		n = edge_v2_2.cross(edge_v2_0);
		if (detHalfSpace(diff, n))
			return false;

		n = edge_v2_1.cross(edge_v2_2);
		if (detHalfSpace(diff, n))
			return false;

		diff = {
			V1[0] - V2[1],
			V1[1] - V2[1],
			V1[2] - V2[1],
			V1[3] - V2[1]
		};
		const Eigen::Vector3d edge_v2_4 = V2[3] - V2[1];
		const Eigen::Vector3d edge_v2_3 = V2[2] - V2[1];
		n = edge_v2_4.cross(edge_v2_3);
		if (detHalfSpace(diff, n))
			return false;

		return true;
	}

	bool intersects(const Tetrahedron<float, 3>& t0, const Tetrahedron<float, 3>& t1)
	{
		std::array<Eigen::Vector3d, 4> V_1;
		std::array<Eigen::Vector3d, 4> V_2;
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 3; j++)
			{
				V_1[i][j] = t0[i][j];
				V_2[i][j] = t1[i][j];
			}

		return tet_a_tet(V_1, V_2);
	}
}}
