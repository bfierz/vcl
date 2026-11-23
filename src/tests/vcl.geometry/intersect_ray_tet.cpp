/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2021 Basil Fierz
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

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

// C++ Standard Library
#include <algorithm>
#include <array>
#include <random>
#include <utility>

// Include the relevant parts from the library
#include <vcl/core/container/array.h>
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/geometry/intersect.h>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

namespace {
	std::array<Eigen::Vector3f, 8> positions = {
		Eigen::Vector3f(0, 0, 0),
		Eigen::Vector3f(1, 0, 0),
		Eigen::Vector3f(0, 1, 0),
		Eigen::Vector3f(1, 1, 0),
		Eigen::Vector3f(0, 0, 1),
		Eigen::Vector3f(1, 0, 1),
		Eigen::Vector3f(0, 1, 1),
		Eigen::Vector3f(1, 1, 1),
	};
	std::array<std::array<int, 4>, 5> tetrahedra = {
		std::make_array(0, 5, 3, 6),
		std::make_array(0, 1, 3, 5),
		std::make_array(0, 2, 6, 3),
		std::make_array(0, 6, 4, 5),
		std::make_array(3, 6, 5, 7),
	};

	//! Corner of the unit cube; all expected ray parameters below are exact in float
	const std::array<Eigen::Vector3f, 4> corner = {
		Eigen::Vector3f(0, 0, 0),
		Eigen::Vector3f(1, 0, 0),
		Eigen::Vector3f(0, 1, 0),
		Eigen::Vector3f(0, 0, 1),
	};

	//! Sets of faces, by the index of the opposite vertex; None stands for -1
	enum Faces
	{
		F0 = 1 << 0,
		F1 = 1 << 1,
		F2 = 1 << 2,
		F3 = 1 << 3,
		None = 1 << 4,
	};

	struct RayCase
	{
		const char* name;
		Eigen::Vector3f origin;
		Eigen::Vector3f direction;
		bool hit;
		float t_entry;
		float t_exit;
		int entry_faces; //!< Faces the ray may enter through
		int exit_faces;  //!< Faces the ray may leave through
	};

	// clang-format off
	const std::array<RayCase, 12> corner_cases = {{
		{ "through two faces",       { 0.25f, 0.25f, -1 },     { 0, 0, 1 },  true,  1.0f,  1.5f, F3, F0 },
		{ "unnormalized direction",  { 0.25f, 0.25f, -1 },     { 0, 0, 2 },  true,  0.5f,  0.75f, F3, F0 },
		{ "pointing away",           { 0.25f, 0.25f, -1 },     { 0, 0, -1 }, false, 0,     0, None, None },
		{ "origin past the exit",    { 0.25f, 0.25f, 1 },      { 0, 0, 1 },  false, 0,     0, None, None },
		{ "origin inside",           { 0.125f, 0.125f, 0.125f }, { 0, 0, 1 }, true, 0.0f,  0.625f, None, F0 },
		{ "origin on a face",        { 0.25f, 0.25f, 0 },      { 0, 0, 1 },  true,  0.0f,  0.5f, F3, F0 },
		{ "parallel to a face",      { 0.25f, 0.25f, -1 },     { 1, 0, 0 },  false, 0,     0, None, None },
		{ "inside a face plane",     { 0.25f, -1, 0 },         { 0, 1, 0 },  true,  1.0f,  1.75f, F2, F0 },
		{ "touching an edge",        { 0.5f, -1, 1 },          { 0, 1, -1 }, true,  1.0f,  1.0f, F2 | F3, F2 | F3 },
		{ "passing an edge",         { 0.5f, -1, 0.75f },      { 0, 1, -1 }, false, 0,     0, None, None },
		{ "touching a vertex",       { -1, -1, 1 },            { 1, 1, 0 },  true,  1.0f,  1.0f, F0 | F1 | F2, F0 | F1 | F2 },
		{ "missing the line",        { 2, 2, 2 },              { 1, 0, 0 },  false, 0,     0, None, None },
	}};
	// clang-format on

	//! Whether a returned face, after mapping its index through the vertex order, is in a set of faces
	bool isOneOf(int face, int faces, const std::array<int, 4>& order = { 0, 1, 2, 3 })
	{
		return (faces & (face < 0 ? None : 1 << order[face])) != 0;
	}

	//! Checks a corner case against the corner tetrahedron with its vertices in the given order
	template<typename Algorithm>
	void checkCase(const std::array<int, 4>& order, const RayCase& c)
	{
		using namespace Vcl::Geometry;

		const Tetrahedron<float, 3> tet{ corner[order[0]], corner[order[1]], corner[order[2]], corner[order[3]] };
		const Ray<float, 3> ray{ c.origin, c.direction };

		float t_entry = -1, t_exit = -1;
		int entry_face = -2, exit_face = -2;
		EXPECT_EQ(c.hit, intersects(tet, ray, t_entry, t_exit, entry_face, exit_face, Algorithm{})) << c.name;
		EXPECT_EQ(c.hit, intersects(tet, ray, t_entry, t_exit, Algorithm{})) << c.name;
		EXPECT_EQ(c.hit, intersects(tet, ray, Algorithm{})) << c.name;
		EXPECT_TRUE(isOneOf(entry_face, c.entry_faces, order)) << c.name << ", entry face " << entry_face;
		EXPECT_TRUE(isOneOf(exit_face, c.exit_faces, order)) << c.name << ", exit face " << exit_face;
		if (c.hit)
		{
			EXPECT_FLOAT_EQ(c.t_entry, t_entry) << c.name;
			EXPECT_FLOAT_EQ(c.t_exit, t_exit) << c.name;
		}
	}

	template<int Width, size_t... Lanes>
	Vcl::VectorScalar<float, Width> gather(const std::array<float, Width>& values, std::index_sequence<Lanes...>)
	{
		return Vcl::VectorScalar<float, Width>(values[Lanes]...);
	}

	//! Runs the corner cases in parallel, one case per lane
	template<typename Algorithm, int Width>
	void checkCornerCasesSimd()
	{
		using namespace Vcl::Geometry;
		using Vcl::select;

		using real_t = Vcl::VectorScalar<float, Width>;
		using vec3_t = Eigen::Matrix<real_t, 3, 1>;

		const auto lanes = std::make_index_sequence<Width>{};
		const auto lane_case = [](int lane) -> const RayCase& { return corner_cases[lane % corner_cases.size()]; };

		vec3_t origin, direction;
		for (int d = 0; d < 3; d++)
		{
			std::array<float, Width> o, dir;
			for (int lane = 0; lane < Width; lane++)
			{
				o[lane] = lane_case(lane).origin[d];
				dir[lane] = lane_case(lane).direction[d];
			}
			origin[d] = gather<Width>(o, lanes);
			direction[d] = gather<Width>(dir, lanes);
		}

		const auto broadcast = [](const Eigen::Vector3f& v) { return vec3_t{ real_t(v.x()), real_t(v.y()), real_t(v.z()) }; };
		const Tetrahedron<real_t, 3> tet{ broadcast(corner[0]), broadcast(corner[1]), broadcast(corner[2]), broadcast(corner[3]) };
		const Ray<real_t, 3> ray{ origin, direction };

		real_t t_entry, t_exit;
		Vcl::VectorScalar<int, Width> entry_face, exit_face;
		const real_t hit = select(intersects(tet, ray, t_entry, t_exit, entry_face, exit_face, Algorithm{}), real_t(1), real_t(0));
		for (int lane = 0; lane < Width; lane++)
		{
			const RayCase& c = lane_case(lane);
			EXPECT_EQ(c.hit ? 1.0f : 0.0f, hit[lane]) << c.name << ", lane " << lane;
			EXPECT_TRUE(isOneOf(entry_face[lane], c.entry_faces)) << c.name << ", lane " << lane << ", entry face " << entry_face[lane];
			EXPECT_TRUE(isOneOf(exit_face[lane], c.exit_faces)) << c.name << ", lane " << lane << ", exit face " << exit_face[lane];
			if (c.hit)
			{
				EXPECT_FLOAT_EQ(c.t_entry, t_entry[lane]) << c.name << ", lane " << lane;
				EXPECT_FLOAT_EQ(c.t_exit, t_exit[lane]) << c.name << ", lane " << lane;
			}
		}
	}

	//! Ring of tetrahedra around the edge between vertices 0 and 1, each with its own vertex order
	struct EdgeRing
	{
		static constexpr int Size = 6;

		std::array<Eigen::Vector3f, Size + 2> vertices;

		//! Vertex ids of each tetrahedron
		std::array<std::array<int, 4>, Size> tetrahedra;

		Vcl::Geometry::Tetrahedron<float, 3> tetrahedron(int i) const
		{
			const auto& t = tetrahedra[i];
			return { vertices[t[0]], vertices[t[1]], vertices[t[2]], vertices[t[3]] };
		}

		//! Sorted vertex ids of the face of tetrahedron i opposite to its vertex f
		std::array<int, 3> face(int i, int f) const
		{
			std::array<int, 3> ids;
			for (int k = 0, n = 0; k < 4; k++)
				if (k != f)
					ids[n++] = tetrahedra[i][k];
			std::sort(ids.begin(), ids.end());
			return ids;
		}

		//! Tetrahedron sharing the face of tetrahedron i opposite to its vertex f; -1 on the outside
		int neighbour(int i, int f) const
		{
			const auto shared = face(i, f);
			for (int j = 0; j < Size; j++)
				for (int g = 0; j != i && g < 4; g++)
					if (face(j, g) == shared)
						return j;
			return -1;
		}
	};

	EdgeRing makeEdgeRing(std::mt19937& rng)
	{
		std::uniform_real_distribution<float> uniform{ 0.0f, 1.0f };
		std::normal_distribution<float> gauss;
		const auto random_direction = [&]() { return Eigen::Vector3f{ gauss(rng), gauss(rng), gauss(rng) }.normalized().eval(); };

		EdgeRing ring;
		const Eigen::Vector3f a = random_direction() * 0.1f;
		const Eigen::Vector3f b = (a + random_direction()).eval();
		const Eigen::Vector3f axis = (b - a).normalized();
		const Eigen::Vector3f side0 = axis.unitOrthogonal();
		const Eigen::Vector3f side1 = axis.cross(side0);

		ring.vertices[0] = a;
		ring.vertices[1] = b;
		for (int i = 0; i < EdgeRing::Size; i++)
		{
			const float angle = (i + 0.8f * uniform(rng) - 0.4f) * 2 * 3.14159265f / EdgeRing::Size;
			const float radius = 0.5f + uniform(rng);
			ring.vertices[2 + i] = a + uniform(rng) * (b - a) + radius * (std::cos(angle) * side0 + std::sin(angle) * side1);
		}
		for (int i = 0; i < EdgeRing::Size; i++)
		{
			ring.tetrahedra[i] = { 0, 1, 2 + i, 2 + (i + 1) % EdgeRing::Size };
			std::shuffle(ring.tetrahedra[i].begin(), ring.tetrahedra[i].end(), rng);
		}
		return ring;
	}

	//! Ray from outside through a random point on the shared edge of a ring
	Vcl::Geometry::Ray<float, 3> rayThroughEdge(const EdgeRing& ring, std::mt19937& rng)
	{
		std::uniform_real_distribution<float> uniform{ 0.0f, 1.0f };
		std::normal_distribution<float> gauss;

		const Eigen::Vector3f& a = ring.vertices[0];
		const Eigen::Vector3f& b = ring.vertices[1];
		const Eigen::Vector3f target = a + (0.05f + 0.9f * uniform(rng)) * (b - a);
		const Eigen::Vector3f direction = Eigen::Vector3f{ gauss(rng), gauss(rng), gauss(rng) }.normalized();
		return { (target - 3 * direction).eval(), direction };
	}

	template<typename Algorithm>
	class RayTetIntersectionAlgorithms : public testing::Test
	{};

	using Algorithms = testing::Types<
		Vcl::Geometry::RayTetIntersectionAlgorithmSelector<Vcl::Geometry::RayTetIntersectionAlgorithm::Default>,
		Vcl::Geometry::RayTetIntersectionAlgorithmSelector<Vcl::Geometry::RayTetIntersectionAlgorithm::Haines>,
		Vcl::Geometry::RayTetIntersectionAlgorithmSelector<Vcl::Geometry::RayTetIntersectionAlgorithm::Plucker>>;
	TYPED_TEST_SUITE(RayTetIntersectionAlgorithms, Algorithms);
}

TEST(RayTetIntersection, IntersectCenter)
{
	using namespace Vcl::Geometry;
	using Vcl::Mathematics::equal;

	using ray3_t = Vcl::Geometry::Ray<float, 3>;

	for (const auto tetrahedron : tetrahedra)
	{
		const Tetrahedron<float, 3> tet{
			positions[tetrahedron[0]],
			positions[tetrahedron[1]],
			positions[tetrahedron[2]],
			positions[tetrahedron[3]],
		};
		EXPECT_GT(tet.computeSignedVolume(), 0);
		const auto center = tet.computeCenter();

		// Intersection points
		const auto& a = tet[0];
		const auto& b = tet[1];
		const auto& c = tet[2];
		const auto& d = tet[3];

		const ray3_t ray_face_abc{ (center + 2 * (center - d)).eval(), (d - center).normalized() };
		const ray3_t ray_face_abd{ (center + 2 * (center - c)).eval(), (c - center).normalized() };
		const ray3_t ray_face_acd{ (center + 2 * (center - b)).eval(), (b - center).normalized() };
		const ray3_t ray_face_bcd{ (center + 2 * (center - a)).eval(), (a - center).normalized() };

		EXPECT_TRUE(intersects(tet, ray_face_abc));
		EXPECT_TRUE(intersects(tet, ray_face_abd));
		EXPECT_TRUE(intersects(tet, ray_face_acd));
		EXPECT_TRUE(intersects(tet, ray_face_bcd));
	}
}

TEST(RayTetIntersection, NoIntersectionParallelRay)
{
	using namespace Vcl::Geometry;
	using Vcl::Mathematics::equal;

	using ray3_t = Vcl::Geometry::Ray<float, 3>;

	for (const auto tetrahedron : tetrahedra)
	{
		const Tetrahedron<float, 3> tet{
			positions[tetrahedron[0]],
			positions[tetrahedron[1]],
			positions[tetrahedron[2]],
			positions[tetrahedron[3]],
		};
		EXPECT_GT(tet.computeSignedVolume(), 0);
		const auto center = tet.computeCenter();

		// Intersection points
		const auto& a = tet[0];
		const auto& b = tet[1];
		const auto& c = tet[2];
		const auto& d = tet[3];

		const ray3_t ray_face_abc{ (center + 2 * (center - d)).eval(), (b - a).normalized() };
		const ray3_t ray_face_abd{ (center + 2 * (center - c)).eval(), (d - a).normalized() };
		const ray3_t ray_face_acd{ (center + 2 * (center - b)).eval(), (d - a).normalized() };
		const ray3_t ray_face_bcd{ (center + 2 * (center - a)).eval(), (d - b).normalized() };

		EXPECT_FALSE(intersects(tet, ray_face_abc));
		EXPECT_FALSE(intersects(tet, ray_face_abd));
		EXPECT_FALSE(intersects(tet, ray_face_acd));
		EXPECT_FALSE(intersects(tet, ray_face_bcd));
	}
}

TEST(RayTetIntersection, IntersectFacePlaneNotTet)
{
	using namespace Vcl::Geometry;
	using Vcl::Mathematics::equal;

	using ray3_t = Vcl::Geometry::Ray<float, 3>;

	for (const auto tetrahedron : tetrahedra)
	{
		const Tetrahedron<float, 3> tet{
			positions[tetrahedron[0]],
			positions[tetrahedron[1]],
			positions[tetrahedron[2]],
			positions[tetrahedron[3]],
		};
		EXPECT_GT(tet.computeSignedVolume(), 0);
		const auto center = tet.computeCenter();

		// Intersection points
		const auto& a = tet[0];
		const auto& b = tet[1];
		const auto& c = tet[2];
		const auto& d = tet[3];

		const auto center_abc = ((a + b + c) / 3.0f).eval();
		const auto center_abd = ((a + b + d) / 3.0f).eval();
		const auto center_acd = ((a + c + d) / 3.0f).eval();
		const auto center_bcd = ((b + c + d) / 3.0f).eval();

		const ray3_t ray_face_abc{ (center + 2 * (center - d)).eval(), (center_abc - a).normalized() };
		const ray3_t ray_face_abd{ (center + 2 * (center - c)).eval(), (center_abd - a).normalized() };
		const ray3_t ray_face_acd{ (center + 2 * (center - b)).eval(), (center_acd - a).normalized() };
		const ray3_t ray_face_bcd{ (center + 2 * (center - a)).eval(), (center_bcd - b).normalized() };

		EXPECT_FALSE(intersects(tet, ray_face_abc));
		EXPECT_FALSE(intersects(tet, ray_face_abd));
		EXPECT_FALSE(intersects(tet, ray_face_acd));
		EXPECT_FALSE(intersects(tet, ray_face_bcd));
	}
}

TEST(RayTetIntersection, DefaultAlgorithm)
{
	using namespace Vcl::Geometry;

	const Tetrahedron<float, 3> tet{ corner[0], corner[1], corner[2], corner[3] };
	for (const auto& c : corner_cases)
	{
		const Ray<float, 3> ray{ c.origin, c.direction };

		float t_entry = -1, t_exit = -1;
		int entry_face = -2, exit_face = -2;
		EXPECT_EQ(c.hit, intersects(tet, ray, t_entry, t_exit, entry_face, exit_face)) << c.name;
		EXPECT_EQ(c.hit, intersects(tet, ray, t_entry, t_exit)) << c.name;
		EXPECT_EQ(c.hit, intersects(tet, ray)) << c.name;
	}
}

TYPED_TEST(RayTetIntersectionAlgorithms, Corner)
{
	using namespace Vcl::Geometry;

	for (const auto& c : corner_cases)
		checkCase<TypeParam>({ 0, 1, 2, 3 }, c);
}

TYPED_TEST(RayTetIntersectionAlgorithms, VertexOrderDoesNotMatter)
{
	using namespace Vcl::Geometry;

	std::array<int, 4> order = { 0, 1, 2, 3 };
	do
	{
		for (const auto& c : corner_cases)
			checkCase<TypeParam>(order, c);
	} while (std::next_permutation(order.begin(), order.end()));
}

TYPED_TEST(RayTetIntersectionAlgorithms, FlatTetrahedron)
{
	using namespace Vcl::Geometry;

	const Tetrahedron<float, 3> tet{ { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 } };
	const Ray<float, 3> through{ { 0.25f, 0.25f, -1 }, { 0, 0, 1 } };
	const Ray<float, 3> inside_plane{ { -1, 0.25f, 0 }, { 1, 0, 0 } };

	EXPECT_FALSE(intersects(tet, through, TypeParam{}));
	EXPECT_FALSE(intersects(tet, inside_plane, TypeParam{}));
}

TYPED_TEST(RayTetIntersectionAlgorithms, CornerFloat4)
{
	checkCornerCasesSimd<TypeParam, 4>();
}

TYPED_TEST(RayTetIntersectionAlgorithms, CornerFloat8)
{
	checkCornerCasesSimd<TypeParam, 8>();
}

TYPED_TEST(RayTetIntersectionAlgorithms, CornerFloat16)
{
	checkCornerCasesSimd<TypeParam, 16>();
}

//! Rays through points on an edge shared by a ring of tetrahedra must hit at least one of them
TYPED_TEST(RayTetIntersectionAlgorithms, NoCrackAroundSharedEdge)
{
	using namespace Vcl::Geometry;

	std::mt19937 rng{ 5489 };
	const int rings = 200;
	const int rays_per_ring = 1000;

	int misses = 0;
	for (int r = 0; r < rings; r++)
	{
		const EdgeRing ring = makeEdgeRing(rng);
		for (int k = 0; k < rays_per_ring; k++)
		{
			const auto ray = rayThroughEdge(ring, rng);

			bool hit = false;
			for (int i = 0; i < EdgeRing::Size; i++)
				hit = hit || intersects(ring.tetrahedron(i), ray, TypeParam{});
			misses += hit ? 0 : 1;
		}
	}
	EXPECT_EQ(0, misses) << "of " << rings * rays_per_ring << " rays slipped between the tetrahedra";
}

/*!
 *	Walks rays through a ring of tetrahedra the way a mesh traversal does: from the tetrahedron
 *	the ray enters from the outside, always on to the neighbour behind the exit face. Counts the
 *	rays for which the walk reaches a neighbour that is not hit, has no exit face or loops.
 */
template<typename Algorithm>
int countFailedWalks(int rings, int rays_per_ring)
{
	using namespace Vcl::Geometry;

	std::mt19937 rng{ 5489 };

	int failed = 0;
	for (int r = 0; r < rings; r++)
	{
		const EdgeRing ring = makeEdgeRing(rng);
		for (int k = 0; k < rays_per_ring; k++)
		{
			const auto ray = rayThroughEdge(ring, rng);

			std::array<bool, EdgeRing::Size> hit;
			std::array<float, EdgeRing::Size> t_entry, t_exit;
			std::array<int, EdgeRing::Size> entry, exit;
			for (int i = 0; i < EdgeRing::Size; i++)
				hit[i] = intersects(ring.tetrahedron(i), ray, t_entry[i], t_exit[i], entry[i], exit[i], Algorithm{});

			// Start where the ray enters the ring from the outside
			int current = -1;
			for (int i = 0; i < EdgeRing::Size; i++)
				if (hit[i] && entry[i] >= 0 && ring.neighbour(i, entry[i]) < 0 && (current < 0 || t_entry[i] < t_entry[current]))
					current = i;

			bool ok = current >= 0;
			std::array<bool, EdgeRing::Size> visited = {};
			while (ok)
			{
				visited[current] = true;
				if (exit[current] < 0)
				{
					ok = false;
					break;
				}

				const int next = ring.neighbour(current, exit[current]);
				if (next < 0)
					break; // Left the ring

				ok = hit[next] && !visited[next];
				current = next;
			}
			failed += ok ? 0 : 1;
		}
	}
	return failed;
}

TEST(RayTetIntersection, PluckerWalksAroundSharedEdge)
{
	using Algorithm = Vcl::Geometry::RayTetIntersectionAlgorithmSelector<Vcl::Geometry::RayTetIntersectionAlgorithm::Plucker>;

	EXPECT_EQ(0, countFailedWalks<Algorithm>(200, 1000)) << "of " << 200 * 1000 << " rays";
}
