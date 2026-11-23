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
#include <vcl/config/eigen.h>

// C++ standard library
#include <array>
#include <limits>

// VCL
#include <vcl/core/simd/vectorscalar.h>
#include <vcl/core/contract.h>
#include <vcl/geometry/ray.h>
#include <vcl/geometry/tetrahedron.h>
#include <vcl/math/math.h>

namespace Vcl { namespace Geometry {

	enum class RayBoxIntersectionAlgorithm
	{
		Default, ///< Selectes one of the defined implementations
		Barnes,  ///< Implementation by Tavian Barnes
		Ize,     ///< Implementation by Thiago Ize
		Pharr    ///< Implementation from the PBR book by Pharr, Jakob, Humphreys
	};

	template<RayBoxIntersectionAlgorithm Algorithm>
	class RayBoxIntersectionAlgorithmSelector
	{};
	template<>
	class RayBoxIntersectionAlgorithmSelector<RayBoxIntersectionAlgorithm::Barnes>
	{};
	template<>
	class RayBoxIntersectionAlgorithmSelector<RayBoxIntersectionAlgorithm::Ize>
	{};
	template<>
	class RayBoxIntersectionAlgorithmSelector<RayBoxIntersectionAlgorithm::Pharr>
	{};
	template<>
	class RayBoxIntersectionAlgorithmSelector<RayBoxIntersectionAlgorithm::Default> : public RayBoxIntersectionAlgorithmSelector<RayBoxIntersectionAlgorithm::Ize>
	{};

	/*!
	 *	\brief Ray-AABB intersection
	 *
	 *	Implementation based on the summary in
	 *	http://tavianator.com/fast-branchless-raybounding-box-intersections/
	 *	http://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/
	 *
	 *	\note Rays aligned with the border of the bounding box produce only very inconsistent intersections
	 */
	inline bool intersects(
		const Eigen::AlignedBox<float, 3>& box,
		const Ray<float, 3>& ray,
		RayBoxIntersectionAlgorithmSelector<RayBoxIntersectionAlgorithm::Barnes>)
	{
		using namespace Vcl::Mathematics;

		constexpr float ctmin = -std::numeric_limits<float>::infinity();
		constexpr float ctmax = std::numeric_limits<float>::infinity();
		float tmin = ctmin;
		float tmax = ctmax;

		for (int i = 0; i < 3; ++i)
		{
			float t1 = (box.min()[i] - ray.origin()[i]) * ray.invDirection()[i];
			float t2 = (box.max()[i] - ray.origin()[i]) * ray.invDirection()[i];

			tmin = max(tmin, min(min(t1, t2), ctmax));
			tmax = min(tmax, max(max(t1, t2), ctmin));
		}

		return tmax > max(tmin, { 0.0f });
	}

	/*!
	 *	\brief Ray-AABB intersection
	 *
	 *	Implementation from
	 *	https://www.solidangle.com/research/jcgt2013_robust_BVH-revised.pdf
	 */
	inline bool intersects(
		const Eigen::AlignedBox<float, 3>& box,
		const Ray<float, 3>& r,
		RayBoxIntersectionAlgorithmSelector<RayBoxIntersectionAlgorithm::Ize>)
	{
		using namespace Vcl::Mathematics;

		float txmin, txmax, tymin, tymax, tzmin, tzmax;

		Eigen::Vector3f bounds[] = { box.min(), box.max() };

		// clang-format off
		txmin = (bounds[    r.signs().x()].x() - r.origin().x()) * r.invDirection().x();
		txmax = (bounds[1 - r.signs().x()].x() - r.origin().x()) * r.invDirection().x();
		tymin = (bounds[    r.signs().y()].y() - r.origin().y()) * r.invDirection().y();
		tymax = (bounds[1 - r.signs().y()].y() - r.origin().y()) * r.invDirection().y();
		tzmin = (bounds[    r.signs().z()].z() - r.origin().z()) * r.invDirection().z();
		tzmax = (bounds[1 - r.signs().z()].z() - r.origin().z()) * r.invDirection().z();
		// clang-format on

		// Disallow any intersection that lies behind the start point of the ray
		float tmin = 0;
		float tmax = std::numeric_limits<float>::infinity();

		tmin = max(tzmin, max(tymin, max(txmin, tmin)));
		tmax = min(tzmax, min(tymax, min(txmax, tmax)));
		tmax *= 1.00000024f; // 4*eps, see eqs 28-32

		return tmin <= tmax;
	}

	template<typename Real, int Width>
	Vcl::VectorScalar<bool, Width> intersects(
		const Eigen::AlignedBox<Vcl::VectorScalar<Real, Width>, 3>& box,
		const Ray<Vcl::VectorScalar<Real, Width>, 3>& r,
		RayBoxIntersectionAlgorithmSelector<RayBoxIntersectionAlgorithm::Ize>)
	{
		using namespace Vcl::Mathematics;

		using real_t = Vcl::VectorScalar<Real, Width>;

		real_t txmin = select(r.signs().x() == 0, box.min().x(), box.max().x()) - r.origin().x();
		real_t txmax = select(r.signs().x() == 1, box.min().x(), box.max().x()) - r.origin().x();
		real_t tymin = select(r.signs().y() == 0, box.min().y(), box.max().y()) - r.origin().y();
		real_t tymax = select(r.signs().y() == 1, box.min().y(), box.max().y()) - r.origin().y();
		real_t tzmin = select(r.signs().z() == 0, box.min().z(), box.max().z()) - r.origin().z();
		real_t tzmax = select(r.signs().z() == 1, box.min().z(), box.max().z()) - r.origin().z();

		const real_t sign_txmin = select(txmin < real_t(0), real_t(-1), real_t(1));
		const real_t sign_txmax = select(txmax < real_t(0), real_t(-1), real_t(1));
		const real_t sign_tymin = select(tymin < real_t(0), real_t(-1), real_t(1));
		const real_t sign_tymax = select(tymax < real_t(0), real_t(-1), real_t(1));
		const real_t sign_tzmin = select(tzmin < real_t(0), real_t(-1), real_t(1));
		const real_t sign_tzmax = select(tzmax < real_t(0), real_t(-1), real_t(1));

		txmin = select(isinf(r.invDirection().x()), sign_txmin * r.invDirection().x(), txmin * r.invDirection().x());
		txmax = select(isinf(r.invDirection().x()), sign_txmax * r.invDirection().x(), txmax * r.invDirection().x());
		tymin = select(isinf(r.invDirection().y()), sign_tymin * r.invDirection().y(), tymin * r.invDirection().y());
		tymax = select(isinf(r.invDirection().y()), sign_tymax * r.invDirection().y(), tymax * r.invDirection().y());
		tzmin = select(isinf(r.invDirection().z()), sign_tzmin * r.invDirection().z(), tzmin * r.invDirection().z());
		tzmax = select(isinf(r.invDirection().z()), sign_tzmax * r.invDirection().z(), tzmax * r.invDirection().z());

		// Disallow any intersection that lies behind the start point of the ray
		real_t tmin = 0;
		real_t tmax = std::numeric_limits<float>::infinity();

		tmin = max(tzmin, max(tymin, max(txmin, tmin)));
		tmax = min(tzmax, min(tymax, min(txmax, tmax)));
		tmax *= 1.00000024f;

		return tmin <= tmax;
	}

	/*!
	 *	\brief Ray-AABB intersection
	 *
	 *	Method from Pharr, Humphrey:
	 *	https://pbr-book.org/3ed-2018/Shapes/Basic_Shape_Interface#RayndashBoundsIntersections
	 */
	inline bool intersects(
		const Eigen::AlignedBox<float, 3>& box,
		const Ray<float, 3>& ray,
		RayBoxIntersectionAlgorithmSelector<RayBoxIntersectionAlgorithm::Pharr>)
	{
		using namespace Vcl::Mathematics;

		float t0 = 0;
		float t1 = std::numeric_limits<float>::infinity();

		for (int i = 0; i < 3; ++i)
		{
			float tNear = (box.min()[i] - ray.origin()[i]) * ray.invDirection()[i];
			float tFar = (box.max()[i] - ray.origin()[i]) * ray.invDirection()[i];

			if (tNear > tFar) std::swap(tNear, tFar);
			t0 = tNear > t0 ? tNear : t0;
			t1 = tFar < t1 ? tFar : t1;

			if (t0 > t1) return false;
		}

		return true;
	}

	/*!
	 * \brief Intersect two tetrahedra
	 *
	 * \note The code is based on the paper and implementation:
	 * Copyright(C) 2002 by Fabio Ganovelli, Federico Ponchio and Claudio Rocchini.
	 * "Fast tetrahedron-tetrahedron overlap algorithm"
	 * Reference implementation:
	 * https://github.com/erich666/jgt-code/blob/master/Volume_07/Number_2/Ganovelli2002/tet_a_tet.h
	 */
	bool intersects(const Tetrahedron<float, 3>& t0, const Tetrahedron<float, 3>& t1);

	enum class RayTetIntersectionAlgorithm
	{
		Default, ///< Selects one of the defined implementations
		Haines,  ///< Clipping against the face planes, by Eric Haines
		Plucker  ///< Signs of Plücker products, by Platis and Theoharis
	};

	template<RayTetIntersectionAlgorithm Algorithm>
	class RayTetIntersectionAlgorithmSelector
	{};
	template<>
	class RayTetIntersectionAlgorithmSelector<RayTetIntersectionAlgorithm::Haines>
	{};
	template<>
	class RayTetIntersectionAlgorithmSelector<RayTetIntersectionAlgorithm::Plucker>
	{};
	template<>
	class RayTetIntersectionAlgorithmSelector<RayTetIntersectionAlgorithm::Default> : public RayTetIntersectionAlgorithmSelector<RayTetIntersectionAlgorithm::Haines>
	{};

	/*!
	 *	\brief Ray-tetrahedron intersection
	 *
	 *	All algorithms share these semantics:
	 *	- A ray hits if it meets the tetrahedron at a parameter t >= 0. Hits on faces, edges and
	 *	  vertices count.
	 *	- t_entry is the ray parameter at which the ray enters the tetrahedron, 0 if the origin is
	 *	  inside. t_exit is the parameter at which it leaves. Both are only meaningful for rays that hit.
	 *	- The ray direction does not need to be normalized.
	 *	- The vertex order, and thus the orientation, of the tetrahedron does not matter.
	 *	- Degenerate (flat) tetrahedra are never hit.
	 *
	 *	The Haines algorithm clips the ray against the half-spaces bounded by the four face planes, as
	 *	described in E. Haines, "Fast Ray-Convex Polyhedron Intersection", Graphics Gems II, 1991.
	 *	Each tetrahedron computes its own face planes, so two tetrahedra sharing a face can round
	 *	differently and disagree on whether a ray grazing the face crosses it.
	 */
	template<typename Scalar>
	typename VectorTypes<Scalar>::bool_t intersects(
		const Tetrahedron<Scalar, 3>& tet,
		const Ray<Scalar, 3>& ray,
		Scalar& t_entry,
		Scalar& t_exit,
		RayTetIntersectionAlgorithmSelector<RayTetIntersectionAlgorithm::Haines>)
	{
		using namespace Vcl::Mathematics;

		using real_t = Scalar;

		// Vertices spanning the face opposite to vertex i
		const int faces[4][3] = { { 1, 2, 3 }, { 0, 2, 3 }, { 0, 1, 3 }, { 0, 1, 2 } };

		const real_t inf = std::numeric_limits<float>::infinity();
		const real_t neg_inf = -std::numeric_limits<float>::infinity();

		// Disallow any intersection that lies behind the start point of the ray
		real_t t_min = 0;
		real_t t_max = inf;

		for (int i = 0; i < 4; i++)
		{
			const auto& a = tet[faces[i][0]];
			const auto& b = tet[faces[i][1]];
			const auto& c = tet[faces[i][2]];

			// Orient the face normal away from the opposite vertex
			const auto n = (b - a).cross(c - a).eval();
			const real_t opposite = n.dot(tet[i] - a);
			const real_t s = select(opposite > real_t(0), real_t(-1), real_t(1));

			// Positive if the origin lies on the inner side of the face plane
			const real_t dist = s * n.dot(a - ray.origin());

			// Negative if the ray enters the inner side through this face
			const real_t denom = s * n.dot(ray.direction());

			// Ray parameter of the plane hit; not used for rays parallel to the plane
			const real_t t = dist / denom;

			t_min = select(denom < real_t(0), max(t_min, t), t_min);
			t_max = select(denom > real_t(0), min(t_max, t), t_max);

			// A ray parallel to the plane on its outer side, or a flat tetrahedron, misses
			t_max = select(denom == real_t(0) && dist < real_t(0), neg_inf, t_max);
			t_max = select(opposite == real_t(0), neg_inf, t_max);
		}

		t_entry = t_min;
		t_exit = t_max;

		return t_min <= t_max;
	}

	namespace Detail {
		//! Lexicographic order of two points, used to give every edge a fixed direction
		template<typename Scalar>
		typename VectorTypes<Scalar>::bool_t lexicographicLess(
			const Eigen::Matrix<Scalar, 3, 1>& a,
			const Eigen::Matrix<Scalar, 3, 1>& b)
		{
			return a.x() < b.x() || (a.x() == b.x() && (a.y() < b.y() || (a.y() == b.y() && a.z() < b.z())));
		}

		/*!
		 *	\brief Plücker product of the ray with the directed edge v0 -> v1
		 *
		 *	The product is computed for the lexicographically ordered edge and negated if needed.
		 *	This makes it bitwise the same, up to its sign, in every tetrahedron sharing the edge.
		 *
		 *	\param q  Ray direction
		 *	\param v0 Start point of the edge
		 *	\param v1 End point of the edge
		 *	\param p0 v0 relative to the ray origin
		 *	\param p1 v1 relative to the ray origin
		 */
		template<typename Scalar>
		Scalar pluckerProduct(
			const Eigen::Matrix<Scalar, 3, 1>& q,
			const Eigen::Matrix<Scalar, 3, 1>& v0,
			const Eigen::Matrix<Scalar, 3, 1>& v1,
			const Eigen::Matrix<Scalar, 3, 1>& p0,
			const Eigen::Matrix<Scalar, 3, 1>& p1)
		{
			using vector_t = Eigen::Matrix<Scalar, 3, 1>;

			const auto ordered = lexicographicLess(v0, v1);
			const vector_t first = select(ordered, p0, p1);
			const vector_t second = select(ordered, p1, p0);
			const Scalar product = q.dot(first.cross(second));

			return select(ordered, product, Scalar(0) - product);
		}
	}

	/*!
	 *	\brief Ray-tetrahedron intersection
	 *
	 *	Classifies the faces by the signs of the Plücker products of the ray with the six edges and
	 *	computes the ray parameters from the barycentric coordinates of the crossing points, as
	 *	described in N. Platis, T. Theoharis, "Fast Ray-Tetrahedron Intersection Using Plücker
	 *	Coordinates", Journal of Graphics Tools 8(4), 2003.
	 *	The product of an edge is bitwise the same in every tetrahedron sharing it, so tetrahedra
	 *	sharing a face always agree on whether the ray crosses that face.
	 *
	 *	\see RayTetIntersectionAlgorithm for the semantics shared by all algorithms
	 */
	template<typename Scalar>
	typename VectorTypes<Scalar>::bool_t intersects(
		const Tetrahedron<Scalar, 3>& tet,
		const Ray<Scalar, 3>& ray,
		Scalar& t_entry,
		Scalar& t_exit,
		RayTetIntersectionAlgorithmSelector<RayTetIntersectionAlgorithm::Plucker>)
	{
		using namespace Vcl::Mathematics;

		using real_t = Scalar;
		using vector_t = Eigen::Matrix<real_t, 3, 1>;

		const real_t zero = 0;
		const real_t inf = std::numeric_limits<float>::infinity();
		const real_t neg_inf = -std::numeric_limits<float>::infinity();

		const vector_t& q = ray.direction();
		const vector_t p[4] = {
			tet[0] - ray.origin(),
			tet[1] - ray.origin(),
			tet[2] - ray.origin(),
			tet[3] - ray.origin(),
		};

		// Products of the ray with the edges i -> j
		const real_t e01 = Detail::pluckerProduct(q, tet[0], tet[1], p[0], p[1]);
		const real_t e02 = Detail::pluckerProduct(q, tet[0], tet[2], p[0], p[2]);
		const real_t e03 = Detail::pluckerProduct(q, tet[0], tet[3], p[0], p[3]);
		const real_t e12 = Detail::pluckerProduct(q, tet[1], tet[2], p[1], p[2]);
		const real_t e13 = Detail::pluckerProduct(q, tet[1], tet[3], p[1], p[3]);
		const real_t e23 = Detail::pluckerProduct(q, tet[2], tet[3], p[2], p[3]);

		// Faces opposite to each vertex as (j, k, l), wound outwards for a positive volume, and the
		// products of their edges j -> k, k -> l, l -> j
		const int faces[4][3] = { { 1, 2, 3 }, { 0, 3, 2 }, { 0, 1, 3 }, { 0, 2, 1 } };
		const real_t edges[4][3] = {
			{ e12, e23, zero - e13 },
			{ e03, zero - e23, zero - e02 },
			{ e01, e13, zero - e03 },
			{ e02, zero - e12, zero - e01 },
		};

		const real_t volume = (tet[1] - tet[0]).cross(tet[2] - tet[0]).dot(tet[3] - tet[0]);
		const real_t orientation = select(volume < zero, real_t(-1), real_t(1));
		const real_t qq = q.dot(q);

		real_t t_min = neg_inf;
		real_t t_max = inf;
		real_t crossed = zero;
		for (int f = 0; f < 4; f++)
		{
			const real_t& jk = edges[f][0];
			const real_t& kl = edges[f][1];
			const real_t& lj = edges[f][2];

			// The line crosses the face if the products agree in sign. If all are zero, the line lies
			// in the plane of the face and crosses it through the neighbouring faces.
			const real_t sum = jk + kl + lj;
			const auto crosses =
				((jk >= zero && kl >= zero && lj >= zero) || (jk <= zero && kl <= zero && lj <= zero)) && sum != zero;

			// Barycentric coordinates of the crossing point: each vertex is weighted by the product
			// of the opposite edge
			const vector_t x = (kl * p[faces[f][0]] + lj * p[faces[f][1]] + jk * p[faces[f][2]]) / sum;
			const real_t t = q.dot(x) / qq;

			// The sum is the ray direction projected on the face normal; negative if the ray enters
			const real_t facing = orientation * sum;
			t_min = select(crosses && facing < zero, max(t_min, t), t_min);
			t_max = select(crosses && facing > zero, min(t_max, t), t_max);
			crossed = crossed + select(crosses, real_t(1), zero);
		}

		// A flat tetrahedron misses
		t_max = select(volume == zero, neg_inf, t_max);

		// Disallow any intersection that lies behind the start point of the ray
		t_entry = max(t_min, zero);
		t_exit = t_max;

		return crossed > zero && t_entry <= t_exit;
	}

	/*!
	 *	\brief Ray-tetrahedron intersection with the default algorithm
	 *	\see RayTetIntersectionAlgorithm
	 */
	template<typename Scalar>
	typename VectorTypes<Scalar>::bool_t intersects(
		const Tetrahedron<Scalar, 3>& tet,
		const Ray<Scalar, 3>& ray,
		Scalar& t_entry,
		Scalar& t_exit)
	{
		return intersects(tet, ray, t_entry, t_exit, RayTetIntersectionAlgorithmSelector<RayTetIntersectionAlgorithm::Default>{});
	}

	/*!
	 *	\brief Ray-tetrahedron intersection without the ray parameters
	 *	\see RayTetIntersectionAlgorithm
	 */
	template<typename Scalar, RayTetIntersectionAlgorithm Algorithm>
	typename VectorTypes<Scalar>::bool_t intersects(
		const Tetrahedron<Scalar, 3>& tet,
		const Ray<Scalar, 3>& ray,
		RayTetIntersectionAlgorithmSelector<Algorithm> algorithm)
	{
		Scalar t_entry, t_exit;
		return intersects(tet, ray, t_entry, t_exit, algorithm);
	}

	/*!
	 *	\brief Ray-tetrahedron intersection with the default algorithm, without the ray parameters
	 *	\see RayTetIntersectionAlgorithm
	 */
	template<typename Scalar>
	typename VectorTypes<Scalar>::bool_t intersects(
		const Tetrahedron<Scalar, 3>& tet,
		const Ray<Scalar, 3>& ray)
	{
		return intersects(tet, ray, RayTetIntersectionAlgorithmSelector<RayTetIntersectionAlgorithm::Default>{});
	}
}}
