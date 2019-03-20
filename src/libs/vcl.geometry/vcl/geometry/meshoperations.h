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
#pragma once

// VCL configuration
#include <vcl/config/global.h>

// C++ standard library
#include <algorithm>
#include <array>
#include <iostream>
#include <unordered_map>
#include <vector>

// VCL
#include <vcl/core/container/array.h>
#include <vcl/core/contract.h>
#include <vcl/core/span.h>
#include <vcl/geometry/cell.h>

namespace Vcl { namespace Geometry
{
	template<typename VertexId>
	size_t makeEdgeHash(VertexId p0, VertexId p1)
	{
		static_assert(sizeof(VertexId) == 4, "Only 32-bit Ids are supported");

		auto e = std::make_pair(std::min(p0, p1), std::max(p0, p1));
		size_t v0 = e.first.id();
		size_t v1 = e.second.id();

		return (v1 << 32) | v0;
	}

	template<typename VertexId>
	size_t makeFaceHash(VertexId p0, VertexId p1, VertexId p2)
	{
		VclRequire(p0.id() < (1 << 21), "Index p0 is 21-bit.");
		VclRequire(p1.id() < (1 << 21), "Index p1 is 21-bit.");
		VclRequire(p2.id() < (1 << 21), "Index p2 is 21-bit.");

		auto f = std::make_array(p0, p1, p2);
		std::sort(std::begin(f), std::end(f));

		size_t v0 = f[0].id();
		size_t v1 = f[1].id();
		size_t v2 = f[2].id();

		return (v2 << 42) | (v1 << 21) | v0;
	}

	template<typename VertexId>
	std::vector<std::array<VertexId, 3>> extractSurface(std::span<std::array<VertexId, 4>> indices)
	{
		using tetra_traits = Vcl::Geometry::CellTraits<TetrahedralCell<VertexId>>;

		// Face cache
		std::unordered_map<size_t, std::array<VertexId, 3>> face_lut;

		// Find all the tet-faces without a neighbour
		for (const auto& idx : indices)
		{
			for (int i = 0; i < 4; i++)
			{
				auto f = std::make_array(idx[tetra_traits::triFaces[i][0]], idx[tetra_traits::triFaces[i][1]], idx[tetra_traits::triFaces[i][2]]);
				auto h = makeFaceHash(f[0], f[1], f[2]);
				auto it = face_lut.find(h);
				if (it != face_lut.end())
				{
					face_lut.erase(it);
				}
				else
				{
					face_lut.insert(std::make_pair(h, f));
				}
			}
		}

		// All remaining faces in the list should represent surface triangles
		std::vector<std::array<VertexId, 3>> tri_adjs;
		tri_adjs.reserve(face_lut.size());
		std::transform(std::begin(face_lut), std::end(face_lut), std::back_inserter(tri_adjs), [](const auto& tri)
		{
			return tri.second;
		});

		return std::move(tri_adjs);
	}

	template<typename VertexId>
	std::vector<std::array<VertexId, 6>> convertToTriangleAdjacency(std::span<std::array<VertexId, 3>> triangles)
	{
		std::vector<std::array<VertexId, 6>> tri_adjs;
		tri_adjs.reserve(triangles.size());

		// Edge cache
		std::unordered_multimap<size_t, size_t> edge_face_lut;

		// Initialize the adjacencies by building a cache for the second pass.
		// For each edge it stores the third vertex
		std::transform(std::begin(triangles), std::end(triangles), std::back_inserter(tri_adjs), [&edge_face_lut](const auto& tri)
		{
			auto e0 = makeEdgeHash(tri[0], tri[1]);
			auto e1 = makeEdgeHash(tri[1], tri[2]);
			auto e2 = makeEdgeHash(tri[0], tri[2]);

			edge_face_lut.emplace(e0, tri[2].id());
			edge_face_lut.emplace(e1, tri[0].id());
			edge_face_lut.emplace(e2, tri[1].id());

			return std::make_array(tri[0], VertexId::InvalidId(), tri[1], VertexId::InvalidId(), tri[2], VertexId::InvalidId());
		});
		
		// Connect the neighbours to the central triangle
		size_t idx = 0;
		for (auto& tri_adj : tri_adjs)
		{
			size_t v[] = { tri_adj[4].id(), tri_adj[0].id(), tri_adj[2].id() };
			size_t e[] =
			{
				makeEdgeHash(tri_adj[0], tri_adj[2]),
				makeEdgeHash(tri_adj[2], tri_adj[4]),
				makeEdgeHash(tri_adj[0], tri_adj[4])
			};

			for (int i = 0; i < 3; i++)
			{
				size_t o = v[i];
				auto tri = edge_face_lut.equal_range(e[i]);
				auto v = std::find_if_not(tri.first, tri.second, [o](const auto v) { return v.second == o; });
				if (v != tri.second)
				{
					tri_adj[2*i + 1] = VertexId(v->second);
				}
			}

			idx++;
		}

		return std::move(tri_adjs);
	}

	template<typename VertexId>
	void computeNormals
	(
		const std::span<std::array<VertexId, 3>>& triangles,
		const std::span<Eigen::Vector3f>& points,
		std::span<Eigen::Vector3f> normals
	)
	{
		for (int idx = 0; idx < (int)triangles.size(); idx++)
		{
			Eigen::Vector3f p0 = points[triangles[idx][0].id()];
			Eigen::Vector3f p1 = points[triangles[idx][1].id()];
			Eigen::Vector3f p2 = points[triangles[idx][2].id()];

			// Compute the edges
			Eigen::Vector3f p0p1 = p1 - p0; float p0p1_l = p0p1.norm();
			Eigen::Vector3f p1p2 = p2 - p1; float p1p2_l = p1p2.norm();
			Eigen::Vector3f p2p0 = p0 - p2; float p2p0_l = p2p0.norm();

			// Normalize the edges
			p0p1 = p0p1_l > 1e-6f ? p0p1.normalized() : Eigen::Vector3f::Zero();
			p1p2 = p1p2_l > 1e-6f ? p1p2.normalized() : Eigen::Vector3f::Zero();
			p2p0 = p2p0_l > 1e-6f ? p2p0.normalized() : Eigen::Vector3f::Zero();

			// Compute the angles
			Eigen::Vector3f angles;

			// Use the dot product between edges: cos t = a.dot(b) / (a.length() * b.length())
			/*angle at v0 */ angles.x() = std::acos((-p2p0).dot(p0p1));
			/*angle at v1 */ angles.y() = std::acos((-p0p1).dot(p1p2));
			/*angle at v2 */ angles.z() = std::acos((-p1p2).dot(p2p0));

			// Compute the normalized face normal
			Eigen::Vector3f n = p0p1.cross(-p2p0);

			normals[triangles[idx][0].id()] += angles.x() * n;
			normals[triangles[idx][1].id()] += angles.y() * n;
			normals[triangles[idx][2].id()] += angles.z() * n;
		}
	}
}}
