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

// GSL
#include <span>

// VCL
#include <vcl/core/container/array.h>

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
	void extractSurface(gsl::span<std::array<VertexId, 4>> indices);

	template<typename VertexId>
	std::vector<std::array<VertexId, 6>> convertToTriangleAdjacency(gsl::span<std::array<VertexId, 3>> triangles)
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
}}
