/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2022 Basil Fierz
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
#include <vcl/geometry/meshlets.h>

// C++ standard library
#include <algorithm>
#include <iterator>

// MeshOptimizer
#include <meshoptimizer.h>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Geometry {

	MeshletMesh generateMeshlets(MeshletGenerator generator_type, stdext::span<const Eigen::Vector3f> vertices, stdext::span<const uint32_t> indices)
	{
		VclRequire(indices.size() % 3 == 0, "Indices form a triangle list");

		// Recommended constants: https://developer.nvidia.com/blog/introduction-turing-mesh-shaders/
		constexpr size_t max_vertices = 64;
		constexpr size_t max_triangles = 124; // NVIDIA uses 126, however meshoptimizer requires a multiple of 4
		constexpr float cone_weight = 0.0f;

		if (indices.empty())
			return {};

		const size_t max_meshlets = meshopt_buildMeshletsBound(indices.size(), max_vertices, max_triangles);
		std::vector<meshopt_Meshlet> meshlets(max_meshlets);
		std::vector<unsigned int> meshlet_vertices(max_meshlets * max_vertices);
		std::vector<unsigned char> meshlet_triangles(max_meshlets * max_triangles * 3);

		size_t meshlet_count = 0;
		switch (generator_type)
		{
		case MeshletGenerator::Scan:
			meshlet_count = meshopt_buildMeshletsScan(
				meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(),
				indices.data(), indices.size(), vertices.size(), max_vertices, max_triangles);
			break;
		case MeshletGenerator::SpatialClustering:
			meshlet_count = meshopt_buildMeshlets(
				meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(),
				indices.data(), indices.size(), vertices.data()->data(), vertices.size(), sizeof(Eigen::Vector3f),
				max_vertices, max_triangles, cone_weight);
			break;
		}

		const meshopt_Meshlet& last = meshlets[meshlet_count - 1];
		meshlet_vertices.resize(last.vertex_offset + last.vertex_count);
		meshlet_triangles.resize(last.triangle_offset + last.triangle_count * 3);
		meshlets.resize(meshlet_count);

		std::vector<Meshlet> meshlets_out;
		meshlets_out.reserve(meshlets.size());
		std::transform(std::begin(meshlets), std::end(meshlets), std::back_inserter(meshlets_out), [](const meshopt_Meshlet& meshlet) -> Meshlet {
			return { meshlet.vertex_count, meshlet.triangle_count, meshlet.vertex_offset, meshlet.triangle_offset };
		});

		return { std::move(meshlets_out), std::move(meshlet_vertices), std::move(meshlet_triangles) };
	}
}}
