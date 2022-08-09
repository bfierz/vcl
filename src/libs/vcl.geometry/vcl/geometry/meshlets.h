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
#pragma once

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

// VCL
#include <vcl/core/span.h>

namespace Vcl { namespace Geometry {

	/// Type of meshlet generation
	enum class MeshletGenerator
	{
		Scan,             ///< Generate meshlets by greedly scanning the index buffer
		SpatialClustering ///< Generate meshlets by spatially clustering vertices and optimizing vertex reuse
	};

	/// Bounds of an individual meshlet.
	/// Layout according to https://developer.nvidia.com/blog/introduction-turing-mesh-shaders
	struct Meshlet
	{
		uint32_t VertexCount;     ///< Number of vertices used
		uint32_t PrimitiveCount;  ///< Number of primitives (e.g. triangles) used
		uint32_t VertexOffset;    ///< Offset into vertexIndices
		uint32_t PrimitiveOffset; ///< Offset into primitiveIndices
	};

	struct MeshletMesh
	{
		std::vector<Meshlet> Meshlets;
		std::vector<uint32_t> VertexIndices;
		std::vector<uint8_t> Primitives;
	};

	MeshletMesh generateMeshlets(MeshletGenerator generator_type, stdext::span<Eigen::Vector3f> vertices, stdext::span<uint32_t> indices);
}}
