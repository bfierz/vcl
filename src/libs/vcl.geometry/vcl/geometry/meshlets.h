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

// C++ standard library
#include <cstdint>
#include <vector>

// VCL
#include <vcl/core/span.h>

namespace Vcl { namespace Geometry {

	//! Algorithm used to partition a triangle list into meshlets
	enum class MeshletGenerator
	{
		Scan,             //!< Greedily scan the index buffer in order
		SpatialClustering //!< Cluster spatially close triangles, maximising vertex reuse
	};

	//! Ranges of one meshlet in the arrays of a MeshletMesh.
	//! Layout according to https://developer.nvidia.com/blog/introduction-turing-mesh-shaders
	struct Meshlet
	{
		uint32_t VertexCount;     //!< Number of vertices used
		uint32_t PrimitiveCount;  //!< Number of primitives (triangles) used
		uint32_t VertexOffset;    //!< First entry in MeshletMesh::VertexIndices
		uint32_t PrimitiveOffset; //!< First entry in MeshletMesh::Primitives
	};

	//! Triangle list partitioned into meshlets
	struct MeshletMesh
	{
		//! Meshlets, each referring to a range of VertexIndices and Primitives
		std::vector<Meshlet> Meshlets;

		//! Indices into the original vertex array
		std::vector<uint32_t> VertexIndices;

		//! Three meshlet-local vertex indices per triangle, relative to Meshlet::VertexOffset
		std::vector<uint8_t> Primitives;
	};

	//! Partition a triangle list into meshlets of at most 64 vertices and 124 triangles
	//! \param generator_type Algorithm partitioning the triangles
	//! \param vertices Vertex positions
	//! \param indices Three vertex indices per triangle
	MeshletMesh generateMeshlets(MeshletGenerator generator_type, stdext::span<const Eigen::Vector3f> vertices, stdext::span<const uint32_t> indices);
}}
