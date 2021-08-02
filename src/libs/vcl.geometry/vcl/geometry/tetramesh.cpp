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
#include <vcl/geometry/tetramesh.h>

 // VCL
#include <vcl/geometry/meshoperations.h>

namespace Vcl { namespace Geometry {
	TetraMesh::TetraMesh(const std::vector<IndexDescriptionTrait<TetraMesh>::Vertex>& vertices, const std::vector<std::array<IndexDescriptionTrait<TetraMesh>::IndexType, 4>>& volumes)
	: _surfaceData("SurfaceGroup")
	{
		volumeProperties().resizeProperties(volumes.size());
		vertexProperties().resizeProperties(vertices.size());

		for (size_t i = 0; i < vertices.size(); ++i)
		{
			_vertices[i] = vertices[i];
		}

		for (size_t i = 0; i < volumes.size(); ++i)
		{
			Volume v
			{
				VertexId{ volumes[i][0] },
				VertexId{ volumes[i][1] },
				VertexId{ volumes[i][2] },
				VertexId{ volumes[i][3] }
			};
			_volumes[i] = v;
		}

		// Allocated buffer for the surface
		_surfaceFaces = _surfaceData.add<SurfaceFace>("SurfaceFaces");
	}

	TetraMesh::TetraMesh(const TetraMesh& rhs)
	: SimplexLevel3(rhs)
	, SimplexLevel0(rhs)
	, _surfaceData(rhs._surfaceData)
	{
		_surfaceFaces = _surfaceData.template property<SurfaceFace>("SurfaceFaces");
	}

	TetraMesh::TetraMesh(TetraMesh&& rhs)
	: SimplexLevel3(std::move(rhs))
	, SimplexLevel0(std::move(rhs))
	, _surfaceData(std::move(rhs._surfaceData))
	{
		rhs._surfaceFaces = nullptr;
		_surfaceFaces = _surfaceData.template property<SurfaceFace>("SurfaceFaces");
	}

	void TetraMesh::clear()
	{
		volumeProperties().clear();
		vertexProperties().clear();

		// Clear all the surface properties
		_surfaceData.clear();
	}

	void TetraMesh::recomputeSurface()
	{
		auto surface = extractSurface<IndexDescriptionTrait<TetraMesh>::VertexId>({ volumes()->data(), nrVolumes() });
		_surfaceData.resizeProperties(surface.size());

		for (size_t i = 0; i < surface.size(); i++)
		{
			_surfaceFaces[i] = surface[i];
		}
	}
}}
