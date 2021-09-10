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
#include <vcl/config/eigen.h>

// C++ standard library
#include <array>
#include <set>
#include <unordered_map>
#include <vector>

// VCL
#include <vcl/core/flags.h>
#include <vcl/geometry/cell.h>
#include <vcl/geometry/genericid.h>
#include <vcl/geometry/property.h>
#include <vcl/geometry/propertygroup.h>
#include <vcl/geometry/simplex.h>

namespace Vcl { namespace Geometry {
	class MultiIndexTriMesh;

	template<>
	struct IndexDescriptionTrait<MultiIndexTriMesh>
	{
	public: // Idx Type
		using IndexType = unsigned int;

	public:                                // IDs
		VCL_CREATEID(VertexId, IndexType); // Size: n0
		VCL_CREATEID(FaceId, IndexType);   // Size: n2

	public: // Basic types
		struct VertexMetaData
		{
			//bool isValid() const { return State.isSet(ElementState::Deleted) == false; }

			//Flags<ElementState> State;
		};

		struct FaceMetaData
		{
			//bool isValid() const { return State.isSet(ElementState::Deleted) == false; }

			//Flags<ElementState> State;
		};

		//! Position data of a single vertex
		using Vertex = Eigen::Vector3f;

		//! Index data of a single tetrahedron
		using Face = std::array<VertexId, 3>;

		//! Index data for dependent mesh levels
		using DependentFace = std::array<IndexType, 3>;
	};

	class MultiIndexTriMesh : public SimplexLevel2<MultiIndexTriMesh>, SimplexLevel0<MultiIndexTriMesh>
	{
	public:
		using DependentFace = IndexDescriptionTrait<MultiIndexTriMesh>::DependentFace;
		using DependentVertedId = IndexDescriptionTrait<MultiIndexTriMesh>::IndexType;

	public: // Default constructors
		MultiIndexTriMesh() = default;
		MultiIndexTriMesh(const MultiIndexTriMesh& rhs) = default;
		MultiIndexTriMesh(MultiIndexTriMesh&& rhs) = default;
		virtual ~MultiIndexTriMesh() = default;

	public:
		MultiIndexTriMesh& operator=(const MultiIndexTriMesh& rhs) = default;
		MultiIndexTriMesh& operator=(MultiIndexTriMesh&& rhs) = default;

	public: // Construct meshes from data
		MultiIndexTriMesh(const std::vector<IndexDescriptionTrait<MultiIndexTriMesh>::Vertex>& vertices, const std::vector<IndexDescriptionTrait<MultiIndexTriMesh>::Face>& faces);

	public:
		//! Clear the content of the mesh
		void clear();

		//! Add a new property to the volume level
		template<typename T>
		Property<T, IndexDescriptionTrait<MultiIndexTriMesh>::FaceId>* addFaceProperty(
			const std::string& name,
			typename Property<T, IndexDescriptionTrait<MultiIndexTriMesh>::FaceId>::reference init_value)
		{
			faceProperties().add<T>(name, init_value);
		}

	private: // Additional layers
		std::vector<PropertyPtr<DependentFace, FaceId>> _additionalFaceLayers;
		std::vector<PropertyGroup<DependentVertedId>> _additionalVertexLayers;
	};
}}
