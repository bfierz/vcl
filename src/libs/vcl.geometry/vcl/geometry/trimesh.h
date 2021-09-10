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
	class TriMesh;

	template<>
	struct IndexDescriptionTrait<TriMesh>
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

		//! Index data of a single triangle
		using Face = std::array<VertexId, 3>;
	};

	class TriMesh : public SimplexLevel2<TriMesh>, public SimplexLevel0<TriMesh>
	{
	public: // Default constructors
		TriMesh() = default;
		TriMesh(const TriMesh& rhs) = default;
		TriMesh(TriMesh&& rhs) = default;
		virtual ~TriMesh() = default;

	public:
		TriMesh& operator=(const TriMesh& rhs) = default;
		TriMesh& operator=(TriMesh&& rhs) = default;

	public: // Construct meshes from data
		TriMesh(const std::vector<IndexDescriptionTrait<TriMesh>::Vertex>& vertices, const std::vector<std::array<IndexDescriptionTrait<TriMesh>::IndexType, 3>>& faces);

	public:
		//! Clear the content of the mesh
		void clear();

		//! Add a new property to the vertex level
		template<typename T>
		PropertyPtr<T, IndexDescriptionTrait<TriMesh>::VertexId> addVertexProperty(
			const std::string& name,
			typename Property<T, IndexDescriptionTrait<TriMesh>::VertexId>::reference init_value)
		{
			return vertexProperties().add<T>(name, init_value);
		}

		//! Add a new property to the vertex level
		template<typename T>
		PropertyPtr<T, IndexDescriptionTrait<TriMesh>::VertexId> addVertexProperty(
			const std::string& name,
			typename Property<T, IndexDescriptionTrait<TriMesh>::VertexId>::rvalue_reference init_value)
		{
			return vertexProperties().add<T>(name, std::move(init_value));
		}

		//! Add a new property to the face level
		template<typename T>
		PropertyPtr<T, IndexDescriptionTrait<TriMesh>::FaceId> addFaceProperty(
			const std::string& name,
			typename Property<T, IndexDescriptionTrait<TriMesh>::FaceId>::reference init_value)
		{
			return faceProperties().add<T>(name, init_value);
		}

		//! Add a new property to the face level
		template<typename T>
		PropertyPtr<T, IndexDescriptionTrait<TriMesh>::FaceId> addFaceProperty(
			const std::string& name,
			typename Property<T, IndexDescriptionTrait<TriMesh>::FaceId>::rvalue_reference init_value)
		{
			return faceProperties().add<T>(name, std::move(init_value));
		}

		//! Access generic vertex properties
		template<typename T>
		ConstPropertyPtr<T, IndexDescriptionTrait<TriMesh>::VertexId> vertexProperty(const std::string& name) const
		{
			return vertexProperties().property<T>(name);
		}

		//! Access generic vertex properties
		template<typename T>
		PropertyPtr<T, IndexDescriptionTrait<TriMesh>::VertexId> vertexProperty(const std::string& name)
		{
			return vertexProperties().property<T>(name);
		}

		//! Access generic face properties
		template<typename T>
		ConstPropertyPtr<T, IndexDescriptionTrait<TriMesh>::FaceId> faceProperty(const std::string& name) const
		{
			return faceProperties().property<T>(name);
		}

		//! Access generic face properties
		template<typename T>
		PropertyPtr<T, IndexDescriptionTrait<TriMesh>::FaceId> faceProperty(const std::string& name)
		{
			return faceProperties().property<T>(name);
		}
	};
}}
