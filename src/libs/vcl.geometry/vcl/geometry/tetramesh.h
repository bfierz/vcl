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

namespace Vcl { namespace Geometry
{
	class TetraMesh;

	template<>
	struct IndexDescriptionTrait<TetraMesh>
	{
	public: // Idx Type
		using IndexType = unsigned int;

	public: // IDs
		VCL_CREATEID(VertexId, IndexType);	// Size: n0
		VCL_CREATEID(VolumeId, IndexType);	// Size: n3

	public: // Basic types
		struct VertexMetaData
		{
			bool isValid() const { return State.isSet(ElementState::Deleted) == false; }

			Flags<ElementState> State;
		};

		struct VolumeMetaData
		{
			bool isValid() const { return State.isSet(ElementState::Deleted) == false; }

			Flags<ElementState> State;
		};

		//! Position data of a single vertex
		using Vertex = Eigen::Vector3f;

		//! Index data of a single tetrahedron
		using Volume = std::array<VertexId, 4>;
	};

	class TetraMesh : public SimplexLevel3<TetraMesh>, public SimplexLevel0<TetraMesh>
	{
	public: // Default constructors
		TetraMesh() = default;
		TetraMesh(const TetraMesh& rhs) = default;
		TetraMesh(TetraMesh&& rhs) = default;
		virtual ~TetraMesh() = default;

	public:
		TetraMesh& operator= (const TetraMesh& rhs) = default;
		TetraMesh& operator= (TetraMesh&& rhs) = default;

	public: // Construct meshes from data
		TetraMesh(const std::vector<IndexDescriptionTrait<TetraMesh>::Vertex>& vertices, const std::vector<std::array<IndexDescriptionTrait<TetraMesh>::IndexType, 4>>& volumes);

	public:
		//! Clear the content of the mesh
		void clear();

		//! Add a new property to the volume level
		template<typename T>
		Property<T, IndexDescriptionTrait<TetraMesh>::VolumeId>* addVolumeProperty
		(
			const std::string& name,
			typename Property<T, IndexDescriptionTrait<TetraMesh>::VolumeId>::reference init_value
		)
		{
			return volumeProperties().add<T>(name, init_value);
		}

	private: // Embedders
	};
}}
