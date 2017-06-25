/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// GSL
#include <gsl/span>

// VCL
#include <vcl/core/contract.h>
#include <vcl/core/flags.h>
#include <vcl/geometry/cell.h>
#include <vcl/geometry/genericid.h>
#include <vcl/geometry/property.h>
#include <vcl/geometry/propertygroup.h>
#include <vcl/geometry/simplex.h>

namespace Vcl { namespace Geometry
{
	class HalfFaceTetraMesh;

	template<>
	struct IndexDescriptionTrait<HalfFaceTetraMesh>
	{
	public: // Idx Type
		using IndexType = unsigned int;

	public: // IDs
		#include "halffacetetramesh_id.inc"
		
	public: // Basic types
		#include "halffacetetramesh_structures.inc"

		struct VertexMetaData
		{
			//bool isValid() const { return State.isSet(ElementState::Deleted) == false; }

			//Flags<ElementState> State;
		};

		struct EdgeMetaData
		{
			//bool isValid() const { return State.isSet(ElementState::Deleted) == false; }

			//Flags<ElementState> State;
		};

		struct FaceMetaData
		{
			//bool isValid() const { return State.isSet(ElementState::Deleted) == false; }

			//Flags<ElementState> State;
		};

		struct VolumeMetaData
		{
			//bool isValid() const { return State.isSet(ElementState::Deleted) == false; }

			//Flags<ElementState> State;
		};

		struct HalfEdgeMetaData
		{
			bool isValid() const { return true; }
			//bool isValid() const { return State.isSet(ElementState::Deleted) == false; }

			//Flags<ElementState> State;
		};

		struct HalfFaceMetaData
		{
			bool isValid() const { return true; }
			//bool isValid() const { return State.isSet(ElementState::Deleted) == false; }

			//Flags<ElementState> State;
		};
	};

	class HalfFaceTetraMesh :
		public SimplexLevel3<HalfFaceTetraMesh>,
		public SimplexLevel2<HalfFaceTetraMesh>,
		public SimplexLevel1<HalfFaceTetraMesh>,
		public SimplexLevel0<HalfFaceTetraMesh>,
		public HalfSimplexLevel2<HalfFaceTetraMesh>,
		public HalfSimplexLevel1<HalfFaceTetraMesh>
	{
	public: // Types
		using IndexType = IndexDescriptionTrait<HalfFaceTetraMesh>::IndexType;
		using VertexId = IndexDescriptionTrait<HalfFaceTetraMesh>::VertexId;
		using EdgeId = IndexDescriptionTrait<HalfFaceTetraMesh>::EdgeId;
		using FaceId = IndexDescriptionTrait<HalfFaceTetraMesh>::FaceId;
		using VolumeId = IndexDescriptionTrait<HalfFaceTetraMesh>::VolumeId;

		using HalfEdgeId = IndexDescriptionTrait<HalfFaceTetraMesh>::HalfEdgeId;
		using HalfFaceId = IndexDescriptionTrait<HalfFaceTetraMesh>::HalfFaceId;

		using HalfEdge = IndexDescriptionTrait<HalfFaceTetraMesh>::HalfEdge;
		using HalfFace = IndexDescriptionTrait<HalfFaceTetraMesh>::HalfFace;

	public: // Aggregated types
		using HalfEdgeVertices = std::array<VertexId, 2>;
		using HalfFaceVertices = std::array<VertexId, 3>;
		using VolumeVertices   = std::array<VertexId, 4>;

	public: // Iterator Types
		using HalfFaceEnumerator = Enumerator<HalfFaceId, HalfFace, HalfFaceTetraMesh>;
		using HalfEdgeEnumerator = Enumerator<HalfEdgeId, HalfEdge, HalfFaceTetraMesh>;

	public: // Default constructors
		HalfFaceTetraMesh();
		HalfFaceTetraMesh(const HalfFaceTetraMesh& rhs) = default;
		HalfFaceTetraMesh(HalfFaceTetraMesh&& rhs) = default;
		virtual ~HalfFaceTetraMesh() = default;

	public:
		HalfFaceTetraMesh& operator= (const HalfFaceTetraMesh& rhs) = default;
		HalfFaceTetraMesh& operator= (HalfFaceTetraMesh&& rhs) = default;

	public: // Construct meshes from data
		HalfFaceTetraMesh(gsl::span<const Eigen::Vector3f> vertices, gsl::span<const std::array<IndexType, 4>> volumes);

	public:
		//! Clear the content of the mesh
		void clear();

		//! Add a new property to the volume level
		template<typename T>
		Property<T, VolumeId>* addVolumeProperty
		(
			const std::string& name,
			typename Property<T, VolumeId>::reference init_value
		)
		{
			return volumeProperties().add<T>(name, init_value);
		}

		//! Add a new property to the vertex level
		template<typename T>
		Property<T, VertexId>* addVertexProperty
		(
			const std::string& name,
			typename Property<T, VertexId>::reference init_value
		)
		{
			return vertexProperties().add<T>(name, init_value);
		}

	public: // Queries
		HalfFaceId opposite(HalfFaceId hf) const;
		HalfEdgeId radial(HalfEdgeId id) const;
		HalfEdgeVertices verticesFromHalfEdge(HalfEdgeId id) const;
		HalfFaceVertices verticesFromHalfFace(HalfFaceId id) const;

		HalfFaceId vertexToHalfFace(VolumeId volume_id, VertexId id) const;

		bool isSurfaceFace(HalfFaceId hf_id) const;
		bool shareFace(VolumeId a, VolumeId b) const;
		bool shareVertex(VolumeId a, VolumeId b) const;

		/// Check if a vertex belongs to a volume
		bool isAdjacient(VolumeId volume, VertexId vertex) const;

	public: // Manipulation
		void splitEdge(EdgeId edge_id);
		void collapseEdge(EdgeId edge_id);
		void deleteVolume(VolumeId vol_id);

	private: // Implementation
		//! Add additional vertices to the mesh
		void addVertices(gsl::span<const Eigen::Vector3f> vertices);

		//! Build up the half-face index structure
		void buildIndex();

		//! Build up the half-face lists
		void connectHalfFaces();

		//! Build up the half-edge lists
		void connectHalfEdges();

		void splitHalfEdgeId(HalfEdgeId he_id, HalfFaceId& hf_id, unsigned int& vhe) const;

		void splitVolume(VolumeId vol_id, EdgeId edge_id, VertexId new_vertex_id, std::map<HalfFaceVertices, HalfFaceId>& loose_half_faces);
		void collapseVolume(VolumeId vol_id, EdgeId edge_id);
		void replaceHalfFace(HalfFaceId old_hf, HalfFaceId new_hf);
		void replaceVertex(VolumeId vol_id, VertexId old_vertex_id, VertexId new_vertex_id);

	private: // Properties

		//! Position data
		PropertyPtr<Eigen::Vector3f, VertexId> _positions;

	private:
		std::set<HalfFaceId> _surfaceFaces;
	};

	inline HalfFaceTetraMesh::HalfFaceId HalfFaceTetraMesh::opposite(HalfFaceId hf) const
	{
		return _halfFaces[hf].Opposite;
	}

	inline HalfFaceTetraMesh::HalfEdgeId HalfFaceTetraMesh::radial(HalfEdgeId id) const
	{
		EdgeId edge = _halfEdges[id].Edge;

		// Opposite half face
		HalfFaceId opposite = _halfFaces[id.halfFace()].Opposite;
		if (opposite.isValid() == false) return HalfEdgeId::InvalidId();

		unsigned int half_edges = opposite.id() * 3;
		if (_halfEdges[half_edges].Edge == edge)
		{
			return HalfEdgeId(half_edges);
		}
		else if (_halfEdges[half_edges + 1].Edge == edge)
		{
			return HalfEdgeId(half_edges + 1);
		}
		else if (_halfEdges[half_edges + 2].Edge == edge)
		{
			return HalfEdgeId(half_edges + 2);
		}

		assert(false);
		return HalfEdgeId();
	}


}}
