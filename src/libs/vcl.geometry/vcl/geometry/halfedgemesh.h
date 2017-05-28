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

// GSL
#include <gsl/gsl>

// VCL
#include <vcl/core/flags.h>
#include <vcl/geometry/cell.h>
#include <vcl/geometry/genericid.h>
#include <vcl/geometry/property.h>
#include <vcl/geometry/propertygroup.h>
#include <vcl/geometry/simplex.h>

namespace Vcl { namespace Geometry
{
	class HalfEdgeMesh;

	template<>
	struct IndexDescriptionTrait<HalfEdgeMesh>
	{
	public: // Idx Type
		using IndexType = unsigned int;

	public: // IDs
		VCL_CREATEID(VertexId, IndexType);   // Size: n0
		VCL_CREATEID(EdgeId, IndexType);     // Size: n1

		VCL_CREATEID(HalfEdgeId, IndexType); // Size: n2 * 3

	public: // Basic types

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

		class Vertex
		{
		public:
			Vertex() {}
			Vertex(HalfEdgeId he_id) : HalfEdge(he_id) {}
		public:
			//! Sample half-edge connected to the vertex
			HalfEdgeId HalfEdge;
		};

		class HalfEdge
		{
		public:
			HalfEdge() {}
			HalfEdge(EdgeId e_id) : Edge(e_id) {}

		public:
			//! Vertex this half-edge is starting from
			VertexId Vertex;

			//! Twin of this half-edge
			HalfEdgeId Twin;

			//! Next half-edge in the polygon
			HalfEdgeId Next;

			//! Previous half-edge in the polygon
			HalfEdgeId Prev;

			// Edge connected this half-edge belongs to
			EdgeId Edge;
		};

		class Edge
		{
		public:
			Edge() {}
			Edge(HalfEdgeId he_id) : HalfEdge(he_id) {}
		public:
			//! One of the half-edges linked to this edge
			HalfEdgeId HalfEdge;
		};
	};

	class HalfEdgeMesh : public SimplexLevel1<HalfEdgeMesh>, public SimplexLevel0<HalfEdgeMesh>
	{
	public: 
		using VertexId = IndexDescriptionTrait<HalfEdgeMesh>::VertexId;
		using EdgeId = IndexDescriptionTrait<HalfEdgeMesh>::EdgeId;
		using HalfEdgeId = IndexDescriptionTrait<HalfEdgeMesh>::HalfEdgeId;

		using Vertex = IndexDescriptionTrait<HalfEdgeMesh>::Vertex;
		using HalfEdge = IndexDescriptionTrait<HalfEdgeMesh>::HalfEdge;
		using Edge = IndexDescriptionTrait<HalfEdgeMesh>::Edge;

	public: // Default constructors
		HalfEdgeMesh();
		HalfEdgeMesh(const HalfEdgeMesh& rhs) = default;
		HalfEdgeMesh(HalfEdgeMesh&& rhs) = default;
		virtual ~HalfEdgeMesh() = default;

	public:
		HalfEdgeMesh& operator= (const HalfEdgeMesh& rhs) = default;
		HalfEdgeMesh& operator= (HalfEdgeMesh&& rhs) = default;

	public:
		const HalfEdge& halfEdge(HalfEdgeId id) const { return _halfEdges[id]; }
		      HalfEdge& halfEdge(HalfEdgeId id)       { return _halfEdges[id]; }

	public:
		//! Clear the content of the mesh
		void clear();

		//! Add vertices to the mesh
		//! @param vertices Vertices to add to the mesh
		void addVertices(gsl::span<const Eigen::Vector2f> vertices);

		//! Add face to the mesh
		//! @param face Vertex indices denoting a face
		//! @returns The id of the new edge
		EdgeId addEdge(const std::array<VertexId, 2>& edge);

	public:
		//! Add a new property to the vertex level
		template<typename T>
		Property<T, VertexId>* addVertexProperty
		(
			const std::string& name,
			typename Property<T, VertexId>::const_reference init_value
		)
		{
			return vertexProperties().add<T>(name, init_value);
		}

		//! Add a new property to the vertex level
		template<typename T>
		Property<T, VertexId>* addEdgeProperty
		(
			const std::string& name,
			typename Property<T, VertexId>::const_reference init_value
		)
		{
			return edgeProperties().add<T>(name, init_value);
		}

	private: // Helpers
		float orientation(HalfEdgeId he);
		void insertHalfEdge(HalfEdgeId he);

	private: // Index structure
		//! Data associated with a half-edge
		PropertyGroup<HalfEdgeId> _halfEdgeData;

		//! Generic half-edge data
		PropertyPtr<HalfEdge, HalfEdgeId> _halfEdges;

	private: // Properties
		//! Position data
		PropertyPtr<Eigen::Vector2f, VertexId> _positions;
	};
}}
