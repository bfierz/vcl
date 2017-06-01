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
	class HalfEdgeTriMesh;

	template<>
	struct IndexDescriptionTrait<HalfEdgeTriMesh>
	{
	public: // Idx Type
		using IndexType = unsigned int;

	public: // IDs
		VCL_CREATEID(VertexId, IndexType);   // Size: n0
		VCL_CREATEID(EdgeId, IndexType);     // Size: n1
		VCL_CREATEID(FaceId, IndexType);     // Size: n2

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

		struct FaceMetaData
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
			VertexId Vertex;
			HalfEdgeId Opposite;
			HalfEdgeId Next;

			EdgeId Edge;
		};

		class Edge
		{
		public:
			Edge() {}
			Edge(HalfEdgeId he_id) : HalfEdge(he_id) {}
		public:
			HalfEdgeId HalfEdge;
		};

		//! \brief Abstraction of a face
		//! Storage is done in a way so that all three half-edges that make up
		//! a face are stored together.
		//! A half-edge is built from the starting vertex and the opposite half-edge.
		//! Next and prev half-edge are found implicitly.
		class Face
		{
		public:
			Face() {}
			Face(VertexId a, VertexId b, VertexId c)
			: Vertices{ a, b, c }
			{}
		public:
			//! Vertices of the face (also starting and denoting the half-edges)
			std::array<VertexId, 3> Vertices;

			//! Opposite half-edges 
			std::array<HalfEdgeId, 3> OppositeHalfEdges;

			//! Edges linked to the half-edges
			//! Opposite half-edges 
			std::array<EdgeId, 3> Edges;
		};
	};

	class HalfEdgeTriMesh : public SimplexLevel2<HalfEdgeTriMesh>, public SimplexLevel1<HalfEdgeTriMesh>, public SimplexLevel0<HalfEdgeTriMesh>
	{
	public: 
		using VertexId = VertexId;

	public: // Default constructors
		HalfEdgeTriMesh();
		HalfEdgeTriMesh(const HalfEdgeTriMesh& rhs) = default;
		HalfEdgeTriMesh(HalfEdgeTriMesh&& rhs) = default;
		virtual ~HalfEdgeTriMesh() = default;

	public:
		HalfEdgeTriMesh& operator= (const HalfEdgeTriMesh& rhs) = default;
		HalfEdgeTriMesh& operator= (HalfEdgeTriMesh&& rhs) = default;

	public: // Construct meshes from data
		HalfEdgeTriMesh(const std::vector<Eigen::Vector3f>& vertices, const std::vector<std::array<IndexDescriptionTrait<HalfEdgeTriMesh>::IndexType, 3>>& faces);

	public:
		//! Clear the content of the mesh
		void clear();

		//! Add vertices to the mesh
		//! @param vertices Vertices to add to the mesh
		void addVertices(gsl::span<const Eigen::Vector3f> vertices);

		//! Add face to the mesh
		//! @param face Vertex indices denoting a face
		void addFace(const std::array<IndexDescriptionTrait<HalfEdgeTriMesh>::IndexType, 3> face);

	public:
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

		//! Add a new property to the face level
		template<typename T>
		Property<T, IndexDescriptionTrait<HalfEdgeTriMesh>::FaceId>* addFaceProperty
		(
			const std::string& name,
			typename Property<T, IndexDescriptionTrait<HalfEdgeTriMesh>::FaceId>::reference init_value
		)
		{
			return faceProperties().add<T>(name, init_value);
		}

	private:
		void buildIndex();
		void connectHalfEdges();

	private:
		PropertyPtr<Eigen::Vector3f, VertexId> _positions;
	};
}}
