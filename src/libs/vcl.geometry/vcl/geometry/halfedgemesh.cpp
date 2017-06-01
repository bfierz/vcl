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
#include <vcl/geometry/halfedgemesh.h>

namespace Vcl { namespace Geometry
{
	HalfEdgeMesh::HalfEdgeMesh()
	: _halfEdgeData("HalfEdgeGroup")
	{
		_halfEdges = _halfEdgeData.template add<HalfEdge>("HalfEdges");

		// Add position data
		_positions = addVertexProperty<Eigen::Vector2f>("Position", Eigen::Vector2f::Zero());
	}

	void HalfEdgeMesh::clear()
	{
		vertexProperties().clear();
		edgeProperties().clear();
	}

	void HalfEdgeMesh::addVertices(gsl::span<const Eigen::Vector2f> vertices)
	{
		const size_t curr_size = vertexProperties().propertySize();
		vertexProperties().resizeProperties(curr_size + vertices.size());

		for (size_t i = 0; i < vertices.size(); ++i)
		{
			_positions[curr_size + i] = vertices[i];
		}
	}

	HalfEdgeMesh::EdgeId HalfEdgeMesh::addEdge(const std::array<VertexId, 2>& vertices)
	{
		edgeProperties().resizeProperties(edgeProperties().propertySize() + 1);
		
		EdgeId new_edge_id{ nrEdges() - 1 };
		Edge& e = edge(new_edge_id);

		_halfEdgeData.resizeProperties(_halfEdgeData.propertySize() + 2);
		HalfEdgeId he_id{ static_cast<unsigned int>(_halfEdgeData.propertySize() - 2) };
		HalfEdge& he{ _halfEdges[he_id] };

		HalfEdgeId op_he_id{ static_cast<unsigned int>(_halfEdgeData.propertySize() - 1) };
		HalfEdge& op_he{ _halfEdges[op_he_id] };

		e.HalfEdge = he_id;
		he.Edge = new_edge_id;
		op_he.Edge = new_edge_id;

		he.Vertex = vertices[0];
		op_he.Vertex = vertices[1];

		he.Twin = op_he_id;
		op_he.Twin = he_id;

		// Connect to existing edges
		insertHalfEdge(he_id);
		insertHalfEdge(op_he_id);

		return new_edge_id;
	}

	void HalfEdgeMesh::removeEdge(HalfEdgeId he_id)
	{
		removeEdge(halfEdge(he_id).Edge);
	}

	void HalfEdgeMesh::removeEdge(EdgeId edge_id)
	{
		auto& edge = this->edge(edge_id);
		removeHalfEdge(edge.HalfEdge);
		removeHalfEdge(halfEdge(edge.HalfEdge).Twin);

		// Cleanup
		edge.HalfEdge = {};
	}

	HalfEdgeMesh::EdgeId HalfEdgeMesh::findEdge(const std::array<VertexId, 2>& vertices)
	{
		HalfEdgeId start_he_id = vertex(vertices[0]).HalfEdge;
		HalfEdgeId curr_he_id = start_he_id;
		do
		{
			const auto& he = halfEdge(halfEdge(curr_he_id).Twin);
			if (he.Vertex == vertices[1])
			{
				return he.Edge;
			}
			curr_he_id = he.Next;
		} while (curr_he_id != start_he_id);

		return{};
	}

	float HalfEdgeMesh::orientation(HalfEdgeId e)
	{
		const auto& he = _halfEdges[e];
		const auto& from = _positions[he.Vertex];
		const auto& to = _positions[_halfEdges[he.Twin].Vertex];

		return atan2(to.y() - from.y(), to.x() - from.x());
	}

	void HalfEdgeMesh::insertHalfEdge(HalfEdgeId he_id)
	{
		auto& he = halfEdge(he_id);
		auto& twin_he = halfEdge(he.Twin);

		auto& v0 = vertex(he.Vertex);
		if (!v0.HalfEdge.isValid())
		{
			v0.HalfEdge = he_id;

			he.Prev = he.Twin;
			twin_he.Next = he_id;
		}
		else
		{
			HalfEdgeId curr_he_id = v0.HalfEdge;
			HalfEdgeId prev_twin_id = halfEdge(halfEdge(curr_he_id).Prev).Twin;
			const float ref_a = orientation(he_id);
			float curr_a = orientation(curr_he_id);
			float next_a;

			// Circulate backwards through the half-edges and find the closest
			// to the edge to insert (last one before crossing 'he_id')
			while (prev_twin_id != v0.HalfEdge)
			{
				next_a = orientation(prev_twin_id);
				if (((curr_a <  next_a) &&  (curr_a <= ref_a) && (ref_a <= next_a)) || // regular case
					((curr_a >= next_a) && ((curr_a <= ref_a) || (ref_a <= next_a))))  // for discont 2*Pi->0
				{
					break;
				}

				curr_he_id = prev_twin_id;
				prev_twin_id = halfEdge(halfEdge(curr_he_id).Prev).Twin;
				curr_a = next_a;
			}

			// Insert the new half-edge into the half-edge chain
			// closest to the new half-edge
			auto& curr_he = halfEdge(curr_he_id);
			he.Prev = curr_he.Prev;
			twin_he.Next = curr_he_id;

			// Update the existing half-edges
			halfEdge(curr_he.Prev).Next = he_id;
			curr_he.Prev = he.Twin;
		}
	}
	
	void HalfEdgeMesh::removeHalfEdge(HalfEdgeId he_id)
	{
		auto& he = halfEdge(he_id);
		VertexId to_v_id = he.Vertex;
		auto& from_v = vertex(to_v_id);

		// The origin vertex of the half-edge is not connected to anything else
		if ((from_v.HalfEdge == he_id) && (halfEdge(he.Prev).Twin == he_id))
		{
			from_v.HalfEdge = {};
		}
		else
		{
			// If the origin vertex points to this half-edge, reconnect it
			if (from_v.HalfEdge == he_id)
				from_v.HalfEdge = halfEdge(he.Prev).Twin;

			// Remove the half-edge from the chain
			halfEdge(halfEdge(he.Twin).Next).Prev = he.Prev;
			halfEdge(he.Prev).Next = halfEdge(he.Twin).Next;
		}
	}
}}
