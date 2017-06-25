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
#include <vcl/geometry/halffacetetramesh.h>

// C++ standard library
#include <map>

// VCL
#include <vcl/util/sort.h>

namespace Vcl { namespace Geometry
{
	HalfFaceTetraMesh::HalfFaceTetraMesh()
	{
		// Add position data
		_positions = addVertexProperty<Eigen::Vector3f>("Position", Eigen::Vector3f{ 0.0f, 0.0f, 0.0f });

		// Half-face index data
		_halfEdges = _halfEdgeData.add<HalfEdge>("HalfEdge");
	}

	HalfFaceTetraMesh::HalfFaceTetraMesh(gsl::span<const Eigen::Vector3f> vertices, gsl::span<const std::array<IndexDescriptionTrait<HalfFaceTetraMesh>::IndexType, 4>> volumes)
	: HalfFaceTetraMesh()
	{
		addVertices(vertices);

		volumeProperties().resizeProperties(volumes.size());
		for (size_t i = 0; i < volumes.size(); ++i)
		{
			Volume v
			{ {
				VertexId{ volumes[i][0] },
				VertexId{ volumes[i][1] },
				VertexId{ volumes[i][2] },
				VertexId{ volumes[i][3] }
			} };
			_volumes[i] = v;
		}

		buildIndex();
	}

	void HalfFaceTetraMesh::clear()
	{
		volumeProperties().clear();
		faceProperties().clear();
		edgeProperties().clear();
		vertexProperties().clear();

		halfFaceProperties().clear();
	}

	void HalfFaceTetraMesh::addVertices(gsl::span<const Eigen::Vector3f> vertices)
	{
		const size_t curr_size = vertexProperties().propertySize();
		vertexProperties().resizeProperties(curr_size + vertices.size());

		for (size_t i = 0; i < vertices.size(); ++i)
		{
			_positions[curr_size + i] = vertices[i];
		}
	}

	HalfFaceTetraMesh::HalfFaceId HalfFaceTetraMesh::vertexToHalfFace(VolumeId volume_id, VertexId id) const
	{
		const VolumeVertices& vertices = _volumes[volume_id.id()].Vertices;

		if (vertices[0] == id) return HalfFaceId((volume_id.id() << 2) + 0);
		if (vertices[1] == id) return HalfFaceId((volume_id.id() << 2) + 1);
		if (vertices[2] == id) return HalfFaceId((volume_id.id() << 2) + 2);
		if (vertices[3] == id) return HalfFaceId((volume_id.id() << 2) + 3);

		return HalfFaceId();
	}

	HalfFaceTetraMesh::HalfEdgeVertices HalfFaceTetraMesh::verticesFromHalfEdge(HalfEdgeId id) const
	{
		static int NextHalfEdge[4][4] =
		{ { -1,  3,  1,  2 },
		  {  2, -1,  3,  0 },
		  {  3,  0, -1,  1 },
		  {  1,  2,  0, -1 } };

		HalfFaceId hf_id;
		unsigned int begin = 0xffffffff;
		splitHalfEdgeId(id, hf_id, begin);

		unsigned int end = static_cast<unsigned int>(NextHalfEdge[begin][hf_id.id() % 4]);

		const VolumeVertices& vertices = _volumes[hf_id.volume().id()].Vertices;
		return { vertices[begin], vertices[end] };
	}

	HalfFaceTetraMesh::HalfFaceVertices HalfFaceTetraMesh::verticesFromHalfFace(HalfFaceId id) const
	{
		const VolumeVertices& vertices = _volumes[id.volume().id()].Vertices;

		switch (id.id() & 3)
		{
		case 0: { return{ vertices[1], vertices[2], vertices[3] }; }
		case 1: { return{ vertices[2], vertices[0], vertices[3] }; }
		case 2: { return{ vertices[3], vertices[0], vertices[1] }; }
		case 3: { return{ vertices[0], vertices[2], vertices[1] }; }
		}

		return HalfFaceVertices();
	}

	void HalfFaceTetraMesh::splitHalfEdgeId(HalfEdgeId he_id, HalfFaceId& hf_id, unsigned int& vhe) const
	{
		static int VertexHalfEdge[4][3] =
		{ { 1, 2, 3 },
		  { 2, 0, 3 },
		  { 3, 0, 1 },
		  { 0, 2, 1 } };


		hf_id = he_id.halfFace();
		vhe = VertexHalfEdge[hf_id.id() % 4][he_id.id() % 3];

		VclEnsure(hf_id.isValid(), "Half-face connected to half-edge is valid.");
		VclEnsure(vhe != 0xffffffff, "Vertex of half-edge exist.");
	}
	void HalfFaceTetraMesh::buildIndex()
	{
		connectHalfFaces();
		connectHalfEdges();
	}

	void HalfFaceTetraMesh::connectHalfFaces()
	{
		using face_map_t = std::map<HalfFaceVertices, HalfFaceId>;
		
		_halfFaceData.resizeProperties(4 * nrVolumes());

		for (size_t i = 0; i < _halfFaces->size(); ++i)
			_halfFaces[i].Opposite = {};

		face_map_t adjacency;
		for (HalfFaceId::IdType i = 0; i < static_cast<HalfFaceId::IdType>(_halfFaces->size()); i++)
		{
			HalfFaceVertices v = verticesFromHalfFace(HalfFaceId(i));
			Vcl::Util::sort(v);
			const face_map_t::key_type key(v);

			face_map_t::iterator item = adjacency.find(key);
			if (item != adjacency.end())
			{
				_halfFaces[i].Opposite = (*item).second;
				_halfFaces[_halfFaces[i].Opposite.id()].Opposite = HalfFaceId(i);
				adjacency.erase(item);

				_halfFaces[i].Face = _halfFaces[_halfFaces[i].Opposite.id()].Face;
			}
			else
			{
				adjacency[key] = HalfFaceId(i);
				_faces->push_back(Face(HalfFaceId(i)));
				_halfFaces[i].Face = FaceId(static_cast<FaceId::IdType>(_faces->size()) - 1);
			}
		}

		for (auto it = adjacency.begin(); it != adjacency.end(); ++it)
		{
			HalfFaceId id = (*it).second;

			// Consistency check. The adjacency map should by now only contain faces adjacent to one volume. Thus surface faces.
			VclCheck(_halfFaces[id.id()].Opposite.isValid() == false, "Half-face is surface face");

			_surfaceFaces.insert(id);
		}
	}

	void HalfFaceTetraMesh::connectHalfEdges()
	{
		using edge_map_t = std::map<HalfEdgeVertices, EdgeId>;

		_halfEdgeData.resizeProperties(12 * nrVolumes());

		for (size_t i = 0; i < _halfEdges->size(); ++i)
			_halfEdges[i].Edge = {};

		edge_map_t edge_map;
		for (HalfEdgeId::IdType i = 0; i < static_cast<HalfEdgeId::IdType>(_halfEdges->size()); i++)
		{
			HalfEdgeVertices v = verticesFromHalfEdge(HalfEdgeId(i));
			Vcl::Util::sort(v);
			const edge_map_t::key_type key(v);

			edge_map_t::iterator item = edge_map.find(key);
			if (item != edge_map.end())
			{
				_halfEdges[i].Edge = (*item).second;
			}
			else
			{
				_edges->push_back(Edge(HalfEdgeId(i)));

				EdgeId id(static_cast<EdgeId::IdType>(_edges->size()) - 1);
				edge_map[key] = id;
				_halfEdges[i].Edge = id;
			}
		}
	}
	
	bool HalfFaceTetraMesh::isSurfaceFace(HalfFaceId hf_id) const
	{
		return !(_halfFaces[hf_id].Opposite.isValid());
	}

	bool HalfFaceTetraMesh::shareFace(VolumeId a, VolumeId b) const
	{
		HalfFaceId cur = HalfFaceId(a.id() * 4);
		HalfFaceId end = cur;
		do
		{
			// Check if the volume 'b' is adjacient
			if (halfFace(cur).Opposite.volume() == b)
				return true;

			// Advance to the next half-face
			cur = cur.next();
		} while (cur != end);

		return false;
	}

	bool HalfFaceTetraMesh::shareVertex(VolumeId a, VolumeId b) const
	{
		const auto& va = _volumes[a].Vertices;
		const auto& vb = _volumes[b].Vertices;

		for (unsigned int i = 0; i < 4; i++)
		{
			VertexId id = va[i];
			if (vb[0] == id || vb[1] == id ||
				vb[2] == id || vb[3] == id)
				return true;
		}

		return false;
	}

	bool HalfFaceTetraMesh::isAdjacient(VolumeId volume, VertexId vertex) const
	{
		const auto& vertices = _volumes[volume].Vertices;

		if (vertices[0] == vertex ||
			vertices[1] == vertex ||
			vertices[2] == vertex ||
			vertices[3] == vertex)
			return true;

		return false;
	}


	void HalfFaceTetraMesh::splitEdge(EdgeId edge_id)
	{
		using namespace std;
		using face_map_t = std::map<HalfFaceVertices, HalfFaceId>;

		// List of old volumes
		std::list<VolumeId> old_volumes;

		// Map of unprocessed half faces
		face_map_t half_face_map;

		// Get the vertices of the edge.
		HalfEdgeVertices old_edge_vertices = verticesFromHalfEdge(_edges[edge_id.id()].HalfEdge);

		// Add a new vertex.
		VertexId new_vertex = addVertex();
		assert(false); //setPoint(new_vertex, 0.5 * (getPoint(get<0>(old_edge_vertices)) + getPoint(get<1>(old_edge_vertices))));

		Edge& edge = _edges[edge_id.id()];
		HalfEdgeId start_he = edge.HalfEdge;
		HalfFaceId start_hf = start_he.halfFace();

		// Check the tetrahedron adjacent to the edge is on the boundary.
		// If yes, use the boundary half edge as the starting point.
		if (isSurfaceFace(start_he.mate().halfFace()))
		{
			start_he = start_he.mate();
			start_hf = start_he.halfFace();
		}

		if (isSurfaceFace(start_hf))
		{
			// The half face is on the boundary.
			// Iterate until we reach the next half face on the boundary.
			HalfEdgeId he = start_he;
			do
			{
				VolumeId vol = he.volume();
				splitVolume(vol, edge_id, new_vertex, half_face_map);
				old_volumes.push_back(vol);

				he = radial(he.mate());
			} while (he.isValid() && !isSurfaceFace(he.halfFace()));
		}
		else
		{
			// Found an internal volume. Iterate through all volumes until the end is reached.
			// If the end is an external face, do a second iteration in the other direction.
			// Else the end is connected to the starting volume.
			HalfEdgeId he = start_he;
			do
			{
				VolumeId vol = he.volume();
				splitVolume(vol, edge_id, new_vertex, half_face_map);
				old_volumes.push_back(vol);

				he = radial(he.mate());
			} while (he.isValid() && !(isSurfaceFace(he.halfFace()) || (he.halfFace() == start_hf)));

			if (he.halfFace() != start_hf)
			{
				he = radial(start_he);
				do
				{
					VolumeId vol = he.volume();
					splitVolume(vol, edge_id, new_vertex, half_face_map);
					old_volumes.push_back(vol);

					he = radial(he.mate());
				} while (he.isValid() && !(isSurfaceFace(he.halfFace()) || (he.halfFace() == start_hf)));
			}
		}

		for (face_map_t::iterator it = half_face_map.begin(); it != half_face_map.end(); ++it)
		{
			HalfFaceId id = (*it).second;

			// Consistency check. The adjacency map should by now only contain faces adjacent to one volume. Thus surface faces.
			assert(_halfFaces[id.id()].Opposite.isValid() == false);

			_surfaceFaces[id] = id;
		}

		for (std::list<VolumeId>::iterator it = old_volumes.begin(); it != old_volumes.end(); ++it)
		{
			deleteVolume(*it);
		}
	}

	void HalfFaceTetraMesh::splitVolume(VolumeId vol_id, EdgeId edge_id, VertexId new_vertex_id, std::map<HalfFaceVertices, HalfFaceId>& loose_half_faces)
	{
		assert(vol_id.isValid());
		assert((_volumes[vol_id.id()].state().isValid()) == false);

		// Get the vertices of the volume
		VolumeVertices vertices = _volumes[vol_id.id()].Vertices;

		// Get the vertices of the edge.
		HalfEdgeVertices old_edge_vertices = verticesFromHalfEdge(_edges[edge_id.id()].HalfEdge);

		// Find the old vertices in the volume
		unsigned int index_in_volume[2] = { static_cast<unsigned int>(-1), static_cast<unsigned int>(-1) };
		for (unsigned int i = 0; i < 4; i++)
		{
			if (vertices[i] == old_edge_vertices[0]) index_in_volume[0] = i;
			if (vertices[i] == old_edge_vertices[1]) index_in_volume[1] = i;
		}

		// Created two volumes
		VolumeId new_volumes[2];
		VertexId new_volume_vertices_a[] = { vertices[0], vertices[1], vertices[2], vertices[3] };
		VertexId new_volume_vertices_b[] = { vertices[0], vertices[1], vertices[2], vertices[3] };

		new_volume_vertices_a[index_in_volume[0]] = new_vertex_id;
		new_volume_vertices_b[index_in_volume[1]] = new_vertex_id;

		new_volumes[0] = addVolume(new_volume_vertices_a);
		new_volumes[1] = addVolume(new_volume_vertices_b);

		// Create a new face between the split volumes
		_faces.push_back(Face());
		FaceId f_id = FaceId(static_cast<FaceId::IdType>(_faces.size()) - 1);
		Face& face = _faces[f_id.id()];

		HalfFaceId hf_a = HalfFaceId(new_volumes[0].id() * 4 + index_in_volume[1]);
		HalfFaceId hf_b = HalfFaceId(new_volumes[1].id() * 4 + index_in_volume[0]);

		face.HalfFace = hf_a;
		_halfFaces[hf_a.id()].Face = f_id;
		_halfFaces[hf_a.id()].Opposite = hf_b;
		_halfFaces[hf_b.id()].Face = f_id;
		_halfFaces[hf_b.id()].Opposite = hf_a;

		// Connect the half faces adjacent to the rest of the volume
		hf_a = HalfFaceId(vol_id.id() * 4 + index_in_volume[0]);
		hf_b = HalfFaceId(vol_id.id() * 4 + index_in_volume[1]);

		replaceHalfFace(hf_a, HalfFaceId(new_volumes[0].id() * 4 + index_in_volume[0]));
		replaceHalfFace(hf_b, HalfFaceId(new_volumes[1].id() * 4 + index_in_volume[1]));

		// Connect the remaining half faces
		for (unsigned int i = 0; i < 2; i++)
		{
			for (unsigned int j = 0; j < 4; j++)
			{
				HalfFaceId hf_id = HalfFaceId(new_volumes[i].id() * 4 + j);
				if (_halfFaces[hf_id.id()].Opposite.isValid()) continue;

				// Check if there are already half faces in the list
				// If so connect them, else create new ones
				typedef std::map<HalfFaceVertices, HalfFaceId> face_map_t;

				HalfFaceVertices v012 = verticesFromHalfFace(hf_id);
				Vcl::Util::Sort(v012);
				face_map_t::key_type key = v012;

				face_map_t::iterator item = loose_half_faces.find(key);
				if (item != loose_half_faces.end())
				{
					_halfFaces[(*item).second.id()].Opposite = hf_id;
					_halfFaces[hf_id.id()].Face = _halfFaces[(*item).second.id()].Face;
					_halfFaces[hf_id.id()].Opposite = (*item).second;

					loose_half_faces.erase(item);
				}
				else
				{
					FaceId f_id = FaceId(static_cast<FaceId::IdType>(_faces.size()));
					Face f;
					f.HalfFace = hf_id;
					_faces.push_back(f);

					_halfFaces[hf_id.id()].Face = f_id;
					loose_half_faces[key] = hf_id;
				}
			}
		}
	}

	void HalfFaceTetraMesh::replaceHalfFace(HalfFaceId old_hf, HalfFaceId new_hf)
	{
		assert(old_hf.isValid());
		assert(new_hf.isValid());

#ifdef DEBUG
		// Sanity check
		HalfFaceVertices ohfv = VerticesFromHalfFace(old_hf);
		HalfFaceVertices nhfv = VerticesFromHalfFace(new_hf);

		Vcl::Util::Sort(ohfv[0], ohfv[1], ohfv[2]);
		Vcl::Util::Sort(nhfv[0], nhfv[1], nhfv[2]);

		bool b1 = ohfv[0] == nhfv[0];
		bool b2 = ohfv[1] == nhfv[1];
		bool b3 = ohfv[2] == nhfv[2];

		if ((b1 && b2 && b3) == false)
		{
			std::cerr << "ERROR: HalfFace " << old_hf.id() << " and HalfFace " << new_hf.id() << " don't share the same vertices." << std::endl;
		}
#endif /* DEBUG */

		HalfFaceId opposite = _halfFaces[old_hf.id()].Opposite;
		if (opposite.isValid())
		{
			_halfFaces[opposite.id()].Opposite = new_hf;
			_halfFaces[new_hf.id()].Opposite = opposite;
		}

		// Replace the half edges
		HalfEdgeId he1 = HalfEdgeId(old_hf.id() * 3 + 0);
		HalfEdgeId he2 = HalfEdgeId(old_hf.id() * 3 + 1);
		HalfEdgeId he3 = HalfEdgeId(old_hf.id() * 3 + 2);

		EdgeId e1 = _halfEdges[he1.id()].Edge;
		EdgeId e2 = _halfEdges[he2.id()].Edge;
		EdgeId e3 = _halfEdges[he3.id()].Edge;

		HalfEdgeId nhe1 = HalfEdgeId(new_hf.id() * 3 + 0);
		HalfEdgeId nhe2 = HalfEdgeId(new_hf.id() * 3 + 1);
		HalfEdgeId nhe3 = HalfEdgeId(new_hf.id() * 3 + 2);

		_halfEdges[nhe1.id()].Edge = e1;
		_halfEdges[nhe2.id()].Edge = e2;
		_halfEdges[nhe3.id()].Edge = e3;

		if (_edges[e1.id()].HalfEdge == he1) _edges[e1.id()].HalfEdge = nhe1;
		if (_edges[e2.id()].HalfEdge == he2) _edges[e2.id()].HalfEdge = nhe2;
		if (_edges[e3.id()].HalfEdge == he3) _edges[e3.id()].HalfEdge = nhe3;

		HalfEdgeVertices hev1 = verticesFromHalfEdge(he1);
		HalfEdgeVertices nhev1 = verticesFromHalfEdge(nhe1);

		HalfEdgeVertices hev2 = verticesFromHalfEdge(he2);
		HalfEdgeVertices nhev2 = verticesFromHalfEdge(nhe2);

		HalfEdgeVertices hev3 = verticesFromHalfEdge(he3);
		HalfEdgeVertices nhev4 = verticesFromHalfEdge(nhe3);
	}

	void HalfFaceTetraMesh::deleteVolume(VolumeId vol_id)
	{
		// Delete edges and the unused connections
		for (unsigned int i = 0; i < 12; i++)
		{
			HalfEdgeId he_id = HalfEdgeId(vol_id.id() * 12 + i);
			_halfEdges[he_id.id()].state().setFlag(StateFlags::Deleted);
		}

		// Delete faces and the unused connections
		for (unsigned int i = 0; i < 4; i++)
		{
			HalfFaceId hf_id = HalfFaceId(vol_id.id() * 4 + i);
			if (_halfFaces[hf_id.id()].Opposite.isValid() == false)
			{
				std::map<HalfFaceId, HalfFaceId>::iterator item = _surfaceFaces.find(hf_id);
				if (item != _surfaceFaces.end())
					_surfaceFaces.erase(item);
				/*else
				Assert(false, "A half face without opposite is a surface face.");*/
			}
			else
			{
				_surfaceFaces[_halfFaces[hf_id.id()].Opposite] = _halfFaces[hf_id.id()].Opposite;
			}

			_halfFaces[hf_id.id()].state().setFlag(StateFlags::Deleted);
			_halfFaces[hf_id.id()].Opposite = HalfFaceId();
		}

		// Delete the volume
		_volumes[vol_id.id()].state().setFlag(StateFlags::Deleted);

	}

	void HalfFaceTetraMesh::collapseEdge(EdgeId edge_id)
	{
		using namespace std;
		typedef std::map<HalfFaceVertices, HalfFaceId> face_map_t;

		// List of old volumes
		std::list<VolumeId> old_volumes;

		// Get the vertices of the edge.
		HalfEdgeVertices old_edge_vertices = verticesFromHalfEdge(_edges[edge_id.id()].HalfEdge);

		// Change the vertex location.
		assert(false); //setPoint(get<0>(old_edge_vertices), 0.5 * (getPoint(get<0>(old_edge_vertices)) + getPoint(get<1>(old_edge_vertices))));

		Edge& edge = _edges[edge_id.id()];
		HalfEdgeId start_he = edge.HalfEdge;
		HalfFaceId start_hf = start_he.halfFace();

		// Check the tetrahedron adjacent to the edge is on the boundary.
		// If yes, use the boundary half edge as the starting point.
		if (isSurfaceFace(start_he.mate().halfFace()))
		{
			start_he = start_he.mate();
			start_hf = start_he.halfFace();
		}

		if (isSurfaceFace(start_hf))
		{
			// The half face is on the boundary.
			// Iterate until we reach the next half face on the boundary.
			HalfEdgeId he = start_he;
			do
			{
				VolumeId vol = he.volume();
				collapseVolume(vol, edge_id);
				old_volumes.push_back(vol);

				he = radial(he.mate());
			} while (he.isValid() && !isSurfaceFace(he.halfFace()));
		}
		else
		{
			// Found an internal volume. Iterate through all volumes until the end is reached.
			// If the end is an external face, do a second iteration in the other direction.
			// Else the end is connected to the starting volume.
			HalfEdgeId he = start_he;
			do
			{
				VolumeId vol = he.volume();
				collapseVolume(vol, edge_id);
				old_volumes.push_back(vol);

				he = radial(he.mate());
			} while (he.isValid() && !(isSurfaceFace(he.halfFace()) || (he.halfFace() == start_hf)));

			if (he.halfFace() != start_hf)
			{
				he = radial(start_he);
				do
				{
					VolumeId vol = he.volume();
					collapseVolume(vol, edge_id);
					old_volumes.push_back(vol);

					he = radial(he.mate());
				} while (he.isValid() && !(isSurfaceFace(he.halfFace()) || (he.halfFace() == start_hf)));
			}
		}

		for (std::list<VolumeId>::iterator it = old_volumes.begin(); it != old_volumes.end(); ++it)
		{
			deleteVolume(*it);
		}
	}

	void HalfFaceTetraMesh::collapseVolume(VolumeId vol_id, EdgeId edge_id)
	{
		assert(vol_id.isValid());
		assert(_volumes[vol_id.id()].state().isValid());

		// Get the vertices of the volume
		VolumeVertices vertices = _volumes[vol_id.id()].Vertices;

		// Get the vertices of the edge.
		HalfEdgeVertices old_half_edge_vertices = verticesFromHalfEdge(_edges[edge_id.id()].HalfEdge);

		HalfFaceId hf_a = vertexToHalfFace(vol_id, old_half_edge_vertices[0]);
		HalfFaceId hf_b = vertexToHalfFace(vol_id, old_half_edge_vertices[1]);

		HalfFaceId opposite_hf_a = opposite(hf_a);
		HalfFaceId opposite_hf_b = opposite(hf_b);
		if (opposite_hf_a.isValid() && opposite_hf_b.isValid())
		{
			VolumeId other_volume = opposite_hf_a.volume();
			replaceVertex(other_volume, old_half_edge_vertices[1], old_half_edge_vertices[0]);

			replaceHalfFace(hf_b, opposite_hf_a);
			replaceHalfFace(hf_a, opposite_hf_b);
		}
		else if (opposite_hf_a.isValid() == false && opposite_hf_b.isValid())
		{
			_halfFaces[_halfFaces[hf_b.id()].Opposite.id()].Opposite = HalfFaceId();
			_surfaceFaces.erase(hf_a);
			_surfaceFaces[opposite_hf_b] = opposite_hf_b;
		}
		else if (opposite_hf_a.isValid() && opposite_hf_b.isValid() == false)
		{
			_halfFaces[_halfFaces[hf_a.id()].Opposite.id()].Opposite = HalfFaceId();
			_surfaceFaces.erase(hf_b);
			_surfaceFaces[opposite_hf_a] = opposite_hf_a;
		}
	}

	void HalfFaceTetraMesh::replaceVertex(VolumeId vol_id, VertexId old_vertex_id, VertexId new_vertex_id)
	{
		VolumeVertices& vertices = _volumes[vol_id.id()].Vertices;

		for (size_t i = 0; i < 4; i++)
		{
			if (vertices[i] == old_vertex_id)
			{
				vertices[i] = new_vertex_id;
				break;
			}
		}
	}

}}
