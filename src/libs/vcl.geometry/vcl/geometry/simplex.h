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

// VCL 
#include <vcl/core/contract.h>
#include <vcl/geometry/genericid.h>
#include <vcl/geometry/propertygroup.h>
#include <vcl/geometry/property.h>

namespace Vcl { namespace Geometry
{
	template<typename MeshIndex>
	struct IndexDescriptionTrait;

	template<typename ID, typename ElementT, typename IndexT>
	class Enumerator
	{
	public:
		Enumerator(const IndexT* mesh, ID start_id, ID end_id)
		: _mesh(mesh), _startId(start_id), _endId(end_id)
		{
			while ((_startId < _endId) && (_mesh->element(_startId).isValid() == false))
				_startId = ID(_startId.id() + 1);
		}

	public:
		ID operator*() const
		{
			return _startId;
		}

		const ElementT* operator-> () const
		{
			return &_mesh->element(_startId);
		}

		Enumerator operator++(int)
		{
			Enumerator tmp(*this);
			_startId = ID(_startId.id() + 1);

			while ((_startId < _endId) && (_mesh->element(_startId).isValid() == false))
				_startId = ID(_startId.id() + 1);

			return tmp;
		}

		Enumerator& operator++()
		{
			_startId = ID(_startId.id() + 1);

			while ((_startId.id() < _endId.id()) && (_mesh->element(_startId).isValid() == false))
				_startId = ID(_startId.id() + 1);

			return *this;
		}

		bool empty() const
		{
			return !(_startId < _endId);
		}

	private:
		const IndexT* _mesh;
		ID _startId, _endId;
	};

	template<typename Derived>
	class SimplexLevel0
	{
	public: // IDs
		using VertexId = typename IndexDescriptionTrait<Derived>::VertexId;

	public: // Structures
		using Vertex = typename IndexDescriptionTrait<Derived>::Vertex;
		using VertexMetaData = typename IndexDescriptionTrait<Derived>::VertexMetaData;

	public: // Iterator Types
		using VertexEnumerator = Enumerator<VertexId, Vertex, SimplexLevel0<Derived>>;

	public:
		SimplexLevel0()
		: _vertexData("VertexData")
		{
            _vertices = _vertexData.template add<Vertex>("Vertices");
            _verticesMetaData = _vertexData.template add<VertexMetaData>("VerticesMetaData");
		}
		virtual ~SimplexLevel0() = default;

	public: // Properties
		unsigned int nrVertices() const { return static_cast<unsigned int>(_vertexData.propertySize()); }

		const Vertex& vertex(VertexId id) const { return element(id); }
		      Vertex& vertex(VertexId id)       { return element(id); }

		const Vertex& element(VertexId id) const { return static_cast<const Derived*>(this)->access(_vertices, id); }
		      Vertex& element(VertexId id)       { return static_cast<      Derived*>(this)->access(_vertices, id); }
			  
		const VertexMetaData& metaData(VertexId id) const { return static_cast<const Derived*>(this)->access(_verticesMetaData, id); }
		      VertexMetaData& metaData(VertexId id)       { return static_cast<      Derived*>(this)->access(_verticesMetaData, id); }
			  
		ConstPropertyPtr<Vertex, VertexId> vertices() const { return _vertices; }

	protected:
		const PropertyGroup<VertexId>& vertexProperties() const { return _vertexData; }
		      PropertyGroup<VertexId>& vertexProperties()       { return _vertexData; }

	public: // Enumerators
		VertexEnumerator vertexEnumerator() const { return{ this, VertexId(0), VertexId(static_cast<typename VertexId::IdType>(_vertices.size())) }; }

	protected: // Element access
		template<typename element_type, typename index_type>
		element_type& access(Property<element_type, index_type>& storage, index_type id)
		{
			Require(id.id() < storage->size(), "Id is in Range.");
			return storage[id];
		}

		template<typename element_type, typename index_type>
		const element_type& access(const Property<element_type, index_type>& storage, index_type id) const
		{
			Require(id.id() < storage->size(), "Id is in Range.");
			return storage[id];
		}

	protected: // Properties

		//! Data associated with a vertex
		PropertyGroup<VertexId> _vertexData;

		//! Generic vertex data
		PropertyPtr<Vertex, VertexId> _vertices;

		//! Vertex meta data
		PropertyPtr<VertexMetaData, VertexId> _verticesMetaData;
	};

	template<typename Derived>
	class SimplexLevel1
	{
	public: // IDs
		using EdgeId = typename IndexDescriptionTrait<Derived>::EdgeId;
		
	public: // Structures
		using Edge = typename IndexDescriptionTrait<Derived>::Edge;
		using EdgeMetaData = typename IndexDescriptionTrait<Derived>::EdgeMetaData;


	public: // Iterator Types
        using EdgeEnumerator = Enumerator<EdgeId, Edge, SimplexLevel1<Derived>>;

	public:
		SimplexLevel1()
		: _edgeData("EdgeGroup")
		{
            _edges = _edgeData.template add<Edge>("Edges");
            _edgesMetaData = _edgeData.template add<EdgeMetaData>("EdgesMetaData");
		}
		virtual ~SimplexLevel1() = default;

	public: // Properties
		unsigned int nrEdges() const { return static_cast<unsigned int>(_edgeData.propertySize()); }

		const Edge& edge(EdgeId id) const { return element(id); }
		      Edge& edge(EdgeId id)       { return element(id); }

		const Edge& element(EdgeId id) const { return static_cast<const Derived*>(this)->access(_edges, id); }
		      Edge& element(EdgeId id)       { return static_cast<      Derived*>(this)->access(_edges, id); }
			  
		const EdgeMetaData& metaData(EdgeId id) const { return static_cast<const Derived*>(this)->access(_edgesMetaData, id); }
		      EdgeMetaData& metaData(EdgeId id)       { return static_cast<      Derived*>(this)->access(_edgesMetaData, id); }
			  
		ConstPropertyPtr<Edge, EdgeId> edges() const { return _edges; }

	protected:
		const PropertyGroup<EdgeId>& edgeProperties() const { return _edgeData; }
		      PropertyGroup<EdgeId>& edgeProperties()       { return _edgeData; }

	public: // Enumerators
		EdgeEnumerator edgeEnumerator() const { return{ this, EdgeId(0), EdgeId(static_cast<typename EdgeId::IdType>(_edges.size())) }; }

	protected: // Properties
		
		//! Data associated with an edge
		PropertyGroup<EdgeId> _edgeData;

		//! Generic edge data
		PropertyPtr<Edge, EdgeId> _edges;

		//! Edge meta data
		PropertyPtr<EdgeMetaData, EdgeId> _edgesMetaData;
	};

	template<typename Derived>
	class SimplexLevel2
	{
	public: // IDs
		using FaceId = typename IndexDescriptionTrait<Derived>::FaceId;
		
	public: // Structures
		using Face = typename IndexDescriptionTrait<Derived>::Face;
		using FaceMetaData = typename IndexDescriptionTrait<Derived>::FaceMetaData;

	public: // Iterator Types
        using FaceEnumerator = Enumerator<FaceId, Face, SimplexLevel2<Derived>>;

	public:
		SimplexLevel2()
		: _faceData("FaceGroup")
		{
            _faces = _faceData.template add<Face>("Faces");
            _facesMetaData = _faceData.template add<FaceMetaData>("FacesMetaData");
		}
		virtual ~SimplexLevel2() = default;
		
	public: // Properties
		unsigned int nrFaces() const { return static_cast<unsigned int>(_faceData.propertySize()); }

		const Face& face(FaceId id) const { return element(id); }
		      Face& face(FaceId id)       { return element(id); }

		const Face& element(FaceId id) const { return static_cast<const Derived*>(this)->access(_faces, id); }
		      Face& element(FaceId id)       { return static_cast<      Derived*>(this)->access(_faces, id); }
			  
		const FaceMetaData& metaData(FaceId id) const { return static_cast<const Derived*>(this)->access(_facesMetaData, id); }
		      FaceMetaData& metaData(FaceId id)       { return static_cast<      Derived*>(this)->access(_facesMetaData, id); }
			  
		ConstPropertyPtr<Face, FaceId> faces() const { return _faces; }

	protected:
		const PropertyGroup<FaceId>& faceProperties() const { return _faceData; }
		      PropertyGroup<FaceId>& faceProperties()       { return _faceData; }

	public: // Enumerators
		FaceEnumerator faceEnumerator() const { return{ this, FaceId(0), FaceId(static_cast<typename FaceId::IdType>(_faces.size())) }; }

	protected: // Properties
		
		//! Data associated with a face
		PropertyGroup<FaceId> _faceData;

		//! Generic face data
		PropertyPtr<Face, FaceId> _faces;

		//! Face meta data
		PropertyPtr<FaceMetaData, FaceId> _facesMetaData;
	};

	template<typename Derived>
	class SimplexLevel3
	{
	public: // IDs
		using VolumeId = typename IndexDescriptionTrait<Derived>::VolumeId;
		
	public: // Structures
		using Volume = typename IndexDescriptionTrait<Derived>::Volume;
		using VolumeMetaData = typename IndexDescriptionTrait<Derived>::VolumeMetaData;

	public: // Iterator Types
        using VolumeEnumerator = Enumerator<VolumeId, Volume, SimplexLevel3<Derived>>;

	public:
		SimplexLevel3()
		: _volumeData("VolumeGroup")
		{
            _volumes = _volumeData.template add<Volume>("Volumes");
            _volumesMetaData = _volumeData.template add<VolumeMetaData>("VolumesMetaData");
		}
		virtual ~SimplexLevel3() = default;

	public: // Properties
		unsigned int nrVolumes() const { return static_cast<unsigned int>(_volumeData.propertySize()); }

		const Volume& volume(VolumeId id) const { return element(id); }
		      Volume& volume(VolumeId id)       { return element(id); }

		const Volume& element(VolumeId id) const { return static_cast<const Derived*>(this)->access(_volumes, id); }
		      Volume& element(VolumeId id)       { return static_cast<      Derived*>(this)->access(_volumes, id); }
			  
		const VolumeMetaData& metaData(VolumeId id) const { return static_cast<const Derived*>(this)->access(_volumesMetaData, id); }
		      VolumeMetaData& metaData(VolumeId id)       { return static_cast<      Derived*>(this)->access(_volumesMetaData, id); }
			  
		ConstPropertyPtr<Volume, VolumeId> volumes() const { return _volumes; }

	protected:
		const PropertyGroup<VolumeId>& volumeProperties() const { return _volumeData; }
		      PropertyGroup<VolumeId>& volumeProperties()       { return _volumeData; }

	public: // Enumerators
		VolumeEnumerator volumeEnumerator() const { return{ this, VolumeId(0), VolumeId(static_cast<typename VolumeId::IdType>(_volumes.size())) }; }

	protected: // Properties

		//! Data associated with a volume
		PropertyGroup<VolumeId> _volumeData;

		//! Generic volume data
		PropertyPtr<Volume, VolumeId> _volumes;

		//! Volume meta data
		PropertyPtr<VolumeMetaData, VolumeId> _volumesMetaData;
	};
}}
