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

VCL_CREATEID(VertexId, IndexType); // Size: n0
VCL_CREATEID(EdgeId, IndexType);   // Size: n1
VCL_CREATEID(FaceId, IndexType);   // Size: n2
class VolumeId;                    // Size: n3

class HalfEdgeId;                  // Size: n3 * 4 * 3
class HalfFaceId;                  // Size: n3 * 4

class HalfEdgeId : public GenericId<HalfEdgeId, IndexType>
{
public:
	HalfEdgeId(){}
	explicit HalfEdgeId(IndexType id) : GenericId<HalfEdgeId, IndexType>(id) {}

public:
	HalfEdgeId next() const
	{
		VclRequire(isValid(), "Id is valid.");

		static int NextHalfEdge[4][4] = { {-1,  3,  1,  2},
										  { 2, -1,  3,  0},
										  { 3,  0, -1,  1},
										  { 1,  2,  0, -1} };

		HalfFaceId hf_id;
		IndexType vhe = 0;
		split(*this, hf_id, vhe);

		return merge(hf_id, static_cast<IndexType>(NextHalfEdge[vhe][hf_id.id() % 4]));
	}
	
	HalfEdgeId mate() const
	{
		VclRequire(isValid(), "Id is valid.");

		static int NextHalfEdge[4][4] = { {-1,  3,  1,  2},
										  { 2, -1,  3,  0},
										  { 3,  0, -1,  1},
										  { 1,  2,  0, -1} };
		
		HalfFaceId hf_id;
		IndexType vhe = 0;
		split(*this, hf_id, vhe);

		IndexType prev = static_cast<IndexType>(NextHalfEdge[hf_id.id() % 4][vhe]);
		IndexType next = static_cast<IndexType>(NextHalfEdge[vhe][hf_id.id() % 4]);

		IndexType base_hf = hf_id.volume().halfFace(0).id();
		return merge(HalfFaceId(base_hf + prev), next);
	}
	
	HalfEdgeId previous() const
	{
		VclRequire(isValid(), "Id is valid.");
		
		static int NextHalfEdge[4][4] = { {-1,  3,  1,  2},
										  { 2, -1,  3,  0},
										  { 3,  0, -1,  1},
										  { 1,  2,  0, -1} };
		
		HalfFaceId hf_id;
		IndexType vhe = 0;
		split(*this, hf_id, vhe);

		return merge(hf_id, static_cast<IndexType>(NextHalfEdge[hf_id.id() % 4][vhe]));
	}

private:
	void split(HalfEdgeId he_id, HalfFaceId& hf_id, IndexType& vhe) const
	{
		static int VertexHalfEdge[4][3] = {{1, 2, 3},
										   {2, 0, 3},
										   {3, 0, 1},
										   {0, 2, 1} };


		hf_id = he_id.halfFace();
		vhe = VertexHalfEdge[hf_id.id() % 4][he_id.id() % 3];
	}

	HalfEdgeId merge(HalfFaceId hf_id, IndexType vhe) const
	{
		static int HalfEdgeIndex[4][4] = { {-1,  0,  1,  2},
										   { 1, -1,  0,  2},
										   { 1,  2, -1,  0},
										   { 0,  2,  1, -1} };

		return HalfEdgeId(hf_id.id() * 3 + static_cast<IndexType>(HalfEdgeIndex[hf_id.id() % 4][vhe])); 
	}

public:
	HalfFaceId halfFace() const { VclRequire(isValid(), "Id is valid."); return HalfFaceId(_id / 3); }
	VolumeId volume() const { VclRequire(isValid(), "Id is valid."); return VolumeId(_id / 12); }
};

class HalfFaceId : public GenericId<HalfFaceId, IndexType>
{
public:
	HalfFaceId(){}
	explicit HalfFaceId(IndexType id) : GenericId<HalfFaceId, IndexType>(id) {}

public:
	HalfFaceId next() const
	{
		VclRequire(isValid(), "Id is valid.");
		return HalfFaceId( ((_id&(~3)) + ((_id+1)&3)) );
	}

	HalfFaceId previous() const
	{
		VclRequire(isValid(), "Id is valid.");
		return HalfFaceId( (_id&(~3)) + ((_id+3)&3) );
	}

public:
	HalfEdgeId halfEdge(IndexType i) const { VclRequire(isValid(), "Id is valid."); VclRequire(i < 3, "Id is in [0, 3["); return HalfEdgeId(_id * 3 + i); }
	VolumeId volume() const { VclRequire(isValid(), "Id is valid."); return VolumeId(_id >> 2); }
};

class VolumeId : public GenericId<VolumeId, IndexType>
{
public:
	VolumeId(){}
	explicit VolumeId(IndexType id) : GenericId<VolumeId, IndexType>(id) {}

public:
	HalfFaceId halfFace(IndexType i) const { VclRequire(isValid(), "Id is valid."); VclRequire(i < 3, "Id is in [0, 4["); return HalfFaceId(_id * 4 + i); }
};
