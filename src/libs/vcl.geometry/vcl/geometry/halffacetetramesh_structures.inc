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

class Vertex
{
public: // Initialisation
	Vertex() = default;
	Vertex(HalfFaceId hf_id) : HalfFace(hf_id) {}
public: // Vertex data
	HalfFaceId HalfFace;
};

class HalfEdge
{
public: // Initialisation
	HalfEdge() = default;
	HalfEdge(EdgeId e_id) : Edge(e_id) {}
public: // Half edge data
	EdgeId Edge;
};

class HalfFace
{
public: // Initialisation
	HalfFace() = default;
public: // Half face data
	HalfFaceId Opposite;
	FaceId Face;
};

class Edge
{
public: // Initialisation
	Edge() = default;
	Edge(HalfEdgeId he_id) : HalfEdge(he_id) {}
public: // Edge data
	HalfEdgeId HalfEdge;
};

class Face
{
public: // Initialisation
	Face() = default;
	Face(HalfFaceId hf_id) : HalfFace(hf_id) {}
public: // Face data
	HalfFaceId HalfFace;
};

class Volume
{
public: // Initialisation
	Volume() = default;
	Volume(const std::array<VertexId, 4>& vertices) : Vertices{vertices}{}
public: // Volume data
	std::array<VertexId, 4> Vertices;
};
