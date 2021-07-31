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
#include <string>

// VCL
#include <vcl/geometry/io/serialiser.h>
#include <vcl/geometry/tetramesh.h>

namespace Vcl { namespace Geometry { namespace IO
{
	class TetraMeshDeserialiser : public AbstractDeserialiser
	{
	public:
		std::unique_ptr<TetraMesh> fetch() { return std::move(_mesh); }

	public:
		virtual void begin();
		virtual void end();

	public:
		virtual void sizeHintNodes(unsigned int hint);
		virtual void sizeHintEdges(unsigned int hint);
		virtual void sizeHintFaces(unsigned int hint);
		virtual void sizeHintVolumes(unsigned int hint);

	public:
		virtual void addNode(const std::vector<float>& coordinates);
		virtual void addEdge(const std::vector<unsigned int>& indices);
		virtual void addFace(const std::vector<unsigned int>& indices);
		virtual void addVolume(const std::vector<unsigned int>& indices);

		virtual void addNormal(const Vector3f& normal);

	private:
		//! Generated mesh
		std::unique_ptr<TetraMesh> _mesh;

		std::vector<Eigen::Vector3f> _positions;
		std::vector<std::array<unsigned int, 4>> _volumes;
	};
}}}
