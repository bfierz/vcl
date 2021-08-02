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
#include <string>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Geometry { namespace IO {
	struct AbstractDeserialiser
	{
		virtual ~AbstractDeserialiser() = default;

		// Inform the deserialiser of begin and end of the file loading process
		virtual void begin() = 0;
		virtual void end() = 0;

		// File formats storing the number of elements may give size hints
		// to the deserialiser to optimise performance
		virtual void sizeHintNodes(unsigned int hint) = 0;
		virtual void sizeHintEdges(unsigned int hint) = 0;
		virtual void sizeHintFaces(unsigned int hint) = 0;
		virtual void sizeHintVolumes(unsigned int hint) = 0;

		// Add elements in sequential order. Indices will be created sequentially when adding
		// elements.
		virtual void addNode(const std::vector<float>& coordinates) = 0;
		virtual void addEdge(const std::vector<unsigned int>& indices) = 0;
		virtual void addFace(const std::vector<unsigned int>& indices) = 0;
		virtual void addVolume(const std::vector<unsigned int>& indices) = 0;

		// Named properties
		virtual void addNormal(const Vector3f& normal) = 0;
	};

	struct AbstractSerialiser
	{
		virtual ~AbstractSerialiser() = default;

		// Inform the serialiser of begin and end of the exporting process
		virtual void begin() = 0;
		virtual void end() = 0;

		// Access the element counts
		virtual unsigned int nrNodes() = 0;
		virtual unsigned int nrEdges() = 0;
		virtual unsigned int nrFaces() = 0;
		virtual unsigned int nrVolumes() = 0;

		virtual void fetchNode(std::vector<float>& coordinates) = 0;
		virtual void fetchEdge(std::vector<unsigned int>& indices) = 0;
		virtual void fetchFace(std::vector<unsigned int>& indices) = 0;
		virtual void fetchVolume(std::vector<unsigned int>& indices) = 0;

		// Named properties
		virtual bool hasNormals() const { return false; }
		virtual void fetchNormal(Vector3f& normal) { normal.setZero(); }
	};

	class Serialiser
	{
	public:
		virtual ~Serialiser() = default;

	public: // Read mesh file
		virtual void load(AbstractDeserialiser* /*deserialiser*/, const std::string& /*path*/) const { VclDebugError("Not implemented."); }

	public: // Write mesh file
		virtual void store(AbstractSerialiser* /*serialiser*/, const std::string& /*path*/) const { VclDebugError("Not implemented."); }
	};
}}}
