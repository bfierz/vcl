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
#include <vcl/geometry/io/serialiser_tetgen.h>

// Standard C++ library
#include <iostream>
#include <fstream>
#include <vector>

// VCL
#include <vcl/util/stringparser.h>

namespace Vcl { namespace Geometry { namespace IO
{
	void TetGenSerialiser::load(AbstractDeserialiser* deserialiser, const std::string& path) const
	{
		using namespace std;

		Require(deserialiser != nullptr, "Deserialiser is given.");

		// Check for the file endings
		string node_path;
		string ele_path;

		if (path.substr(path.length() - 5) == ".node")
		{
			node_path = path;
			ele_path = path.substr(0, path.length() - 5) + ".ele";
		}
		else if (path.substr(path.length() - 4) == ".ele")
		{
			node_path = path.substr(0, path.length() - 4) + ".node";
			ele_path = path;
		}

		ifstream fin_node(node_path.c_str());
		ifstream fin_ele(ele_path.c_str());

		Check(fin_node.is_open() && !fin_node.eof(), "File exists, is not locked and is not empty.");
		if (!fin_node.is_open() || fin_node.eof())
			return;

		Check(fin_ele.is_open() && !fin_ele.eof(), "File exists, is not locked and is not empty.");
		if (!fin_ele.is_open() || fin_ele.eof())
			return;

		Vcl::Util::StringParser node_parser;
		node_parser.setInputStream(&fin_node);

		Vcl::Util::StringParser ele_parser;
		ele_parser.setInputStream(&fin_ele);

		// Start importing the mesh
		deserialiser->begin();

		std::vector<float> position(3);
		std::vector<unsigned int> tetrahedron(4);

		// Read the node header
		string token;
		while (node_parser.loadLine())
		{
			// Skips all the empty lines
			if (node_parser.readString(&token))
			{
				if (token == "#")
				{
					// Ignore the rest of the line
					node_parser.skipLine();
				}
				else
				{
					int size = stoi(token);
					int dim;
					int attribute;
					int boundary;

					node_parser.readInt(&dim);
					node_parser.readInt(&attribute);
					node_parser.readInt(&boundary);

					deserialiser->sizeHintNodes(size);

					break;
				}
			}
		}
		
		// Read the position data
		while (node_parser.loadLine())
		{
			// Skips all the empty lines
			if (node_parser.readString(&token))
			{
				if (token == "#")
				{
					// Ignore the rest of the line
					node_parser.skipLine();
				}
				else
				{
					int index = stoi(token);
					node_parser.readFloat(&position[0]);
					node_parser.readFloat(&position[1]);
					node_parser.readFloat(&position[2]);

					deserialiser->addNode(position);
				}
			}
		}
		
		// Read the element header
		while (ele_parser.loadLine())
		{
			// Skips all the empty lines
			if (ele_parser.readString(&token))
			{
				if (token == "#")
				{
					// Ignore the rest of the line
					ele_parser.skipLine();
				}
				else
				{
					int size = stoi(token);
					int dim;
					int attribute;

					ele_parser.readInt(&dim);
					ele_parser.readInt(&attribute);

					deserialiser->sizeHintVolumes(size);

					break;
				}
			}
		}
		
		// Read the element data
		while (ele_parser.loadLine())
		{
			// Skips all the empty lines
			if (ele_parser.readString(&token))
			{
				if (token == "#")
				{
					// Ignore the rest of the line
					ele_parser.skipLine();
				}
				else
				{
					int index = stoi(token);
					ele_parser.readInt((int*) &tetrahedron[0]); tetrahedron[0] -= 1;
					ele_parser.readInt((int*) &tetrahedron[1]); tetrahedron[1] -= 1;
					ele_parser.readInt((int*) &tetrahedron[2]); tetrahedron[2] -= 1;
					ele_parser.readInt((int*) &tetrahedron[3]); tetrahedron[3] -= 1;

					deserialiser->addVolume(tetrahedron);
				}
			}
		}
		
		deserialiser->end();

		fin_node.close();
		fin_ele.close();
	}
}}}
