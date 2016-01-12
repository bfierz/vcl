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
#include <vcl/geometry/io/serialiser_nvidia_tet_file.h>

// C++ standard library
#include <array>
#include <iostream>
#include <fstream>
#include <vector>

// VCL
#include <vcl/util/stringparser.h>

namespace Vcl { namespace Geometry { namespace IO
{
	void NvidiaTetSerialiser::load(AbstractDeserialiser* deserialiser, const std::string& path) const
	{
		using namespace std;

		Require(deserialiser != nullptr, "Deserialiser is given.");

		ifstream fin(path.c_str());
		if (!fin.is_open() || fin.eof())
		{
			std::cout << "Could not open file: " << path << std::endl;
			return;
		}

		Vcl::Util::StringParser parser;
		parser.setInputStream(&fin);

		// Start importing the mesh
		deserialiser->begin();

		std::vector<float> position(3);
		std::vector<unsigned int> volume(4);

		string buffer;
		while (parser.loadLine())
		{
			parser.readString(&buffer);
			if (buffer == "v")
			{
				float p0, p1, p2;
				parser.readFloat(&p0);
				parser.readFloat(&p1);
				parser.readFloat(&p2);

				position = { p0, p1, p2 };
				deserialiser->addNode(position);
			}
			else if (buffer == "t")
			{
				int i0, i1, i2, i3;
				parser.readInt(&i0);
				parser.readInt(&i1);
				parser.readInt(&i2);
				parser.readInt(&i3);

				volume = { (unsigned) i0, (unsigned)i1, (unsigned)i2, (unsigned)i3 };
				deserialiser->addVolume(volume);
			}
			else if (buffer == "l")
			{
			}
		}
		fin.close();

		// Finalise the mesh
		deserialiser->end();
	}

	void NvidiaTetSerialiser::store(AbstractSerialiser* serialiser, const std::string& path) const
	{
		using namespace std;

		Require(serialiser != nullptr, "Serialiser is given.");

		ofstream fout(path.c_str());
		if (!fout.is_open())
			return;

		// Start writing the mesh
		serialiser->begin();

		// Element counts
		int nr_nodes = serialiser->nrNodes();
		int nr_cells = serialiser->nrVolumes();

		// Write vertices
		for (unsigned int i = 0; i < nr_nodes; ++i)
		{
			std::vector<float> pos;
			serialiser->fetchNode(pos);
			fout << "v " << pos[0] << " " << pos[1] << " " << pos[2] << endl;
		}

		// Write tetrahedra
		for (unsigned int i = 0; i < nr_cells; ++i)
		{
			std::vector<unsigned int> cell;
			serialiser->fetchVolume(cell);

			if (cell.size() != 4)
				continue;

			fout << "t " << cell[0] << " " << cell[1] << " " << cell[2] << " " << cell[3] << endl;
		}

		// Write footer
		fout.close();
		
		// End writing the mesh
		serialiser->end();
	}
}}}
