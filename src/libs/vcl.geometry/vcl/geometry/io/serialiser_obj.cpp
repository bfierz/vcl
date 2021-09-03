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
#include <vcl/geometry/io/serialiser_obj.h>

// Standard C++ library
#include <iostream>
#include <fstream>
#include <vector>

// VCL library
#include <vcl/util/stringparser.h>

namespace Vcl { namespace Geometry { namespace IO {
	static int convertToIndex(const std::string& str, int latest_idx)
	{
		int idx = atoi(str.c_str());
		if (idx < 0)
		{
			return latest_idx + 1 + idx;
		} else
		{
			return idx - 1;
		}
	}

	void ObjSerialiser::load(AbstractDeserialiser* deserialiser, const std::string& path) const
	{
		VclRequire(deserialiser != nullptr, "Deserialiser is given.");

		std::ifstream fin(path.c_str());
		VclCheck(fin.is_open() && !fin.eof(), "File exists, is not locked and is not empty.");
		if (!fin.is_open() || fin.eof())
			return;

		Vcl::Util::StringParser parser;
		parser.setInputStream(&fin);

		// Start importing the mesh
		deserialiser->begin();

		int latest_v_idx = -1;
		int latest_vn_idx = -1;
		int latest_vt_idx = -1;
		int latest_vc_idx = -1;
		Eigen::Vector2f v2;
		Eigen::Vector3f v3;
		Eigen::Vector4f v4;
		std::vector<float> vN;
		std::vector<std::array<unsigned int, 2>> primitive_point;
		std::vector<std::array<unsigned int, 3>> primitive_line;
		std::vector<std::array<unsigned int, 4>> primitive_face;
		std::vector<unsigned int> primitive_vertices;
		std::vector<std::string> names;

		// Record if a material was used.
		bool materials_are_loaded = false;
		bool materials_are_used = false;

		std::string token;
		std::string face_corner;
		while (parser.loadLine())
		{
			parser.readString(&token);

			if (token == "o")
			{
				std::string name;
				parser.readString(&name);

				// Set the mesh name
				//loader->setName(name);
			}

			// Vertex data
			else if (token == "v")
			{
				float p0, p1, p2;
				parser.readFloat(&p0);
				parser.readFloat(&p1);
				parser.readFloat(&p2);

				vN = { p0, p1, p2 };
				deserialiser->addNode(vN);
				latest_v_idx++;
			} else if (token == "vn")
			{
				parser.readFloat(&v3(0));
				parser.readFloat(&v3(1));
				parser.readFloat(&v3(2));

				deserialiser->addNormal(v3);
				latest_vn_idx++;
			} else if (token == "vt")
			{
				parser.readFloat(&v2(0));
				parser.readFloat(&v2(1));

				//deserialiser->addTexture(v2);
				latest_vt_idx++;
			} else if (token == "vc") // Non standard extension
			{
				parser.readFloat(&v4(0));
				parser.readFloat(&v4(1));
				parser.readFloat(&v4(2));
				parser.readFloat(&v4(3));

				//latest_vc_idx = deserialiser->addColour(v4);
				latest_vc_idx++;
			}

			// Primitive data
			else if (token == "p")
			{
				VclDebugError("Not implemented.");
				//primitive_point
			} else if (token == "l")
			{
				VclDebugError("Not implemented.");
				//primitive_line
			} else if (token == "f")
			{
				std::array<unsigned int, 4> corner;
				corner.fill(0xffffffff);
				primitive_face.clear();
				primitive_vertices.clear();

				while (parser.readString(&face_corner))
				{
					std::string::size_type pos_one, pos_two, pos_three;

					// Read position
					pos_one = face_corner.find("/", 0);
					corner[0] = convertToIndex(face_corner.substr(0, pos_one), latest_v_idx);

					// Read texture coordinate
					if (pos_one != face_corner.npos)
					{
						pos_two = face_corner.find("/", pos_one + 1);
						if (pos_two - pos_one > 1)
							corner[1] = convertToIndex(face_corner.substr(pos_one + 1, pos_two - (pos_one + 1)), latest_vt_idx);

						// Read normal
						if (pos_two != face_corner.npos)
						{
							pos_three = face_corner.find("/", pos_two + 1);
							if (pos_three - pos_two > 1)
								corner[2] = convertToIndex(face_corner.substr(pos_two + 1, pos_three - (pos_two + 1)), latest_vn_idx);

							// Read colour
							if (pos_three != face_corner.npos)
							{
								if (face_corner.length() - pos_three > 2)
									corner[3] = convertToIndex(face_corner.substr(pos_three + 1), latest_vc_idx);
							}
						}
					}

					primitive_face.emplace_back(corner);
					primitive_vertices.emplace_back(corner[0]);
				}

				deserialiser->addFace(primitive_vertices);
			}

			parser.skipLine();
		}
		deserialiser->end();
		fin.close();
	}

	void ObjSerialiser::store(AbstractSerialiser* serialiser, const std::string& path) const
	{
		using namespace std;

		if (serialiser == NULL) return;

		// Definition of the IO format
		Eigen::IOFormat io_format(8, 0, "", " ", "", "", "", "");

		ofstream fout(path.c_str());
		if (!fout.is_open())
			return;

		// Start writing the mesh
		serialiser->begin();

		// Element counts
		int nr_nodes = serialiser->nrNodes();
		int nr_faces = serialiser->nrFaces();

		// Write vertices
		for (size_t i = 0; i < nr_nodes; ++i)
		{
			std::vector<float> pos;
			serialiser->fetchNode(pos);
			fout << "v " << pos[0] << " " << pos[1] << " " << pos[2] << endl;

			if (serialiser->hasNormals())
			{
				Vector3f normal;
				serialiser->fetchNormal(normal);

				fout << "vn " << normal.format(io_format) << endl;
			}
		}
		//if (tex_coords.size() > 0)
		//{
		//	for (size_t i = 0; i < tex_coords.size(); ++i)
		//	{
		//		fout << "vt " << tex_coords[i].format(io_format) << endl;
		//	}
		//}

		// Write polygon
		for (size_t i = 0; i < nr_faces; ++i)
		{
			std::vector<unsigned int> face;
			serialiser->fetchFace(face);

			// Start a face
			fout << "f";

			// Write the indices
			for (int j = 0; j < face.size(); j++)
			{
				fout << " " << face[j] + 1;
				if (serialiser->hasNormals())
				{
					fout << "/";
					//if (tex_coords.size() > 0) fout << it[i].x() + 1;
					fout << "/";
					fout << face[j] + 1;
				}
			}

			// Finisch the face
			fout << endl;
		}

		// Write footer
		fout.close();

		// End writing the mesh
		serialiser->end();
	}
}}}
