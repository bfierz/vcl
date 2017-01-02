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
#include <vcl/geometry/meshfactory.h>

// VCL
#include <vcl/geometry/tetrahedron.h>

namespace Vcl { namespace Geometry
{
	std::unique_ptr<TetraMesh> MeshFactory<TetraMesh>::createHomogenousCubes(unsigned int count_x, unsigned int count_y, unsigned int count_z)
	{
		#define SpatialToLinearIndex(a, b, c) ((c) * (xmax + 1) * (ymax + 1) + (b) * (xmax + 1) + (a))

		const unsigned int xmax = count_x;
		const unsigned int ymax = count_y;
		const unsigned int zmax = count_z;

		typedef std::array<unsigned int, 4> volume_t;
		std::vector<Vector3f> positions;
		std::vector<volume_t> volumes;

		positions.resize((xmax + 1)*(ymax + 1)*(zmax + 1));
		for (unsigned int k = 0; k < zmax + 1; k++)
		{
			for (unsigned int j = 0; j < ymax + 1; j++)
			{
				for (unsigned int i = 0; i < xmax + 1; i++)
				{
					positions[SpatialToLinearIndex(i, j, k)] = Vector3f((float)i, (float)j, (float)k);
				}
			}
		}

		for (unsigned int k = 0; k < zmax; k++)
		{
			for (unsigned int j = 0; j < ymax; j++)
			{
				for (unsigned int i = 0; i < xmax; i++)
				{
					if (((i % 2 == 0) && (j % 2 == 0) && (k % 2 == 0)) ||
						((i % 2 == 1) && (j % 2 == 1) && (k % 2 == 0)))
					{
						volume_t v0 =
						{
							SpatialToLinearIndex(i + 0, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 1),
							SpatialToLinearIndex(i + 1, j + 1, k + 0),
							SpatialToLinearIndex(i + 0, j + 1, k + 1)
						};
						volumes.push_back(v0);

						volume_t v1 =
						{
							SpatialToLinearIndex(i + 0, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 1)
						};
						volumes.push_back(v1);

						volume_t v2 =
						{
							SpatialToLinearIndex(i + 0, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 0),
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 0, j + 1, k + 1)
						};
						volumes.push_back(v2);

						volume_t v3 =
						{
							SpatialToLinearIndex(i + 0, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 1),
							SpatialToLinearIndex(i + 0, j + 1, k + 1),
							SpatialToLinearIndex(i + 0, j + 0, k + 1)
						};
						volumes.push_back(v3);

						volume_t v4 =
						{
							SpatialToLinearIndex(i + 1, j + 1, k + 0),
							SpatialToLinearIndex(i + 0, j + 1, k + 1),
							SpatialToLinearIndex(i + 1, j + 0, k + 1),
							SpatialToLinearIndex(i + 1, j + 1, k + 1)
						};
						volumes.push_back(v4);
					}

					if (((i % 2 == 1) && (j % 2 == 0) && (k % 2 == 0)) ||
						((i % 2 == 0) && (j % 2 == 1) && (k % 2 == 0)))
					{
						volume_t v0 =
						{
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 1),
							SpatialToLinearIndex(i + 0, j + 0, k + 1),
							SpatialToLinearIndex(i + 1, j + 0, k + 0)
						};
						volumes.push_back(v0);

						volume_t v1 =
						{
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 1),
							SpatialToLinearIndex(i + 1, j + 0, k + 0)
						};
						volumes.push_back(v1);

						volume_t v2 =
						{
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 0),
							SpatialToLinearIndex(i + 0, j + 0, k + 1),
							SpatialToLinearIndex(i + 0, j + 0, k + 0)
						};
						volumes.push_back(v2);

						volume_t v3 =
						{
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 1),
							SpatialToLinearIndex(i + 0, j + 1, k + 1),
							SpatialToLinearIndex(i + 0, j + 0, k + 1)
						};
						volumes.push_back(v3);

						volume_t v4 =
						{
							SpatialToLinearIndex(i + 1, j + 0, k + 0),
							SpatialToLinearIndex(i + 0, j + 0, k + 1),
							SpatialToLinearIndex(i + 1, j + 0, k + 1),
							SpatialToLinearIndex(i + 1, j + 1, k + 1)
						};
						volumes.push_back(v4);

					}

					if (((i % 2 == 0) && (j % 2 == 0) && (k % 2 == 1)) ||
						((i % 2 == 1) && (j % 2 == 1) && (k % 2 == 1)))
					{
						volume_t v0 =
						{
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 1),
							SpatialToLinearIndex(i + 0, j + 0, k + 1),
							SpatialToLinearIndex(i + 1, j + 0, k + 0)
						};
						volumes.push_back(v0);

						volume_t v1 =
						{
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 1),
							SpatialToLinearIndex(i + 1, j + 0, k + 0)
						};
						volumes.push_back(v1);

						volume_t v2 =
						{
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 0),
							SpatialToLinearIndex(i + 0, j + 0, k + 1),
							SpatialToLinearIndex(i + 0, j + 0, k + 0)
						};
						volumes.push_back(v2);

						volume_t v3 =
						{
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 1),
							SpatialToLinearIndex(i + 0, j + 1, k + 1),
							SpatialToLinearIndex(i + 0, j + 0, k + 1)
						};
						volumes.push_back(v3);

						volume_t v4 =
						{
							SpatialToLinearIndex(i + 1, j + 0, k + 0),
							SpatialToLinearIndex(i + 0, j + 0, k + 1),
							SpatialToLinearIndex(i + 1, j + 0, k + 1),
							SpatialToLinearIndex(i + 1, j + 1, k + 1)
						};
						volumes.push_back(v4);

					}

					if (((i % 2 == 1) && (j % 2 == 0) && (k % 2 == 1)) ||
						((i % 2 == 0) && (j % 2 == 1) && (k % 2 == 1)))
					{
						volume_t v0 =
						{
							SpatialToLinearIndex(i + 0, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 1),
							SpatialToLinearIndex(i + 1, j + 1, k + 0),
							SpatialToLinearIndex(i + 0, j + 1, k + 1)
						};
						volumes.push_back(v0);

						volume_t v1 =
						{
							SpatialToLinearIndex(i + 0, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 1)
						};
						volumes.push_back(v1);

						volume_t v2 =
						{
							SpatialToLinearIndex(i + 0, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 0),
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 0, j + 1, k + 1)
						};
						volumes.push_back(v2);

						volume_t v3 =
						{
							SpatialToLinearIndex(i + 0, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 1),
							SpatialToLinearIndex(i + 0, j + 1, k + 1),
							SpatialToLinearIndex(i + 0, j + 0, k + 1)
						};
						volumes.push_back(v3);

						volume_t v4 =
						{
							SpatialToLinearIndex(i + 1, j + 1, k + 0),
							SpatialToLinearIndex(i + 0, j + 1, k + 1),
							SpatialToLinearIndex(i + 1, j + 0, k + 1),
							SpatialToLinearIndex(i + 1, j + 1, k + 1)
						};
						volumes.push_back(v4);

					}
				}
			}
		}

		// Fix the orientations of the elements
		for (int i = 0; i < volumes.size(); i++)
		{
			Vector3f p0 = positions[volumes[i][0]];
			Vector3f p1 = positions[volumes[i][1]];
			Vector3f p2 = positions[volumes[i][2]];
			Vector3f p3 = positions[volumes[i][3]];

			if (Vcl::Geometry::Tetrahedron<float, 3>(p0, p1, p2, p3).computeSignedVolume() < 0)
				std::swap(volumes[i][2], volumes[i][3]);
		}

		return std::make_unique<TetraMesh>(positions, volumes);

		#undef SpatialToLinearIndex
	}

	std::unique_ptr<TriMesh> TriMeshFactory::createSphere(const Vector3f& center, float radius, unsigned int stacks, unsigned int slices, bool inverted)
	{
		using face_t = std::array<unsigned int, 3>;

		size_t nr_vertices = (stacks - 1) * (slices + 1) + 2;
		size_t face_count = ((stacks - 1) * slices) * 2;
		size_t nr_indices = face_count * 3;

		std::vector<Vector3f> positions{ nr_vertices };
		std::vector<Vector3f> normals{ nr_vertices };
		std::vector<face_t>   faces{ face_count };

		// Create the positions
		size_t index = 0;
		float fstacks = static_cast<float>(stacks);
		float fslices = static_cast<float>(slices);
		float pi = Mathematics::pi<float>();

		positions[index].y() = center.y() + radius;
		positions[index].z() = 0;
		positions[index].x() = 0;
		index++;
		for (unsigned int i = 1; i < static_cast<unsigned int>(stacks); ++i)
		{
			float fi = static_cast<float>(i);
			float rad_y = pi / 2.0f - fi / fstacks * pi;
			float sin_y = sin(rad_y);
			float cos_y = cos(rad_y);

			for (unsigned int j = 0; j <= static_cast<unsigned int>(slices); ++j)
			{
				float fj = static_cast<float>(j);
				float rad_xz = fj / fslices * 2.0f * pi;
				float sin_xz = sin(rad_xz - pi);
				float cos_xz = cos(rad_xz - pi);

				positions[index].y() = center.y() + sin_y * radius;
				positions[index].z() = center.z() + sin_xz * cos_y * radius;
				positions[index].x() = center.x() + cos_xz * cos_y * radius;
				++index;
			}
		}
		positions[index].y() = center.y() - radius;
		positions[index].z() = 0;
		positions[index].x() = 0;
		
		// Create the normals
		for (unsigned int i = 0; i < static_cast<unsigned int>(nr_vertices); ++i)
		{
			normals[i] = positions[i] - center;
			normals[i].normalize();
		}

		if (inverted)
		{
			for (unsigned int i = 0; i < static_cast<unsigned int>(nr_vertices); ++i)
			{
				normals[i] *= -1;
			}
		}

		// Create the indices:
		// Deal with the first and the last stack seperately
		unsigned int i0 = inverted ? 1 : 0;
		unsigned int i1 = inverted ? 0 : 1;
		index = 0;
		for (unsigned int j = 0; j < static_cast<unsigned int>(slices); ++j)
		{
			faces[index][0] = 0;
			faces[index][1] = 1 + (j + i1) % slices;
			faces[index][2] = 1 + (j + i0) % slices;
			++index;
		}

		for (unsigned int i = 0; i < static_cast<unsigned int>(stacks) - 2; ++i)
		{
			unsigned int row0_base = 1 + (i + 0) * (static_cast<unsigned int>(slices) + 1);
			unsigned int row1_base = 1 + (i + 1) * (static_cast<unsigned int>(slices) + 1);

			for (unsigned int j = 0; j < static_cast<unsigned int>(slices); ++j)
			{
				faces[index][0] = row0_base + (j + i0) % slices;
				faces[index][1] = row0_base + (j + i1) % slices;
				faces[index][2] = row1_base + (j +  0) % slices;
				++index;

				faces[index][0] = row1_base + (j + i0) % slices;
				faces[index][1] = row0_base + (j +  1) % slices;
				faces[index][2] = row1_base + (j + i1) % slices;
				++index;
			}
		}

		unsigned int row0_base = nr_vertices - 1 - (static_cast<unsigned int>(slices) + 1);
		for (unsigned int j = 0; j < static_cast<unsigned int>(slices); ++j)
		{
			faces[index][0] = row0_base + (j + i0) % slices;
			faces[index][1] = row0_base + (j + i1) % slices;
			faces[index][2] = nr_vertices - 1;
			++index;
		}
		
		return std::make_unique<TriMesh>(positions, faces);
	}
}}
