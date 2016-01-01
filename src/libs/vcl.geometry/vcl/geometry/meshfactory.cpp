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
}}
