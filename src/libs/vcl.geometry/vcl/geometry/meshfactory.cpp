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
#include <vcl/math/math.h>

namespace Vcl { namespace Geometry {
	std::unique_ptr<TetraMesh> MeshFactory<TetraMesh>::createHomogenousCubes(unsigned int count_x, unsigned int count_y, unsigned int count_z)
	{
#define SpatialToLinearIndex(a, b, c) ((c) * (xmax + 1) * (ymax + 1) + (b) * (xmax + 1) + (a))

		const unsigned int xmax = count_x;
		const unsigned int ymax = count_y;
		const unsigned int zmax = count_z;

		typedef std::array<unsigned int, 4> volume_t;
		std::vector<Vector3f> positions;
		std::vector<volume_t> volumes;

		positions.resize((xmax + 1) * (ymax + 1) * (zmax + 1));
		for (unsigned int k = 0; k < zmax + 1; k++)
		{
			for (unsigned int j = 0; j < ymax + 1; j++)
			{
				for (unsigned int i = 0; i < xmax + 1; i++)
				{
					positions[SpatialToLinearIndex(i, j, k)] = Vector3f(static_cast<float>(i), static_cast<float>(j), static_cast<float>(k));
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
						volume_t v0 = {
							SpatialToLinearIndex(i + 0, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 1),
							SpatialToLinearIndex(i + 1, j + 1, k + 0),
							SpatialToLinearIndex(i + 0, j + 1, k + 1)
						};
						volumes.push_back(v0);

						volume_t v1 = {
							SpatialToLinearIndex(i + 0, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 1)
						};
						volumes.push_back(v1);

						volume_t v2 = {
							SpatialToLinearIndex(i + 0, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 0),
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 0, j + 1, k + 1)
						};
						volumes.push_back(v2);

						volume_t v3 = {
							SpatialToLinearIndex(i + 0, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 1),
							SpatialToLinearIndex(i + 0, j + 1, k + 1),
							SpatialToLinearIndex(i + 0, j + 0, k + 1)
						};
						volumes.push_back(v3);

						volume_t v4 = {
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
						volume_t v0 = {
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 1),
							SpatialToLinearIndex(i + 0, j + 0, k + 1),
							SpatialToLinearIndex(i + 1, j + 0, k + 0)
						};
						volumes.push_back(v0);

						volume_t v1 = {
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 1),
							SpatialToLinearIndex(i + 1, j + 0, k + 0)
						};
						volumes.push_back(v1);

						volume_t v2 = {
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 0),
							SpatialToLinearIndex(i + 0, j + 0, k + 1),
							SpatialToLinearIndex(i + 0, j + 0, k + 0)
						};
						volumes.push_back(v2);

						volume_t v3 = {
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 1),
							SpatialToLinearIndex(i + 0, j + 1, k + 1),
							SpatialToLinearIndex(i + 0, j + 0, k + 1)
						};
						volumes.push_back(v3);

						volume_t v4 = {
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
						volume_t v0 = {
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 1),
							SpatialToLinearIndex(i + 0, j + 0, k + 1),
							SpatialToLinearIndex(i + 1, j + 0, k + 0)
						};
						volumes.push_back(v0);

						volume_t v1 = {
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 1),
							SpatialToLinearIndex(i + 1, j + 0, k + 0)
						};
						volumes.push_back(v1);

						volume_t v2 = {
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 0),
							SpatialToLinearIndex(i + 0, j + 0, k + 1),
							SpatialToLinearIndex(i + 0, j + 0, k + 0)
						};
						volumes.push_back(v2);

						volume_t v3 = {
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 1),
							SpatialToLinearIndex(i + 0, j + 1, k + 1),
							SpatialToLinearIndex(i + 0, j + 0, k + 1)
						};
						volumes.push_back(v3);

						volume_t v4 = {
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
						volume_t v0 = {
							SpatialToLinearIndex(i + 0, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 1),
							SpatialToLinearIndex(i + 1, j + 1, k + 0),
							SpatialToLinearIndex(i + 0, j + 1, k + 1)
						};
						volumes.push_back(v0);

						volume_t v1 = {
							SpatialToLinearIndex(i + 0, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 1)
						};
						volumes.push_back(v1);

						volume_t v2 = {
							SpatialToLinearIndex(i + 0, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 1, k + 0),
							SpatialToLinearIndex(i + 0, j + 1, k + 0),
							SpatialToLinearIndex(i + 0, j + 1, k + 1)
						};
						volumes.push_back(v2);

						volume_t v3 = {
							SpatialToLinearIndex(i + 0, j + 0, k + 0),
							SpatialToLinearIndex(i + 1, j + 0, k + 1),
							SpatialToLinearIndex(i + 0, j + 1, k + 1),
							SpatialToLinearIndex(i + 0, j + 0, k + 1)
						};
						volumes.push_back(v3);

						volume_t v4 = {
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
		for (size_t i = 0; i < volumes.size(); i++)
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

	std::unique_ptr<TriMesh> TriMeshFactory::createCube(unsigned int count_x, unsigned int count_y, unsigned int count_z)
	{
		VclRequire(count_x > 0, "Resolution in x is greater than 0");
		VclRequire(count_y > 0, "Resolution in y is greater than 0");
		VclRequire(count_z > 0, "Resolution in z is greater than 0");

		using face_t = std::array<unsigned int, 3>;

		std::vector<Vector3f> positions = {
			Vector3f(0, 0, 0),
			Vector3f(1, 0, 0),
			Vector3f(1, 1, 0),
			Vector3f(0, 1, 0),
			Vector3f(0, 1, 1),
			Vector3f(1, 1, 1),
			Vector3f(1, 0, 1),
			Vector3f(0, 0, 1),
		};

		std::vector<face_t> faces = {
			{ 0, 2, 1 }, //face front
			{ 0, 3, 2 },
			{ 2, 3, 4 }, //face top
			{ 2, 4, 5 },
			{ 1, 2, 5 }, //face right
			{ 1, 5, 6 },
			{ 0, 7, 4 }, //face left
			{ 0, 4, 3 },
			{ 5, 4, 7 }, //face back
			{ 5, 7, 6 },
			{ 0, 6, 7 }, //face bottom
			{ 0, 1, 6 }
		};

		auto mesh = std::make_unique<TriMesh>(positions, faces);

		// Create the normals
		Eigen::Vector3f center{ 0.5f, 0.5f, 0.5f };
		auto normals = mesh->addVertexProperty<Vector3f>("Normals", Vector3f{ 0, 0, 0 });
		for (unsigned int i = 0; i < static_cast<unsigned int>(positions.size()); ++i)
			normals[i] = (positions[i] - center).normalized();

		return mesh;
	}

	std::unique_ptr<TriMesh> TriMeshFactory::createSphere(const Vector3f& center, float radius, unsigned int stacks, unsigned int slices, bool inverted)
	{
		using face_t = std::array<unsigned int, 3>;

		unsigned int nr_vertices = (stacks + 1) * (slices + 1);
		unsigned int face_count = (stacks * slices) * 2;

		std::vector<Vector3f> positions{ nr_vertices };
		std::vector<face_t> faces{ face_count };

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
				faces[index][2] = row1_base + (j + 0) % slices;
				++index;

				faces[index][0] = row1_base + (j + i0) % slices;
				faces[index][1] = row0_base + (j + 1) % slices;
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

		auto mesh = std::make_unique<TriMesh>(positions, faces);
		auto normals = mesh->addVertexProperty<Vector3f>("Normals", Vector3f{ 0, 0, 0 });

		// Create the normals
		float sign = inverted ? -1.0f : 1.0f;
		for (unsigned int i = 0; i < static_cast<unsigned int>(nr_vertices); ++i)
		{
			normals[i] = sign * (positions[i] - center).normalized();
		}

		return mesh;
	}

	std::unique_ptr<TriMesh> TriMeshFactory::createArrow(float small_radius, float large_radius, float handle_length, float head_length, unsigned int slices)
	{
		using face_t = std::array<unsigned int, 3>;

		const unsigned int nr_handle_vertices = 3 * slices + 1;
		const unsigned int nr_handle_faces = 3 * slices;

		const unsigned int nr_head_vertices = 3 * slices + 1;
		const unsigned int nr_head_faces = 2 * slices;

		const unsigned int nr_vertices = nr_handle_vertices + nr_head_vertices;
		const unsigned int nr_indices = (nr_handle_faces + nr_head_faces) * 3;

		const float fslices = static_cast<float>(slices);
		const unsigned int stride = static_cast<unsigned int>(slices);

		const float pi = Mathematics::pi<float>();

		std::vector<Vector3f> positions{ nr_vertices };
		std::vector<Vector3f> normals{ nr_vertices };
		std::vector<face_t> faces{ nr_indices / 3 };

		size_t index = 0;

		/*
		 *	Create the handle positions
		 */
		positions[index].setZero();
		normals[index] = { 0, -1, 0 };
		index++;
		for (unsigned int j = 0; j < static_cast<unsigned int>(slices); ++j)
		{
			float fj = static_cast<float>(j);
			float rad_xz = fj / fslices * 2.0f * pi;
			float sin_xz = sin(rad_xz - pi);
			float cos_xz = cos(rad_xz - pi);

			positions[index].x() = cos_xz * small_radius;
			positions[index].y() = 0;
			positions[index].z() = sin_xz * small_radius;
			normals[index] = { 0, -1, 0 };

			positions[index + stride].x() = cos_xz * small_radius;
			positions[index + stride].y() = 0;
			positions[index + stride].z() = sin_xz * small_radius;
			normals[index + stride] = { cos_xz, 0, sin_xz };

			positions[index + 2 * stride].x() = cos_xz * small_radius;
			positions[index + 2 * stride].y() = handle_length;
			positions[index + 2 * stride].z() = sin_xz * small_radius;
			normals[index + 2 * stride] = { cos_xz, 0, sin_xz };

			++index;
		}

		/*
		 *	Create the head positions
		 */
		index += 2 * stride;
		positions[index] = { 0, handle_length, 0 };
		normals[index] = { 0, -1, 0 };
		index++;
		for (unsigned int j = 0; j < static_cast<unsigned int>(slices); ++j)
		{
			float fj = static_cast<float>(j);
			float rad_xz = fj / fslices * 2.0f * pi;
			float sin_xz = sin(rad_xz - pi);
			float cos_xz = cos(rad_xz - pi);

			positions[index].x() = cos_xz * large_radius;
			positions[index].y() = handle_length;
			positions[index].z() = sin_xz * large_radius;
			normals[index] = Vector3f(0, -1, 0);

			Vector3f normal(cos_xz, large_radius, sin_xz);
			normal.normalize();

			positions[index + stride].x() = cos_xz * large_radius;
			positions[index + stride].y() = handle_length;
			positions[index + stride].z() = sin_xz * large_radius;
			normals[index + stride] = normal;

			positions[index + 2 * stride].x() = 0;
			positions[index + 2 * stride].y() = handle_length + head_length;
			positions[index + 2 * stride].z() = 0;
			normals[index + 2 * stride] = normal;

			++index;
		}

		/*
		 *	Create the handle mIndices
		 */
		index = 0;
		for (unsigned int i = 0; i < slices; i++)
		{
			faces[index][0] = 0;
			faces[index][1] = static_cast<uint32_t>(i + 1);
			faces[index][2] = static_cast<uint32_t>((i + 1) % slices + 1);
			index++;
		}
		uint32_t base_index = 1 + static_cast<uint32_t>(stride);
		for (unsigned int i = 0; i < stride; i++)
		{
			faces[index][0] = base_index + i;
			faces[index][1] = base_index + static_cast<uint32_t>(stride) + i;
			faces[index][2] = base_index + static_cast<uint32_t>((i + 1) % slices);
			index++;

			faces[index][0] = base_index + static_cast<uint32_t>(stride) + i;
			faces[index][1] = base_index + static_cast<uint32_t>(stride) + (i + 1) % static_cast<uint32_t>(slices);
			faces[index][2] = base_index + static_cast<uint32_t>((i + 1) % slices);
			index++;
		}

		/*
		 *	Create the head mIndices
		 */
		base_index = static_cast<uint32_t>(nr_handle_vertices);
		for (unsigned int i = 0; i < slices; i++)
		{
			faces[index][0] = base_index;
			faces[index][1] = base_index + i + 1;
			faces[index][2] = base_index + static_cast<uint32_t>((i + 1) % slices + 1);
			index++;
		}
		base_index = nr_handle_vertices + 1 + stride;
		for (unsigned int i = 0; i < stride; i++)
		{
			faces[index][0] = base_index + i;
			faces[index][1] = base_index + static_cast<uint32_t>(stride) + i;
			faces[index][2] = base_index + static_cast<uint32_t>((i + 1) % slices);
			index++;
		}

		auto mesh = std::make_unique<TriMesh>(positions, faces);
		auto normal_prop = mesh->addVertexProperty<Vector3f>("Normals", Vector3f{ 0, 0, 0 });

		for (unsigned int i = 0; i < static_cast<unsigned int>(nr_vertices); ++i)
		{
			normal_prop[i] = normals[i];
		}

		return mesh;
	}

	std::unique_ptr<TriMesh> TriMeshFactory::createTorus(
		float outer_radius,
		float inner_radius,
		unsigned int nr_radial_segments,
		unsigned int nr_sides)
	{
		// The Formula
		// x = Cos(theta) * (radius + ringRadius * Cos(phi))
		// y = Sin(theta) * (radius + ringRadius * Cos(phi))
		// z = ringRadius * Sin(phi)

		using Vcl::Mathematics::pi;
		using Vcl::Mathematics::rad2deg;

		// Default up direction
		Eigen::Vector3f up{ 0, 1, 0 };

		// Define the vertices
		std::vector<Eigen::Vector3f> vertices((nr_radial_segments + 1) * (nr_sides + 1));
		float two_pi = 2.0f * pi<float>();
		for (unsigned int seg = 0; seg <= nr_radial_segments; seg++)
		{
			unsigned int curr_seg = (seg == nr_radial_segments) ? 0 : seg;

			float t1 = (float)curr_seg / nr_radial_segments * two_pi;
			Eigen::Vector3f r1{ std::cos(t1) * outer_radius, 0.0f, std::sin(t1) * outer_radius };

			for (unsigned int side = 0; side <= nr_sides; side++)
			{
				unsigned int curr_side = (side == nr_sides) ? 0 : side;

				float t2 = (float)curr_side / nr_sides * two_pi;
				Eigen::Vector3f r2 = Eigen::AngleAxisf{ -t1, up } * Eigen::Vector3f(cos(t2) * inner_radius, sin(t2) * inner_radius, 0);
				vertices[side + seg * (nr_sides + 1)] = r1 + r2;
			}
		}

		// Define the normales
		std::vector<Eigen::Vector3f> normals(vertices.size());
		for (unsigned int seg = 0; seg <= nr_radial_segments; seg++)
		{
			unsigned int curr_seg = (seg == nr_radial_segments) ? 0 : seg;

			float t1 = (float)curr_seg / nr_radial_segments * two_pi;
			Eigen::Vector3f r1{ cos(t1) * outer_radius, 0.0f, sin(t1) * outer_radius };

			for (unsigned int side = 0; side <= nr_sides; side++)
			{
				normals[side + seg * (nr_sides + 1)] = (vertices[side + seg * (nr_sides + 1)] - r1).normalized();
			}
		}

		// Define UVs
		std::vector<Eigen::Vector2f> uvs(vertices.size());
		for (unsigned int seg = 0; seg <= nr_radial_segments; seg++)
		{
			for (unsigned int side = 0; side <= nr_sides; side++)
			{
				uvs[side + seg * (nr_sides + 1)] = { (float)seg / nr_radial_segments, (float)side / nr_sides };
			}
		}

		// Define triangles
		size_t nr_faces = vertices.size();
		size_t nr_triangles = nr_faces * 2;

		using face_t = std::array<unsigned int, 3>;
		std::vector<face_t> triangles(nr_triangles);

		for (uint32_t i = 0, seg = 0; seg <= nr_radial_segments; seg++)
		{
			for (uint32_t side = 0; side <= nr_sides - 1; side++)
			{
				uint32_t current = side + seg * (nr_sides + 1);
				uint32_t next = side + (seg < (nr_radial_segments) ? (seg + 1) * (nr_sides + 1) : 0);

				if (i < triangles.size() - 6)
				{
					triangles[i][0] = current;
					triangles[i][1] = next + 1;
					triangles[i][2] = next;
					i++;

					triangles[i][0] = current;
					triangles[i][1] = current + 1;
					triangles[i][2] = next + 1;
					i++;
				}
			}
		}

		auto mesh = std::make_unique<TriMesh>(vertices, triangles);
		auto normal_prop = mesh->addVertexProperty<Vector3f>("Normals", Vector3f{ 0, 0, 0 });

		for (unsigned int i = 0; i < static_cast<unsigned int>(vertices.size()); ++i)
		{
			normal_prop[i] = normals[i];
		}

		return mesh;
	}
}}
