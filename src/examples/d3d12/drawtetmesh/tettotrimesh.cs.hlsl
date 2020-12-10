/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2020 Basil Fierz
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
cbuffer TetToTriMeshConversionParameters : register(b0)
{
	// Number of tet indices
	uint NrIndices;

	// Scaling of the generated tets
	float Scale;
};

StructuredBuffer<int>      tetTndices   : register(t0);
StructuredBuffer<float3>   tetPositions : register(t1);
RWStructuredBuffer<int>    triIndices   : register(u0);
RWStructuredBuffer<float3> triPositions : register(u1);

groupshared float3 sharedPositions[128 * 1 * 1];

[numthreads(128, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID, uint wave_id : SV_GroupIndex)
{
	if (NrIndices <= thread_id.x)
		return;

	const int tri_indices[] =
	{
		0, 2, 1,
		0, 1, 3,
		1, 2, 3,
		2, 0, 3
	};

	// Four threads collaborate on one tet
	const int tet_idx = thread_id.x / 4;
	const int tet_corner = thread_id.x % 4;

	// Create 4 triangles representing the tet
	const int local_tri_idx = 12*tet_idx + 3*tet_corner;
	triIndices[local_tri_idx + 0] = 4*tet_idx + tri_indices[3*tet_corner+0];
	triIndices[local_tri_idx + 1] = 4*tet_idx + tri_indices[3*tet_corner+1];
	triIndices[local_tri_idx + 2] = 4*tet_idx + tri_indices[3*tet_corner+2];

	// Store the position data for each corner
	const int tet_pos_idx = tetTndices[thread_id.x];
	const float3 tet_pos = tetPositions[tet_pos_idx];
	sharedPositions[wave_id] = tet_pos;
	GroupMemoryBarrierWithGroupSync();

	const int base_wave_id = 4 * (wave_id / 4);
	const float3 center = (
		sharedPositions[base_wave_id + 0] +
		sharedPositions[base_wave_id + 1] +
		sharedPositions[base_wave_id + 2] +
		sharedPositions[base_wave_id + 3]) / 4;

	triPositions[thread_id.x] = center + Scale*(tet_pos - center);
}
