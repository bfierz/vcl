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
struct GeometryShaderInput
{
	int4 Indices : Index;
};

struct GeometryShaderOutput
{
	float4 Color    : COLOR;
	float4 Position : SV_Position;
};

struct ModelViewProjection
{
	#pragma pack_matrix(column_major)
	float4x4 MVP;
};
ConstantBuffer<ModelViewProjection> PerObjectData : register(b0);

StructuredBuffer<float3> tetPositions : register(t0);

void emitPoint(int i, int4 indices, float3 c, inout TriangleStream<GeometryShaderOutput> OutputStream)
{
	GeometryShaderOutput Out = (GeometryShaderOutput)0;

	Out.Color = float4(0.2f, 0.8f, 0.2f, 1.0f);
	Out.Position = mul(PerObjectData.MVP, float4(0.9f*(tetPositions[indices[i]] - c) + c, 1));
	OutputStream.Append(Out);
}

[instance(4)]
[maxvertexcount(3)]
void main(uint InstanceID : SV_GSInstanceID, point GeometryShaderInput In[1], inout TriangleStream<GeometryShaderOutput> OutputStream)
{
	const int tri_indices[] =
	{
		0, 2, 1,
		0, 1, 3,
		1, 2, 3,
		2, 0, 3
	};

	float3 c = float3(0, 0, 0);
	for (int i = 0; i < 4; i++)
		c += tetPositions[In[0].Indices[i]];
	c /= 4;

	emitPoint(tri_indices[3*InstanceID + 0], In[0].Indices, c, OutputStream);
	emitPoint(tri_indices[3*InstanceID + 1], In[0].Indices, c, OutputStream);
	emitPoint(tri_indices[3*InstanceID + 2], In[0].Indices, c, OutputStream);
	OutputStream.RestartStrip();
}
