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
struct VertexShaderOutput
{
	float4 Color    : COLOR;
	float4 Position : SV_Position;
};

VertexShaderOutput main(uint id : SV_VertexID)
{
	VertexShaderOutput OUT;

	// (0, 1, 2) -> (0, 1, 2)
	// (3, 4, 5) -> (2, 3, 0)
	float2 TexCoord;
	if (id == 0 || id == 5) {
		TexCoord = float2(0, 0);
	}
	else if (id == 1) {
		TexCoord = float2(1, 0);
	}
	else if (id == 2 || id == 3) {
		TexCoord = float2(1, 1);
	}
	else if (id == 4) {
		TexCoord = float2(0, 1);
	}

	OUT.Position = float4(TexCoord * float2(2, -2) + float2(-1, 1), 0, 1);
	OUT.Color = float4(1.0f, 0.0f, 1.0f, 1.0f);

	return OUT;
}
