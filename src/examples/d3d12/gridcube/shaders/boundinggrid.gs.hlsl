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
////////////////////////////////////////////////////////////////////////////////
// Shader Input
////////////////////////////////////////////////////////////////////////////////
struct GSInput
{
	float3 Colour   : COLOR;
	float4 Position : POSITION;
};

////////////////////////////////////////////////////////////////////////////////
// Shader Output
////////////////////////////////////////////////////////////////////////////////
struct GSOutput
{
	float3 Colour   : COLOR;
	float4 Position : SV_Position;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////
void emitLine(in GSInput a, in GSInput b, inout LineStream<GSOutput> OutputStream)
{
	GSOutput Out = (GSOutput)0;

	Out.Colour = a.Colour;
	Out.Position = a.Position;
	OutputStream.Append(Out);

	Out.Colour = a.Colour;
	Out.Position = b.Position;
	OutputStream.Append(Out);

	OutputStream.RestartStrip();
}

[instance(4)]
[maxvertexcount(2)]
void main(uint InstanceID : SV_GSInstanceID, lineadj GSInput In[4], inout LineStream<GSOutput> OutputStream)
{
	emitLine(In[InstanceID], In[(InstanceID + 1) % 4], OutputStream);
}
