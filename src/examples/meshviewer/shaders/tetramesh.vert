#version 430 core

// Data from input-assembler stage
in ivec4 Index;
in  vec4 Colour;

out VertexData
{
	ivec4 Indices;
} Out;

void main()
{
	// Pass index data to next stage
	Out.Indices = Index;

	// Abuse the position output to pass the colour to the next stage
	gl_Position = Colour;
}
