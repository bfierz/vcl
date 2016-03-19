#version 430 core

// Data from input-assembler stage
in ivec4 Index;

out VertexData
{
	ivec4 Indices;
} Out;

void main()
{
	// Pass index data to next stage
	Out.Indices = Index;
}
