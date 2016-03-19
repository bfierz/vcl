#version 430 core
#extension GL_ARB_enhanced_layouts : enable

// Data from input-assembler stage
in ivec3 Index;
in  vec4 Colour;

layout(location = 0) out VertexData
{
	ivec3 Indices;
} Out;

void main()
{
	// Pass index data to next stage
	Out.Indices = Index;

	// Abuse the position output to pass the colour to the next stage
	gl_Position = Colour;
}
