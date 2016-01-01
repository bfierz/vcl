#version 400 core

// Input data from last stage
in VertexData
{
	vec4 Position;
	vec3 Normal;
	vec4 Colour;
} In;

// Output data
out vec4 FragColour;

void main(void)
{
	FragColour = In.Colour;
}
