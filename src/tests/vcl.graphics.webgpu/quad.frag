#version 440 core

layout(location = 0) in PerVertexData
{
	vec3 Colour;
} In;

layout(location = 0) out vec4 Colour;

void main()
{	
	Colour = vec4(In.Colour, 1);
}
