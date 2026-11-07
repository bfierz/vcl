#version 450 core

layout(location = 0) in vec3 InColour;

layout(location = 0) out vec4 Colour;

void main()
{
	Colour = vec4(InColour, 1);
}
