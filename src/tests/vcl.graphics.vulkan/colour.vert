#version 450 core

layout(location = 0) in vec2 Position;
layout(location = 1) in vec3 Colour;
layout(location = 2) in mat4 Transform;

layout(location = 0) out vec3 OutColour;

void main()
{
	gl_Position = Transform * vec4(Position, 0, 1);
	OutColour = Colour;
}
