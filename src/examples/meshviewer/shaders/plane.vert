#version 430 core

// Data from input-assembler stage
in vec4 PlaneEquation;

void main()
{
	// Pass data to next stage
	gl_Position = PlaneEquation;
}
